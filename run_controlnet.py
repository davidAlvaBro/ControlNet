from share import *
import config # TODO make a config - also why is it already used? -> config is imported, maybe call it something else.  
import random
from pathlib import Path
import argparse
import json

import cv2
import numpy as np
import einops
import torch

from prepare_labels import get_annotation
from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


def run_controlnet(condition: np.ndarray, gen_path: Path): 
    """
    Given a path to an annotation dataset, a path to the camera parameters, and an output path 
    generate new pose conditioned images from each of these camera views. 
    """
    # TODO load configs and replace the below with those configs 
    # instanciate diffusion model 
    model = create_model(config.model_config_path).cpu()
    model.load_state_dict(load_state_dict(config.model_checkpoint_path, location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    # Convert numpy image to torch, put on GPU, convert to float, and put channels in front
    control = torch.from_numpy(condition).float().cuda().unsqueeze(0) / 255.0
    control = einops.rearrange(control, 'b h w c -> b c h w')# TODO no need to clone right? .clone()
    _, _, cond_h, cond_w = control.shape

    seed = config.seed
    if seed == -1: 
        seed = random.randint(0, 65535)
    seed_everything(seed)
    
    # Prepare prompt 
    positive_prompt = config.prompt + ', ' + config.a_prompt 
    if config.surveillance_view: 
        positive_prompt += config.surveillance_prompt
    
    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)
    
    # prepare control signals 
    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([positive_prompt])]}
    un_cond = {"c_concat": None if config.guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([config.n_prompt])]}
    # shape = (4, config.image_resolution // 8, config.image_resolution // 8)
    shape = (4, cond_h // 8, cond_w // 8)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=True)

    # sample the ddim steps and apply the diffusing unet
    model.control_scales = [config.strength * (0.825 ** float(12 - i)) for i in range(13)] if config.guess_mode else ([config.strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    sample, intermediates = ddim_sampler.sample(config.ddim_steps, config.batch_size,
                                                    shape, cond, verbose=False, eta=config.eta,
                                                    unconditional_guidance_scale=config.scale,
                                                    unconditional_conditioning=un_cond)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)
    # use the vae-decoder and get back to images 
    x_sample = model.decode_first_stage(sample)
    final_image = (einops.rearrange(x_sample, 'b c h w -> b h w c') * 127.5 + 127.5).squeeze().cpu().numpy().clip(0, 255).astype(np.uint8)
    
    cv2.imwrite(str(gen_path), cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

    # No need to do minibatching in the pipeline as I am only interested in one image 
    # final_images = (einops.rearrange(final_images, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)# go through the control conditions in a "minibatch" manner 
    # n_images_to_generate = len(control)
    # surveillance_prompts = ["", surveillance_prompt]*((1+n_images_to_generate)//2)# TODO fix this hard coded 
    # indices = list(range(n_images_to_generate))
    # final_images = torch.zeros((n_images_to_generate, 3, image_resolution, image_resolution), device=control.device)
    # for start in range(0, n_images_to_generate, batch_size):
    #     batch = indices[start:start + batch_size] 
    #     positive_prompts = [prompt + ', ' + a_prompt + surveillance_prompts[i] for i in batch]

    #     if config.save_memory:
    #         model.low_vram_shift(is_diffusing=False)
    #     # prepare control signals 
    #     cond = {"c_concat": [control[batch]], "c_crossattn": [model.get_learned_conditioning(positive_prompts)]}
    #     un_cond = {"c_concat": None if guess_mode else [control[batch]], "c_crossattn": [model.get_learned_conditioning([n_prompt] * batch_size)]}
    #     shape = (4, image_resolution // 8, image_resolution // 8)

    #     if config.save_memory:
    #         model.low_vram_shift(is_diffusing=True)
    #     # sample the ddim steps and apply the diffusing unet
    #     model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    #     samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size,
    #                                                     shape, cond, verbose=False, eta=eta,
    #                                                     unconditional_guidance_scale=scale,
    #                                                     unconditional_conditioning=un_cond)

    #     if config.save_memory:
    #         model.low_vram_shift(is_diffusing=False)
    #     # use the vae-decoder and get back to images 
    #     x_samples = model.decode_first_stage(samples)

    #     final_images[batch] = x_samples

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Apply pose controlnet on the reference image and save it for lifting pipeline.")
    parser.add_argument("--data-dir", default="/data/test", type=str, help="Path to folder where images will be stored in the folder 'images'.")
    args = parser.parse_args()
    data_dir = Path(args.data_dir) 

    anno_path = data_dir / "images"/ "gt_annotation.npz"
    metadata_path = data_dir / "transforms.json"
    out_json_path = data_dir / "controlnet/transforms.json"
    out_imgs_path = data_dir / "controlnet"

    # Get pose annotation 
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    ref_num = metadata["ref"]
    ref_frame = metadata["frames"][ref_num]
    canvas, metadata["frames"][ref_num] = get_annotation(annotations_path=anno_path, frame=ref_frame)
    generated_path = out_imgs_path / metadata["frames"][ref_num]["file_path"]
    generated_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    run_controlnet(condition=canvas, gen_path=generated_path)

    


# TODO beter way of doing the forloop with mini batches: 
# def batched(iterable, n):
#     if n <= 0:
#         raise ValueError("batch size must be > 0")
#     for i in range(0, len(iterable), n):
#         yield iterable[i:i+n]

# for batch in batched(indices, batch_size):
#     # do your diffusion step here with this mini-batch
#     pass