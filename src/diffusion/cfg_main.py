from PIL import Image
from pprint import pprint
from tqdm import tqdm

import torch
import torchvision.transforms.functional as TF
from diffusers import DiffusionPipeline

from huggingface_hub import login
import os
from src.diffusion.utils import seed_everything, save_images
from src.diffusion.diffusion_model import DiffusionProcess

# 요거는 저번학기 다른 수업 프로젝트에서 사용한 파일인데, 그냥 diffusion model을 사용할거라면, 이를 수정해서 사용해도 괜찮을듯?

def main():
    device = 'cuda'
    # Load DeepFloyd IF stage I
    stage_1 = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-I-L-v1.0", # Change to "DeepFloyd/IF-I-M-v1.0" for smaller model
        text_encoder=None,
        variant="fp16",
        torch_dtype=torch.float16,
    )
    stage_1.to(device)

    # Downloading Precomputed Text Embeddings
    prompt_embeds_dict = torch.load('prompt_embeds_dict.pth')

    pprint(list(prompt_embeds_dict.keys()))

    YOUR_SEED = 200
    seed_everything(YOUR_SEED)

    # Sampling from the Model
    # Get scheduler parameters
    diffusion = DiffusionProcess(stage_1)

    # Prompt embeddings for CFG
    prompt_embeds = prompt_embeds_dict["a high quality photo"]
    uncond_prompt_embeds = prompt_embeds_dict[""]

    # Diffusion Model Sampling
    generated = {}

    strided_timesteps = [i for i in range(990, -30, -30)]
    diffusion.set_timesteps(strided_timesteps)
    prompt_embeds=prompt_embeds.cuda()
    uncond_prompt_embeds=uncond_prompt_embeds.cuda()
    

    # Problem (a)
    for i in range(5):
        noise = torch.randn((1, 3, 64, 64)).half().to(device)
        denoise = diffusion.iterative_denoise_cfg(image=noise, i_start=0, prompt_embeds=prompt_embeds,
                                            uncond_prompt_embeds=uncond_prompt_embeds,
                                            timesteps=strided_timesteps, scale=7.0, display=False)
        save_images(denoise.cpu().float(), f"cfg_sample_image_{i}")
        generated[f"cfg_sample_image_{i}"] = denoise
    print("Finished problem(a)!")


    # Problem (b)
    # Get image list
    image_list = ['campanile.jpg', 'star.jpg']

    for image in image_list:
        # For stage 1: Resize to (64, 64), convert to tensor, rescale to [-1, 1], and
        # add a batch dimension. The result is a (1, 3, 64, 64) tensor
        test_im = Image.open(image).resize((64, 64))
        test_im = TF.to_tensor(test_im)
        test_im = 2 * test_im - 1
        test_im = test_im[None]

        # Show test image
        save_images(test_im, f"test_im_{image}")

        # Set starting index
        starting_index = [1, 3, 5, 7, 10, 20]
        
        for i_start in starting_index:
            t = strided_timesteps[i_start]
            
            # Add noise
            im_noisy = diffusion.forward(test_im, t).half().to(device)
            save_images(im_noisy.cpu(), f"noise_{t}_image_start_{i_start}_{image}")
            
            # Denoising
            denoise = diffusion.iterative_denoise_cfg(image=im_noisy, i_start=i_start, prompt_embeds=prompt_embeds,
                                                uncond_prompt_embeds=uncond_prompt_embeds,
                                                timesteps=strided_timesteps, scale=7.0, display=False)
            save_images(denoise, f"cfg_sdedit_start_{i_start}_{image}")
    print("Finished problem(b)!")


    # Prob (c)
    # Generate mask
    mask = torch.zeros_like(test_im, dtype=torch.float16).to(device)
    mask[:, :, :20, :] = 1.0
    save_images(mask.float(), "repaint_mask")

    # Test image
    test_im = Image.open('campanile.jpg').resize((64, 64))
    test_im = TF.to_tensor(test_im)
    test_im = 2 * test_im - 1
    test_im = test_im[None].half().to(device)
    save_images(test_im.cpu().float(), "repaint_original")
    
    # Generate noisy image
    # Fill inside of mask to noise
    t = strided_timesteps[0]
    noise_paint = mask * torch.randn_like(test_im, dtype=torch.float16).to(device)
    im_noisy = noise_paint + (1 - mask) * test_im
    save_images(im_noisy.cpu().float(), "repaint_initial")

    # Denoising Logic
    denoise = diffusion.iterative_denoise_cfg_repaint(image=im_noisy, image_org=test_im, mask=mask, 
                                                      i_start=0, prompt_embeds=prompt_embeds,
                                                      uncond_prompt_embeds=uncond_prompt_embeds,
                                                      timesteps=strided_timesteps, scale=7.0, display=False)
    save_images(denoise.float(), "repaint_result")
    print("Finished problem(c)!")


if __name__ == "__main__":
    # Login to Hugging Face Hub for DeepFloyd
    # TODO:
    token = 'token'
    login(token=token)
    os.makedirs('outputs', exist_ok=True)
    main()

