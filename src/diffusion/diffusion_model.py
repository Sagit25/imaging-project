import torch
from src.diffusion.utils import save_images


# 요거는 저번학기 다른 수업 프로젝트에서 사용한 파일인데, 그냥 diffusion model을 사용할거라면, 이를 수정해서 사용해도 괜찮을듯?

class DiffusionProcess:
    def __init__(self, stage_1):
        self.stage_1 = stage_1
        self.alphas_cumprod = stage_1.scheduler.alphas_cumprod  

    # Implement forward process of DDPM
    # TODO (Q1.1)
    # ===== your code here! =====
    def forward(self, image, t):
        alpha_t = self.alphas_cumprod[t]
        image_t = torch.sqrt(1-alpha_t) * torch.randn_like(image)
        image_t += torch.sqrt(alpha_t) * image
        return image_t
        
    # ==== end of code ====
    
    # TODO (Q1.3)
    # ===== your code here! =====
    def one_step_denoise(self, im_noisy, prompt_embeds, t):
        # Estimate noise in noisy image
        noise_est = self.stage_1.unet(
            im_noisy.half().cuda(),
            t,
            encoder_hidden_states=prompt_embeds,
            return_dict=False
        )[0]

        # Take only first 3 channels, and move result to cpu
        noise_est = noise_est[:, :3].cpu()
        
        # Esitimate clean_est
        alpha_t = self.alphas_cumprod[t.item()]
        clean_est = im_noisy - torch.sqrt(1-alpha_t) * noise_est.cuda()
        clean_est = clean_est / torch.sqrt(alpha_t)
        return clean_est
        
    # ==== end of code ====

    def set_timesteps(self, timesteps):
        self.stage_1.scheduler.set_timesteps(timesteps=timesteps)    # Need this b/c variance computation

    def add_variance(self, predicted_variance, t, image):
        '''
        Args:
            predicted_variance : (1, 3, 64, 64) tensor, last three channels of the UNet output
            t: scale tensor indicating timestep
            image : (1, 3, 64, 64) tensor, noisy image

        Returns:
            (1, 3, 64, 64) tensor, image with the correct amount of variance added
        '''
        # Add learned variance
        variance = self.stage_1.scheduler._get_variance(t, predicted_variance=predicted_variance)
        variance_noise = torch.randn_like(image)
        variance = torch.exp(0.5 * variance) * variance_noise
        return image + variance

    def iterative_denoise(self, image, i_start, prompt_embeds, timesteps, display=True):
        with torch.no_grad():
            for i in range(i_start, len(timesteps) - 1):
                # Get timesteps
                t = timesteps[i]
                prev_t = timesteps[i+1]

                # Get alphas, betas
                # TODO (Q2):
                # get `alpha_cumprod` and `alpha_cumprod_prev` for timestep t from `alphas_cumprod`
                # compute `alpha`
                # compute `beta`
                # ===== your code here! =====
                alpha_cumprod = self.alphas_cumprod[t]
                beta_cumprod = 1.0 - alpha_cumprod
                alpha_cumprod_prev = self.alphas_cumprod[prev_t]
                beta_cumprod_prev = 1.0 - alpha_cumprod_prev
                alpha = alpha_cumprod / alpha_cumprod_prev
                beta = 1.0 - alpha
                
                # ==== end of code ====

                # Get noise estimate
                model_output = self.stage_1.unet(
                    image,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False
                )[0]

                # Split estimate into noise and variance estimate
                noise_est, predicted_variance = torch.split(model_output, image.shape[1], dim=1)

                # Eq (6) and (7) of DDPM
                # TODO (Q2):
                # compute `pred_prev_image`, the DDPM estimate for the image at the
                # next timestep, which is slightly less noisy. Use the equation for
                # x_{t'} in the notes above.
                # ===== your code here! =====
                # Get x_0
                image0 = image - torch.sqrt(beta_cumprod) * noise_est.cuda()
                image0 = image0 / torch.sqrt(alpha_cumprod)
                # Set x_0 term
                pred_prev_image = image0
                pred_prev_image *= torch.sqrt(alpha_cumprod_prev) * beta
                # Add x_t term
                pred_prev_image_t = image
                pred_prev_image_t *= torch.sqrt(alpha) * beta_cumprod_prev
                pred_prev_image += pred_prev_image_t
                pred_prev_image = pred_prev_image / beta_cumprod
                # Add var term
                pred_prev_image = self.add_variance(predicted_variance, t, pred_prev_image)
                
                # ==== end of code ====

                image = pred_prev_image

                # TODO (Q2):
                # Visualize the noisy image every 5th loop of denoising
                # ===== your code here! =====
                if i % 5 == 0 and display:
                    save_images(image.float().cpu(), f"denoising_iter_{i}")
                
                # ==== end of code ====

        clean = image.detach().cpu()  #.numpy()

        return clean


    # (Skeleton code for Extra Credit)
    def iterative_denoise_cfg(self, image, i_start, prompt_embeds, uncond_prompt_embeds, timesteps, scale=7, display=True):
        with torch.no_grad():
            for i in range(i_start, len(timesteps) - 1):
                # Get timesteps
                t = timesteps[i]
                prev_t = timesteps[i+1]

                # Get alphas, betas
                # ===== your code here! =====

                # TODO:
                # Get `alpha_cumprod`, `alpha_cumprod_prev`, `alpha`, `beta`
                # Feel free to copy code from part 1.4
                
                # Copy code from iterative_denoise()
                alpha_cumprod = self.alphas_cumprod[t]
                beta_cumprod = 1.0 - alpha_cumprod
                alpha_cumprod_prev = self.alphas_cumprod[prev_t]
                beta_cumprod_prev = 1.0 - alpha_cumprod_prev
                alpha = alpha_cumprod / alpha_cumprod_prev
                beta = 1.0 - alpha

                # ==== end of code ====

                # Get cond noise estimate
                model_output = self.stage_1.unet(
                    image,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False
                )[0]

                # Get uncond noise estimate
                uncond_model_output = self.stage_1.unet(
                    image,
                    t,
                    encoder_hidden_states=uncond_prompt_embeds,
                    return_dict=False
                )[0]

                # Split estimate into noise and variance estimate
                noise_est, predicted_variance = torch.split(model_output, image.shape[1], dim=1)
                uncond_noise_est, _ = torch.split(uncond_model_output, image.shape[1], dim=1)

                # Do classifier free guidance
                # ===== your code here! =====

                # TODO:
                # Compute the CFG noise estimate and put it in `model_output`.
                # Hint: use `model_output` and `uncond_model_output`. Should only require
                # one line of code
                noise_cfg = (1 + scale) * noise_est - scale * uncond_noise_est

                # ==== end of code ====

                # Eq (6) and (7) of DDPM
                # ===== your code here! =====

                # TODO:
                # Get `pred_prev_image`, the next less noisy image.
                # Feel free to copy code from part 1.4
                # Show denoised image

                # Copy code from iterative_denoise() and change to noise_cfg
                # Get x_0
                image0 = image - torch.sqrt(beta_cumprod) * noise_cfg.cuda()
                image0 = image0 / torch.sqrt(alpha_cumprod)
                # Set x_0 term
                pred_prev_image = image0
                pred_prev_image *= torch.sqrt(alpha_cumprod_prev) * beta
                # Add x_t term
                pred_prev_image_t = image
                pred_prev_image_t *= torch.sqrt(alpha) * beta_cumprod_prev
                pred_prev_image += pred_prev_image_t
                pred_prev_image = pred_prev_image / beta_cumprod
                # Add var term
                pred_prev_image = self.add_variance(predicted_variance, t, pred_prev_image)

                # ==== end of code ====

                image = pred_prev_image

            clean = image.cpu().detach()

        return clean
    

    def iterative_denoise_cfg_repaint(self, image, image_org, mask, i_start, prompt_embeds, uncond_prompt_embeds, timesteps, scale=7, display=True):
        with torch.no_grad():
            for i in range(i_start, len(timesteps) - 1):
                # Get timesteps
                t = timesteps[i]
                prev_t = timesteps[i+1]

                # Get alphas, betas
                # ===== your code here! =====

                # TODO:
                # Get `alpha_cumprod`, `alpha_cumprod_prev`, `alpha`, `beta`
                # Feel free to copy code from part 1.4
                
                # Copy code from iterative_denoise()
                alpha_cumprod = self.alphas_cumprod[t]
                beta_cumprod = 1.0 - alpha_cumprod
                alpha_cumprod_prev = self.alphas_cumprod[prev_t]
                beta_cumprod_prev = 1.0 - alpha_cumprod_prev
                alpha = alpha_cumprod / alpha_cumprod_prev
                beta = 1.0 - alpha

                # ==== end of code ====

                # Get cond noise estimate
                model_output = self.stage_1.unet(
                    image,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False
                )[0]

                # Get uncond noise estimate
                uncond_model_output = self.stage_1.unet(
                    image,
                    t,
                    encoder_hidden_states=uncond_prompt_embeds,
                    return_dict=False
                )[0]

                # Split estimate into noise and variance estimate
                noise_est, predicted_variance = torch.split(model_output, image.shape[1], dim=1)
                uncond_noise_est, _ = torch.split(uncond_model_output, image.shape[1], dim=1)

                # Do classifier free guidance
                # ===== your code here! =====

                # TODO:
                # Compute the CFG noise estimate and put it in `model_output`.
                # Hint: use `model_output` and `uncond_model_output`. Should only require
                # one line of code
                noise_cfg = (1 + scale) * noise_est - scale * uncond_noise_est

                # ==== end of code ====

                # Eq (6) and (7) of DDPM
                # ===== your code here! =====

                # TODO:
                # Get `pred_prev_image`, the next less noisy image.
                # Feel free to copy code from part 1.4
                # Show denoised image

                # Copy code from iterative_denoise() and change to noise_cfg
                # Get x_0
                image0 = image - torch.sqrt(beta_cumprod) * noise_cfg.cuda()
                image0 = image0 / torch.sqrt(alpha_cumprod)
                # Set x_0 term
                pred_prev_image = image0
                pred_prev_image *= torch.sqrt(alpha_cumprod_prev) * beta
                # Add x_t term
                pred_prev_image_t = image
                pred_prev_image_t *= torch.sqrt(alpha) * beta_cumprod_prev
                pred_prev_image += pred_prev_image_t
                pred_prev_image = pred_prev_image / beta_cumprod
                # Add var term
                pred_prev_image = self.add_variance(predicted_variance, t, pred_prev_image)

                # ==== end of code ====

                # RePaint Logic
                noisy_org = self.forward(image_org, torch.tensor(prev_t).cuda())
                image = mask.cuda() * pred_prev_image.cuda() + (1 - mask).cuda() * noisy_org.cuda()

            clean = image.cpu().detach()

        return clean

