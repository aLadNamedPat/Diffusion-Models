import torch
import math
import numpy as np
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Diffusion_Scheduler():
    # Initialize the diffusion process
    def __init__(
        self,
        # image_size : tuple[int, int], # Define the size of the image that noise is being added to
        v_begin : float, #Defining variance schedule beginning
        v_end : float, #Defining variance schedule ending
        steps : int, #Define the number of steps of noising that will be undertaken
        cosine : bool = True,
    ) -> None:

        self.steps = steps

        if not cosine:
            # self.image_size = image_size
            self.schedule = torch.linspace(v_begin, v_end, steps = steps).to(device)
            # self.curr_step = 0
            self.beta = self.schedule
            self.alpha = 1 - self.schedule
            self.cumulAlpha = torch.cumprod(self.alpha, dim = 0).to(device)
        
        else:
            # https://www.zainnasir.com/blog/cosine-beta-schedule-for-denoising-diffusion-models/
            # https://github.com/tonyduan/diffusion/blob/main/src/simple/diffusion.py
            lin_space = torch.linspace(0, steps, steps=steps + 1).to(device)
            cumulAlpha = torch.cos(((lin_space / steps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
            self.cumulAlpha = cumulAlpha / cumulAlpha[0]
            self.beta = torch.zeros_like(self.cumulAlpha)
            self.beta[1:] = 1 - (self.cumulAlpha[1:] / self.cumulAlpha[:-1]).clamp(min=0.0, max=0.999)
            self.alpha = 1 - self.beta[1:]
            self.cumulAlpha = torch.cumprod(self.alpha, dim=0).to(device)
    # Add noise to the training process
    def add_noise(
        self,
        step : torch.Tensor, #Timestep to generate noise to
        clean_image : torch.Tensor, #The image that noise is being added to
    ) -> torch.Tensor:
        # self.curr_step += 1
        cumulAlphaSqrt = torch.sqrt(self.cumulAlpha[step]).view(-1, 1, 1, 1)
        new_mean = cumulAlphaSqrt * clean_image
        alpha_std = torch.sqrt(1 - self.cumulAlpha[step]).view(-1, 1, 1, 1)
        noise = torch.randn_like(clean_image)
        noise_generated = noise * alpha_std # Find the noisy image after using the normal

        noised_image = new_mean + noise_generated #Find the noise that was added to the image

        return noise, noised_image
    
    # # Sample and return the partially denoised image
    # def sample(
    #     self,
    #     noisy_image : torch.tensor, #Original noisy image
    #     model, #Model that is doing the denoising
    # ) -> torch.tensor:
    #     with torch.no_grad():
    #         for i in reversed(range(self.steps)):
    #             # print(i)
    #             z = torch.randn_like(noisy_image)
    #             denoised_image = 1 / self.alpha[i] * (noisy_image  - (1 - self.alpha[i]) / (torch.sqrt(1 - self.cumulAlpha[i]) * model(noisy_image, torch.tensor([i]).to(device))))
    #             if i == 0:
    #                 return denoised_image
    #             noisy_image = denoised_image + torch.sqrt(self.schedule[i]) * z
    #     return denoised_image

    # def sample(
    #     self,
    #     noisy_image: torch.Tensor,  # Original noisy image
    #     model,  # Model that is doing the denoising
    # ) -> torch.Tensor:
    #     with torch.no_grad():
    #         for i in reversed(range(self.steps)):
    #             alpha = self.alpha[i]
    #             cumulAlpha = self.cumulAlpha[i]
    #             timestep = torch.tensor([i]).to(device)
                
    #             predicted_noise = model(noisy_image, timestep)
    #             denoised_image = (noisy_image - torch.sqrt(1 - cumulAlpha) * predicted_noise) / torch.sqrt(cumulAlpha)
                
    #             if i > 0:
    #                 noise = torch.randn_like(noisy_image)
    #                 sigma = torch.sqrt(self.schedule[i-1] - self.schedule[i] * (cumulAlpha[i-1] / cumulAlpha))
    #                 noisy_image = torch.sqrt(alpha) * denoised_image + sigma * noise
    #             else:
    #                 return denoised_image
                

    def sample(
        self, 
        noisy_image, 
        model
    ):
        with torch.no_grad():
            x_list = []
            x = noisy_image
            for i in reversed(range(self.steps)):
                z = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
                alpha = self.alpha[i]
                alpha_cumul = self.cumulAlpha[i]
                
                predicted_noise = model(x, torch.tensor([i]).to(device))
                
                if i > 0:
                    sigma = torch.sqrt(self.beta[i] * (1. - alpha_cumul) / (1. - self.cumulAlpha[i-1]))
                else:
                    sigma = 0

                x = 1. / torch.sqrt(alpha) * (x - self.beta[i] / torch.sqrt(1 - alpha_cumul) * predicted_noise) + sigma * z

                x_list.append(x)
            return x, x_list
        
    # Taken directly from https://github.com/filipbasara0/simple-diffusion/blob/main/simple_diffusion/scheduler/ddim.py
    def take_step(
        self, 
        x_0, 
        timestep, 
        sample
    ):
        previous_timestep = timestep - self.steps // len(self.timesteps)
        cumulAlpha = self.cumulAlpha[timestep]

        if previous_timestep >= 0:
            cumulAlphaPrev = self.cumulAlpha[previous_timestep]
        else:
            cumulAlphaPrev = self.cumulAlpha[0]

        beta_prod_t = 1 - cumulAlpha

        pred_original_sample = (sample - beta_prod_t ** 0.5 * x_0) / cumulAlpha ** 0.5

        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        variance = (1 - cumulAlphaPrev) / (1 - cumulAlpha) * (1 - cumulAlpha / cumulAlphaPrev)

        std_dev = variance ** 0.5

        pred_sample_direction = (1 - cumulAlphaPrev - std_dev**2) ** 0.5 * x_0

        prev_sample = cumulAlphaPrev**0.5 * pred_original_sample + pred_sample_direction

        noise = torch.randn(x_0.shape).to(device)
        prev_sample += std_dev * noise

        return prev_sample
    
    def generate(
        self,
        model,
        input_channels : int = 1,
        num_inference_steps : int = 100,
    ):
        with torch.no_grad():
            image = torch.randn((1, input_channels, 32, 32)).to(device)
            self.set_timesteps(num_inference_steps)
            for t in tqdm(self.timesteps):

                model_output = model(image, torch.tensor([t]).to(device))
                # predict previous mean of image x_t-1 and add variance depending on eta
                # do x_t -> x_t-1
                image = self.take_step(model_output, t, image)

        return image
    
    def set_timesteps(
        self,
        timesteps : int,
    ):
        self.timesteps = (
            np.arange(
                0,
                self.steps,
                self.steps //  timesteps,
            )[::-1]
        )