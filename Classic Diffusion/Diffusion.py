import torch

class Diffusion():
    # Initialize the diffusion process
    def __init__(
        self,
        image_size : tuple[int, int], # Define the size of the image that noise is being added to
        v_begin : float, #Defining variance schedule beginning
        v_end : float, #Defining variance schedule ending
        steps : int, #Define the number of steps of noising that will be undertaken
    ) -> None:
        self.image_size = image_size
        self.schedule = torch.linspace(v_begin, v_end, steps = steps)
        self.steps = steps
        self.curr_step = 0

        self.alpha = 1 - self.schedule
        self.cumulAlpha = torch.cumprod(self.alpha)

    # Add noise to the sampling process
    def add_noise(
        self,
        step : int, #Timestep to generate noise to
        noisy_image : torch.tensor, #The image that noise is being added to
    ) -> torch.tensor:
        self.curr_step += 1

        noised_image = torch.normal(torch.sqrt(self.cumulAlpha[step]) * noisy_image, (1 - self.cumulAlpha[step])) # Find the noisy image after using the normal

        noise_generated = noised_image - noisy_image #Find the noise that was added to the image

        return noise_generated, noised_image
    

    # Sample and return the partially denoised image
    def sample(
        self,
        noisy_image : torch.tensor, #Original noisy image
        model, #Model that is doing the denoising
    ) -> torch.tensor:
        
        for i in reversed(range(self.steps)):
            z = torch.normal(0, torch.ones(noisy_image.shape))
            denoised_image = 1 / self.alpha[i] * (noisy_image  - (1 - self.alpha[i]) / (torch.sqrt(1 - self.cumulAlpha[i]) * model(noisy_image, i)))
            if i == 1:
                return denoised_image
            denoised_image += self.schedule(i) * z