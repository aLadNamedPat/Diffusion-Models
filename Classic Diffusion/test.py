from model import UNet
from Diffusion_Scheduler import Diffusion_Scheduler
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import imageio

def tensor_to_image(tensor):
    tensor = tensor.squeeze(0)
    tensor = tensor * 0.5 + 0.5
    tensor = tensor.clamp(0, 1)
    return T.ToPILImage()(tensor)

def show_images(denoised_images):
    fig, axs = plt.subplots(1, len(denoised_images), figsize=(10, 5))

    for i in range(len(denoised_images)):
        axs[i].imshow(tensor_to_image(denoised_images[i]))
        axs[i].axis('off')

    plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Used for generating images after training
v_begin = 1e-4
v_end = 0.02
steps = 1000

# diffusion = Diffusion_Scheduler(v_begin, v_end, steps)

# input_channels = 1
# output_channels = 1
# time_embedding_dims = 256
# hidden_dims = [64, 64, 128, 256, 256]

# Network = UNet(input_channels, output_channels, hidden_dims, time_embedding_dims).to(device)
# Network.load_state_dict(torch.load("Best_1000_Step.pth"))
# Network.eval()

# denoised_images = []
# for i in range(5):
#     random_noise = torch.randn((1, input_channels, 32, 32)).to(device)
    
#     one_step = Network(random_noise, torch.tensor([1]).to(device))
#     denoised_image = diffusion.generate(model = Network, input_channels=1, num_inference_steps=1000)
#     denoised_images.append(denoised_image)

# show_images(denoised_images)

diffusion = Diffusion_Scheduler(v_begin, v_end, steps)

input_channels = 1
output_channels = 1
time_embedding_dims = 256
hidden_dims = [64, 64, 128, 256, 256]

Network = UNet(input_channels, output_channels, hidden_dims, time_embedding_dims).to(device)
Network.load_state_dict(torch.load("Best_1000_Step.pth"))
Network.eval()

denoised_images = []
for i in range(5):
    images = diffusion.gifGenerator(model=Network, input_channels=input_channels, num_inference_steps=1000)
    denoised_images.append(images)

# Save GIF
for idx, images in enumerate(denoised_images):
    gif_path = f"denoised_images_{idx}.gif"
    imageio.mimsave(gif_path, images, fps=5)
    print(f"Saved GIF: {gif_path}")

# Show images (optional)
def show_images(denoised_images):
    fig, axs = plt.subplots(1, len(denoised_images), figsize=(10, 5))

    for i in range(len(denoised_images)):
        axs[i].imshow(tensor_to_image(denoised_images[i][-1]))
        axs[i].axis('off')

    plt.show()

show_images(denoised_images)
