import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
import os
import wandb
from Diffusion_Scheduler import Diffusion_Scheduler
from model import UNet
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.init(
    # set the wandb project where this run will be logged
    project="mnist-diffusion-model",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0001,
    "architecture": "UNET",
    "dataset": "MNIST",
    "batch_size" : 32,
    "latent_dims" : 512,
    "epochs": 100,
    }
)

def tensor_to_image(tensor):
    tensor = tensor.squeeze(0)
    tensor = tensor * 0.5 + 0.5
    tensor = tensor.clamp(0, 1)
    return T.ToPILImage()(tensor)

def show_images(noised_image, noise, actual_image, noisedActualImage):
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))

    axs[0].imshow(tensor_to_image(noised_image))
    axs[0].set_title('Noised image')
    axs[0].axis('off')

    axs[1].imshow(tensor_to_image(noise))
    axs[1].set_title('Noise added at step')
    axs[1].axis('off')

    axs[2].imshow(tensor_to_image(actual_image))
    axs[2].set_title('Actual Color Image')
    axs[2].axis('off')

    axs[3].imshow(tensor_to_image(noisedActualImage))
    axs[3].set_title('Actual Color Image')
    axs[3].axis('off')

    plt.show()

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
# parent_dir = os.path.dirname(project_dir)
image_dir = os.path.join(project_dir, "data/color")

class CustomImageData(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        images = os.path.join(self.image_dir, self.image_files[idx])

        im = Image.open(images).convert('RGB')  # Convert to RGB

        im = im.resize((128, 128), Image.Resampling.LANCZOS)

        if self.transform:
            im = self.transform(im)
        
        return im

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5]),
])

# color_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# ])

train_dataset = datasets.MNIST(root='./data', train = True, download = True, transform = transform)
test_dataset = datasets.MNIST(root='./data', train = False, download = True, transform = transform)
batch_size = 128

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_channels = 1
output_channels = 1
time_embedding_dims = 256
hidden_dims = [64, 64, 128, 256, 256]

Network = UNet(input_channels, output_channels, hidden_dims, time_embedding_dims).to(device)

v_begin = 1e-4
v_end = 0.02
steps = 1000

diffusion = Diffusion_Scheduler(v_begin, v_end, steps , cosine = True)
# Define optimizer
optimizer = torch.optim.Adam(Network.parameters(), lr= 1e-4)

# Training loop
epochs = 100

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
best_loss = float('inf')

for epoch in range(epochs):
    Network.train()
    train_loss = 0

    for images, _ in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        images = images.to(device)

        batch_size, num_channels, w, h = images.shape
        rand_steps = torch.randint(1, 512, (batch_size,)).to(device)


        optimizer.zero_grad()
        
        with autocast():
            noise, new_image = diffusion.add_noise(rand_steps, images)

            # Forward pass
            predicted_noise = Network(new_image, rand_steps)
        
            # Compute loss
            loss = Network.find_loss(predicted_noise, noise)
        
        loss.backward()
        optimizer.step()

        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(Network.parameters(), max_norm=1.0)
        # scaler.step(optimizer)
        # scaler.update()

        train_loss += loss.item()
    train_loss_avg = train_loss / len(train_dataloader)
    wandb.log({"Train Loss": train_loss_avg})
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss_avg}')

    # Validation loop
    Network.eval()
    test_loss = 0
    with torch.no_grad():
        for images, _ in test_dataloader:
            images = images.to(device)
            batch_size, num_channels, w, h = images.shape

            rand_steps = torch.randint(1, 512, (batch_size,)).to(device)
            noise, new_image = diffusion.add_noise(rand_steps, images)

            predicted_noise = Network(new_image, rand_steps)

            loss = Network.find_loss(predicted_noise, noise)
            test_loss += loss.item()

    test_loss_avg = test_loss / len(test_dataloader)
    wandb.log({"Test Loss": test_loss_avg})
    print(f'Epoch {epoch+1}/{epochs}, Test Loss: {test_loss/len(test_dataloader)}')


    scheduler.step(test_loss)

    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(Network.state_dict(), 'best_model.pth')

    if epoch % 2 == 0:  # Log images every 2 epochs
        random_noise = torch.randn((1, input_channels, 32, 32)).to(device)
        
        one_step = Network(random_noise, torch.tensor([1]).to(device))
        wandb.log({"Single step" : [wandb.Image(tensor_to_image(one_step), caption=f"Noisy Image")]})
        denoised_image = diffusion.generate(model = Network, input_channels=1)

        # Convert the initial noisy image to a format suitable for wandb
        noisy_image = wandb.Image(tensor_to_image(random_noise), caption="Initial Noisy Image")

        # Create a list to store all denoising steps as wandb.Image objects
        # denoising_steps = []
        # for i, img in enumerate(image_list):
        #     denoised = wandb.Image(tensor_to_image(img), caption=f"Denoising Step {i+1}")
        #     denoising_steps.append(denoised)

        # Add the final denoised image
        final_denoised = wandb.Image(tensor_to_image(denoised_image), caption="Final Denoised Image")
        # denoising_steps.append(final_denoised)

        # Log all images to wandb
        # wandb.log({
        #     "Denoising Process": [noisy_image] + denoising_steps,
        #     "Epoch": epoch
        # })

        wandb.log({"Final step" : [wandb.Image(tensor_to_image(denoised_image), caption=f"Denoised Image")]})

# Save the model
model_path = os.path.join(project_dir, 'UNET_Model2.pth')
torch.save(Network.state_dict(), model_path)
print(f'Model saved to {model_path}')