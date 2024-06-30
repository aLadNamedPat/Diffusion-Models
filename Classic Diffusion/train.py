import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
import wandb
from Diffusion_Scheduler import Diffusion_Scheduler
from model import UNet
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0005,
    "architecture": "UNET",
    "dataset": "Landscape-Color",
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
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

color_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

dataset = CustomImageData(image_dir=image_dir, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

batch_size = 16

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_channels = 3
output_channels = 3
time_embedding_dims = 100
hidden_dims = [64, 64, 128, 256, 256]

Network = UNet(input_channels, output_channels, hidden_dims, time_embedding_dims).to(device)

v_begin = 1e-4
v_end = 0.02
steps = 100

diffusion = Diffusion_Scheduler(v_begin, v_end, steps)
# Define optimizer
optimizer = torch.optim.Adam(Network.parameters(), lr= 5e-4)

# Training loop
epochs = 100

for epoch in range(epochs):
    Network.train()
    train_loss = 0

    for images in train_dataloader:
        images = images.to(device)

        batch_size, num_channels, w, h = images.shape
        rand_steps = torch.randint(1, 100, (batch_size,)).to(device)


        optimizer.zero_grad()
        
        noise, new_image = diffusion.add_noise(rand_steps, images)

        # Forward pass
        reconstructed = Network(images, rand_steps)
        
        # Compute loss
        loss = Network.find_loss(reconstructed, noise)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss_avg = train_loss / len(train_dataloader)
    wandb.log({"Train Loss": train_loss_avg})
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss_avg}')

    # Validation loop
    Network.eval()
    test_loss = 0
    with torch.no_grad():
        for images in test_dataloader:
            images = images.to(device)
            batch_size, num_channels, w, h = images.shape

            rand_steps = torch.randint(1, 100, (batch_size,)).to(device)
            noise, new_image = diffusion.add_noise(rand_steps, images)

            reconstructed = Network(images, rand_steps)

            loss = Network.find_loss(reconstructed, noise)
            test_loss += loss.item()
    test_loss_avg = test_loss / len(test_dataloader)
    wandb.log({"Test Loss": test_loss_avg})
    print(f'Epoch {epoch+1}/{epochs}, Test Loss: {test_loss/len(test_dataloader)}')


    if epoch % 10 == 0:  # Log images every 10 epochs
        random_noise = torch.randn((1, input_channels, 128, 128)).to(device)
        images_list = [random_noise]

        for step in reversed(range(diffusion.steps)):
            random_noise = diffusion.sample(random_noise, Network)
            images_list.append(random_noise.cpu())

        images_list = [tensor_to_image(img) for img in images_list]
        wandb.log({"Denoising Steps": [wandb.Image(img, caption=f"Step {i}") for i, img in enumerate(images_list)]})

# Save the model
model_path = os.path.join(project_dir, 'UNET_Model2.pth')
torch.save(Network.state_dict(), model_path)
print(f'Model saved to {model_path}')