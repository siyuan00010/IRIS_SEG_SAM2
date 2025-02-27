import os
import hydra
import torch
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import sam2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from training.model.sam2 import SAM2Train  # model class of Train/Finetune
from training import trainer, loss_fns  # train configuration & Loss functions
from sam2.sam2_image_predictor import SAM2ImagePredictor

class Config(object):
    def __init__(self):
        self.dataset_root = '/Users/Downloads/dataset'
        self.img_size = 1024
        self.images_path = os.path.join(self.dataset_root, 'images')
        self.masks_path = os.path.join(self.dataset_root, 'masks')
        self.val_images_path = os.path.join(self.dataset_root, 'val_images')
        self.val_masks_path = os.path.join(self.dataset_root, 'val_masks')
        self.pretrained_sam_path = 'sam2_hiera_s.yaml'
        self.checkpoint_save = 'checkpoints'
        self.output_dir = "SAM_predictions"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train = False

cfg = Config()
hydra.core.global_hydra.GlobalHydra.instance().clear()
# reinit hydra with a new search path for configs
hydra.initialize_config_module(cfg.pretrained_sam_path, version_base='1.2')
# Load the pretrained SAM model
sam2 = build_sam2(cfg.pretrained_sam_path, cfg.checkpoint_save, device=cfg.device)
sam2.eval()  # Set the model to evaluation mode

""" Data Loader"""

class CustomSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load the image
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(image_path).convert("L")  # Convert to grayscale (1 channel)

        # Load the mask
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx].replace('.jpg', '_mask.png'))  # Adjust for your mask filename format
        mask = Image.open(mask_path).convert("RGB") 

        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)
            mask = np.array(mask)
            mask = torch.tensor(mask, dtype=torch.long)  # Convert mask to tensor

        return image, mask

# Define any data augmentation or normalization you'd like
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# Create Dataset and DataLoader
dataset = CustomSegmentationDataset(cfg.images_path, cfg.masks_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Create validation Dataset and DataLoader
val_dataset = CustomSegmentationDataset(cfg.val_images_path,cfg. val_masks_path, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

"""##Fine Tune"""
predictor = SAM2ImagePredictor(build_sam2(cfg.pretrained_sam_path, cfg.checkpoint, device=cfg.device))
predictor.model.sam_mask_decoder.train(True) # enable training of mask decoder
predictor.model.sam_prompt_encoder.train(True) # enable training of prompt encoder

# Set the model to training mode
sam2.train()

# Optimizer (use Adam, SGD, or another optimizer as needed)
optimizer = optim.Adam(sam2.parameters(), lr=1e-5)

# Loss function (use DiceLoss or CrossEntropyLoss for segmentation)
loss_fn = nn.CrossEntropyLoss()

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for images, masks in dataloader:
        optimizer.zero_grad()

        # Forward pass through SAM
        outputs = sam2(images)  # Modify based on SAM API and how the input is passed

        # Calculate loss
        loss = loss_fn(outputs, masks)  # Modify this if using custom segmentation output
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")

# Set the model to evaluation mode
sam2.eval()

outputs = []
# Run the evaluation on a validation set
with torch.no_grad():
    for images, masks in val_dataloader:  # Assume val_dataloader is defined similarly to dataloader
        mask_generator = SAM2AutomaticMaskGenerator(sam2)
        for image in images:
            output = mask_generator.generate(image)
            outputs.append(output)
        # Calculate evaluation metrics here (IoU, Dice, etc.)
        
torch.save(sam2.state_dict(), cfg.checkpoint_save)
sam2.eval()

# Example inference
image = Image.open("new_image.jpg").convert("RGB")
image = transform(image).unsqueeze(0)  # Add batch dimension
image = np.array(image)

with torch.no_grad():
    mask_generator = SAM2AutomaticMaskGenerator(sam2)  # Modify based on SAM's API
    predicted_mask = mask_generator.generate(image)

# Visualize or save the predicted mask
predicted_mask = predicted_mask.squeeze().numpy()  # Remove batch dimension and convert to numpy
predicted_mask = Image.fromarray(predicted_mask)
predicted_mask.show()
predicted_mask.save("predicted_mask.png")