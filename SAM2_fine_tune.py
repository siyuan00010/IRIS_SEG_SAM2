import os
import torch
import cv2
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import random
import torchvision.transforms.functional as TF
class arguments:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config(object):
  """Change the args in the "args" variable to change the config"""

  def __init__(self):
    args = arguments(
        # root path
        root_path = '',
        # json file path
        json_file_path = '',
        # data path
        data_path = '',
        # number of classes
        num_classes = 10,
        img_size = 512,
        # model
        model='SAM2',
        checkpoint_dir='checkpoints',
        output_dir="SAM2_output",
        test_input_dir="test/input",
        test_output_dir="test/output",
        test_size=0.15,
        val_size=0.15,
        train=True
    )
    self.model = args.model
    self.checkpoint_dir = args.checkpoint_dir
    self.output_dir = args.output_dir
    self.test_input_dir = args.test_input_dir
    self.test_output_dir = args.test_output_dir
    self.train = args.train
    self.num_classes = args.num_classes
    self.img_size = args.img_size
    self.train_size = args.train_size
    self.test_size = args.test_size
    self.data_path = args.data_path
    self.json_file_path = args.json_file_path
    self.root_path = args.root_path
cfg = Config()
# Load data
class IrisDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, transform = None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = os.listdir(images_dir)
        self.masks = os.listdir(masks_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        # read image
        image = cv2.imread(os.path.join(self.images_dir, img_path), cv2.IMREAD_GRAYSCALE)
        # read mask
        mask = cv2.imread(os.path.join(self.masks_dir, mask_path), 1)  # read RGB

        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample

# # calculate class distributions
# from sklearn.model_selection import train_test_split

# full_dataset = IrisDataset(cfg.images_dir, cfg.masks_dir)
# # Split indices while preserving class balance
# indices = list(range(len(full_dataset)))
# train_indices, test_indices = train_test_split(
#     indices, 
#     test_size=0.3, 
#     stratify=class_labels,  # Approximated from masks
#     random_state=42
# )
# train_indices, val_indices = train_test_split(
#     train_indices, 
#     test_size=0.15/0.7,  # 15% of full dataset
#     stratify=class_labels[train_indices],
#     random_state=42
# )
# transform
class ToTensorAndNormalize:
    def __init__(self, img_size):
        self.img_size = img_size  # Set the desired image size
        # self.random_flip = transforms.RandomHorizontalFlip(0.5)

    def __call__(self, sample):

        image, mask = sample['image'], sample['mask']

        # # Convert to PIL Image if not already
        # if not isinstance(image, Image.Image):
        #     image = Image.fromarray(image)

        # get image width, height, and channels
        width,height,channel = image.shape
        # get mask width, height, and channels
        mask_width,mask_height,mask_channel = mask.shape
        
        # convert to tensor
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        # resize image and mask
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))

        # normalize
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Apply the same spatial augmentations to both image and mask
        if random.random() > 0.5:
            # Random horizontal flip
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if random.random() > 0.5:
            # Random vertical flip
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        if random.random() > 0.5:
            # Random rotation (-15 to +15 degrees)
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
        
        # Color jitter (only on the image)
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        image = TF.adjust_brightness(image, brightness)
        image = TF.adjust_contrast(image, contrast)
  
        if image is None:
            raise ValueError("Error loading image. Check the file path.")

        ### histogram equalization

        # Apply Gaussian Blur to reduce noise while preserving edges
        blurred_image = cv2.GaussianBlur(image, (1, 1), 0)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced_image = clahe.apply(blurred_image)

        # # Convert back to PIL Image for compatibility
        # return Image.fromarray(enhanced_image)
        return {'image': enhanced_image, 'mask': mask}

transform = ToTensorAndNormalize(img_size=cfg.img_size)

# Create the datasets
def return_dataset(image_dirs, mask_dirs, transform=None):
    # Loop through each pair of image and mask directories
    for img_dir, msk_dir in zip(image_dirs, mask_dirs):
        # Debug print to check the number of files in each directory
        print(f"Images in {img_dir}: {len(os.listdir(img_dir))}")
        print(f"Masks in {msk_dir}: {len(os.listdir(msk_dir))}")

        if len(os.listdir(img_dir)) == 0:
            print(f"Warning: {img_dir} is empty, skipping.")
            continue
    # Load iris dataset
    full_dataset = IrisDataset(cfg.images_dir, cfg.masks_dir)

    # Split into train/val/test
    len = len(full_dataset)
    train_size = 1 - cfg.test_size - cfg.val_size

    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size*len, cfg.val_size*len, cfg.test_size*len],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    )

    # Apply transforms ONLY to the training subset
    train_dataset = IrisDataset(
        images=train_subset.dataset.images[train_subset.indices],  # Directly access subset data
        masks=train_subset.dataset.masks[train_subset.indices],
        transform=transform  # Apply augmentations
    )

    # Validation and test sets (no transforms)
    val_dataset = IrisDataset(
        images=val_subset.dataset.images[val_subset.indices],
        masks=val_subset.dataset.masks[val_subset.indices],
        transform=None
    )

    test_dataset = IrisDataset(
        images=test_subset.dataset.images[test_subset.indices],
        masks=test_subset.dataset.masks[test_subset.indices],
        transform=None
    )
    return train_dataset, val_dataset, test_dataset
# Modified mask decoder if dataset size > 10k
# Modify the mask decoder for multi-class output for large dataset
class SAM2IrisMaskDecoder(torch.nn.Module):
    def __init__(self, original_decoder, num_classes):
        super().__init__()
        self.original_decoder = original_decoder
        # Replace the output layer with a classification head
        self.classifier = torch.nn.Conv2d(256, num_classes, kernel_size=1)  # 256 is SAM's mask embedding dim

    def forward(self, image_embeddings):
        # Pass dummy prompts (not used, placeholder for SAM2 decoder)
        dummy_prompts = torch.zeros(image_embeddings.shape[0], 1, 256, device=image_embeddings.device)
        # Generate mask embeddings
        masks, _ = self.original_decoder(
            image_embeddings=image_embeddings,
            image_pe=dummy_prompts,  # Placeholder
            sparse_prompt_embeddings=None,
            dense_prompt_embeddings=None,
        )
        # Convert to class logits
        return self.classifier(masks)
    
# Loss functions
def loss_function(pred, target):
    # pred: (B, C, H, W), target: (B, H, W) with class indices
    ce_loss = torch.nn.CrossEntropyLoss()(pred, target)
    dice_loss = 1 - dice_score(pred.softmax(dim=1), target)
    return ce_loss + dice_loss

def dice_score(pred, target, smooth=1e-6):
    # pred: (B, C, H, W), target: (B, H, W)
    pred = pred.softmax(dim=1)
    target_onehot = torch.nn.functional.one_hot(target, num_classes=pred.shape[1]).permute(0,3,1,2)
    intersection = (pred * target_onehot).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target_onehot.sum(dim=(2,3))
    return (2. * intersection + smooth) / (union + smooth)

def IoU(pred, target, smooth=1e-6):
    pred = pred.argmax(dim=1)  # Convert to class indices
    intersection = (pred == target).sum()
    union = (pred != 0).sum() + (target != 0).sum() - intersection
    return (intersection + smooth) / (union + smooth)

# training
batch_size = 8  # Small batch size for limited data
tranform = ToTensorAndNormalize(cfg.img_size)
train_dataset, val_dataset, test_dataset = return_dataset(cfg.image_dirs, cfg.mask_dirs, transform=tranform)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,    # Critical for training
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,   # No need to shuffle validation
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,    # Batch size 1 for per-image evaluation
    shuffle=False
)

# Check a training batch
for images, masks in train_loader:
    # check if a mask is a class-indexed tensor
    print("Training batch shapes:", images.shape, masks.shape)
    break  # Only check first batch

# Check validation (no transforms)
for images, masks in val_loader:
    print("Validation batch (no transforms):", images.shape, masks.shape)
    break
# # Modified mask decoder for training SAM2 when dataset is large
# sam.mask_decoder = SAM2IrisMaskDecoder(sam.mask_decoder, num_classes=cfgs.num_classes)
import torch
### fine tuning on original mask decoder for small dataset
from segment_anything import sam_model_registry

# Load SAM2 and freeze the encoder
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
# Remove the prompt encoder (not needed)
sam.prompt_encoder = None

# Freeze the image encoder (no gradients)
for param in sam.image_encoder.parameters():
    param.requires_grad = False # <--- unfreeze for >10k images
    
# Unfreeze the mask decoder (fine-tune it)
for param in sam.mask_decoder.parameters():
    param.requires_grad = True  # <--- Only change from earlier!

# Loss and optimizer (only affects the decoder)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=1e-5) # small lr for fine tuning

best_val_iou = 0.0
patience = 15  # Stop if no improvement for 15 epochs
epochs_without_improvement = 0

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5, verbose=True
)

for epoch in range(num_epochs=100):
    # Training phase
    sam.train()
    for images, masks in train_loader:
        # Forward pass
        image_embeddings = sam.image_encoder(images)
        outputs = sam.mask_decoder(image_embeddings)
        loss = criterion(outputs, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation phase
    sam.eval()
    val_iou = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            outputs = sam.mask_decoder(sam.image_encoder(images))
            val_iou += IoU(outputs, masks)
    val_iou /= len(val_loader)

    scheduler.step(val_iou)  # Adjust LR based on validation IoU

    # Check validation IoU
    current_val_iou = val_iou
    if current_val_iou > best_val_iou:
        best_val_iou = current_val_iou
        epochs_without_improvement = 0
        # best iou checkpoint
        torch.save(sam.mask_decoder.state_dict(), "best_iris_sam2.pt")
        print(f"Saved best model at epoch {epoch} with IoU {val_iou:.4f}")
    
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Save periodic checkpoint
    if epoch % 5 == 0:
        torch.save({
            "epoch": epoch,
            "model": sam.mask_decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, f"checkpoints/epoch_{epoch}.pt")

# Load the best model
checkpoint = torch.load("checkpoints/epoch_10.pt")
sam.mask_decoder.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optimizer_state"])
# start_epoch = checkpoint["epoch"] + 1  # Resume training from next epoch
best_model = torch.load("best_iris_sam2.pt")
sam.mask_decoder.load_state_dict(best_model["model_state"])
optimizer.load_state_dict(best_model["optimizer_state"])
# Evaluation
predictions = []
sam.eval()
with torch.no_grad():
    for test_image, test_mask in test_loader:
      image_embeddings = sam.image_encoder(test_image)
      logits = sam.mask_decoder(image_embeddings)
      pred_mask = logits.argmax(dim=1)  # (H, W) class indices
      predictions.append(pred_mask)
cv2.imshow(predictions[0].cpu().numpy())
cv2.imshow(test_dataset[0]['mask'].numpy())