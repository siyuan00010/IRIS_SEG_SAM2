# Commented out IPython magic to ensure Python compatibility.
import os
import numpy as np # linear algebra
# %matplotlib inline
import matplotlib.pyplot as plt
import torch
from torchvision import transforms,models,transforms
from torch.utils.data import Dataset, DataLoader,Subset,random_split, ConcatDataset
import torch.optim as optim
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import random
import sklearn
from sklearn.model_selection import KFold

from lovasz_losses import lovasz_hinge
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
# torch.cuda.manual_seed(random_seed)  # For GPU

"""# Config file"""

class arguments:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config(object):
  """Change the args in the "args" variable to change the config"""

  def __init__(self):
    args = arguments(
        # Basic train config
        dataset='combined', #choices ['openEDS','iitd','ubiris','combined']
        batch_size=16,
        setting='local', #choices ['colab','local']
        dataset_root='/Volumes/My Passport/IRIS_ANNOTATION',
        zipped=True,
        random_seed=42,
        train_size=0.8, #Size you want of training for each dataset
        semisupervised=False,
        crossvalidation=True,
        model='unet', #choices ['unet','attention_unet','nested_unet','MTANet']
        lr=1e-4,
        checkpoint_save='checkpoints',
        checkpoint_path='',
        img_size=128,
        output_dir="valid/combined",
        test_output_dir="test/combined",
        verbose=True,
        val_size=0.15,
        train=False

    )
    self.dataset = args.dataset
    self.batch_size = args.batch_size
    self.setting=args.setting
    self.dataset_root=args.dataset_root
    self.zipped=args.zipped
    self.random_seed=args.random_seed
    self.train_size=args.train_size
    self.semisupervised=args.semisupervised
    self.crossvalidation=args.crossvalidation
    self.model=args.model
    self.lr=args.lr
    self.checkpoint_save=args.checkpoint_save
    self.img_size=args.img_size
    self.checkpoint_path=args.checkpoint_path
    self.output_dir=args.output_dir
    self.verbose=args.verbose
    self.val_size=args.val_size
    self.train=args.train
    self.test_output_dir=args.test_output_dir

cfg = Config()

dataset_path = cfg.dataset_root

Images_path_iitd = dataset_path+'/Images/iitd/'
Masks_path_iitd = dataset_path+'/Masks/iitd/'
Images_path_openEDS = dataset_path+'/Images/openEDS/'
Masks_path_openEDS = dataset_path+'/Masks/openEDS/'
Images_path_ubiris= dataset_path+'/Images/ubiris/'
Masks_path_ubiris = dataset_path+'/Masks/ubiris/'

"""If you need to convert npy to masks (openEDS)"""

# # Set up source and destination directories
# source_dir = '/content/drive/MyDrive/CS678-FinalProject/Datasets/Masks/openEDS/labels/'  # Folder containing .npy label files
# output_dir = '/content/drive/MyDrive/CS678-FinalProject/Datasets/Masks/openEDS/Masks/'   # Folder where processed mask images will be saved

# # Ensure the output directory exists
# # Path(output_dir).mkdir(parents=True, exist_ok=True)

# # Loop through all .npy files in the source directory
# for filename in os.listdir(source_dir):
#     if filename.endswith('.npy'):
#         # Construct full file path
#         file_path = os.path.join(source_dir, filename)

#         # Load the mask from the .npy file
#         mask = np.load(file_path)
#         print(mask)
#         # Optionally, process the mask (e.g., convert to uint8 binary mask)
#         binary_mask = (mask ==2).astype(np.uint8) * 255  # Ensure values are 0 or 255

#         # Construct the output image path (changing extension to .png)
#         output_path = os.path.join(output_dir, f"{Path(filename).stem}.png")

#         # Save the mask as a PNG image
#         cv2.imwrite(output_path, binary_mask)

#         print(f"Saved mask image: {output_path}")

"""# Dataloader for images and masks

"""

def custom_collate_fn(batch):
    images = []
    masks = []

    for sample in batch:
        images.append(sample['image'])

        # For unlabeled data, 'mask' will be None, so we append None in that case
        if sample['mask'] is not None:
            masks.append(sample['mask'])
        else:
            masks.append(None)

    # Convert image list into a tensor
    images = torch.stack(images)

    # If masks are provided, convert them to a tensor; otherwise, return None for the masks
    if masks[0] is not None:
        masks = torch.stack(masks)
    else:
        masks = None  # For unlabeled data, masks will be None

    return {'image': images, 'mask': masks}


class ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None, semi_supervised=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.semi_supervised = cfg.semisupervised
        self.image_filenames = sorted(os.listdir(image_dir))  # Ensure sorting

        # Only load mask filenames if mask_dir is provided
        if mask_dir:
            self.mask_filenames = sorted(os.listdir(mask_dir))
        else:
            self.mask_filenames = None

        # Create lists for labeled and unlabeled samples
        self.labeled_indices = [i for i, fname in enumerate(self.image_filenames) if self.has_mask(i)]
        self.unlabeled_indices = [i for i, fname in enumerate(self.image_filenames) if not self.has_mask(i)]
        print(self.labeled_indices)
        print(self.unlabeled_indices)
        # Debugging: Print counts of labeled/unlabeled data
        print(f"Debug Info: {image_dir} - Found {len(self.labeled_indices)} labeled samples and {len(self.unlabeled_indices)} unlabeled samples.")

    def has_mask(self, idx):
        """Check if a mask exists for the given index (i.e., whether the sample is labeled)."""
        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, self.image_filenames[idx])
            return os.path.exists(mask_path)
        return False  # If there's no mask_dir, consider data unlabeled

    def __len__(self):
        """The length will depend on whether we're training with labeled or unlabeled data."""
        if self.semi_supervised:
            # If semi-supervised, return total number of labeled + unlabeled samples
            return len(self.labeled_indices) + len(self.unlabeled_indices)
        else:
            # Otherwise, return only the labeled samples
            return len(self.labeled_indices)

    def __getitem__(self, idx):
        """Return either a labeled or unlabeled sample based on `semi_supervised` flag."""
        if self.semi_supervised:
            # Randomly pick either from labeled or unlabeled
            if idx < len(self.labeled_indices):
                sample_idx = self.labeled_indices[idx]  # Labeled data
                labeled = True
            else:
                sample_idx = self.unlabeled_indices[idx - len(self.labeled_indices)]  # Unlabeled data
                labeled = False
        else:
            # Always pick labeled data
            sample_idx = self.labeled_indices[idx]
            labeled = True

        image_path = os.path.join(self.image_dir, self.image_filenames[sample_idx])
        image = Image.open(image_path).convert('L')  # Assuming grayscale images

        if labeled:
            mask_filename = self.image_filenames[sample_idx]  # Mask matches image filename
            mask_path = os.path.join(self.mask_dir, mask_filename)

            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')  # Convert mask to grayscale ('L')
            else:
                mask = None
        else:
            mask = None  # No mask for unlabeled data

        # Apply transformations if provided
        sample = {'image': image, 'mask': mask, 'labeled': labeled}
        if self.transform:
            sample = self.transform(sample)

        # If the mask is None (for unlabeled data), replace it with a tensor of zeros
        if sample['mask'] is None:
            # Create a tensor of zeros with the same size as the image
            height, width = sample['image'].size[1], sample['image'].size[0]  # height, width
            sample['mask'] = torch.zeros((height, width), dtype=torch.long)  # Unlabeled data
        sample['path']=self.image_filenames[sample_idx]
        return sample


class ToTensorAndNormalize:
    def __init__(self, img_size):
        self.img_size = img_size  # Set the desired image size for cropping
        self.random_crop = transforms.RandomCrop(self.img_size)  # Enable cropping to image size
        self.random_flip = transforms.RandomHorizontalFlip(0.5)

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        # If the image is a PIL object, get the width and height for cropping
        img_width, img_height = image.size
        if(cfg.train==True):
            # Crop the larger side (either height or width)
            if img_width > img_height:
                # Randomly crop the width to match img_height (making it square)
                left = random.randint(0, img_width - img_height)
                top = 0
                right = left + img_height
                bottom = img_height
            else:
                # Randomly crop the height to match img_width (making it square)
                left = 0
                top = random.randint(0, img_height - img_width)
                right = img_width
                bottom = top + img_width

            # Perform the crop
            image = image.crop((left, top, right, bottom))

            # If mask is provided, apply the same crop to the mask
            if mask is not None:
                mask = mask.crop((left, top, right, bottom))

        # Resize the cropped image and mask to target size (128x128 or user-defined)
        image = transforms.Resize((self.img_size,self.img_size))(image)

        image = transforms.ToTensor()(image)  # Convert to tensor
        image = transforms.Normalize(mean=[0.5], std=[0.5])(image)  # Normalize image

        if mask is not None:
            mask = transforms.Resize((self.img_size,self.img_size), interpolation=transforms.InterpolationMode.NEAREST)(mask)
            mask = transforms.ToTensor()(mask)  # Convert mask to tensor

            # Ensure mask is of shape [1, H, W]
            if mask.ndimension() == 2:  # If it's a single channel without the channel dimension
                mask = mask.unsqueeze(0)  # Add the channel dimension to make it [1, H, W]

        # If the mask is None, replace it with a tensor of zeros
        if mask is None:
            # Create a tensor of zeros with the same size as the image
            height, width = image.shape[1], image.shape[2]  # Use the tensor shape (C, H, W)
            mask = torch.zeros((1, height, width), dtype=torch.long)  # Create a mask of zeros

        return {'image': image, 'mask': mask}

"""## Load all datasets, labeled and unlabeled data"""

transform = ToTensorAndNormalize(cfg.img_size)


image_dir_iitd = cfg.dataset_root+"/Images/IITD"
mask_dir_iitd = cfg.dataset_root+"/Masks/IITD"

image_dir_openEDS = cfg.dataset_root+"/Images/openEDS"
mask_dir_openEDS = cfg.dataset_root+"/Masks/openEDS"

image_dir_ubiris = cfg.dataset_root+"/Images/ubiris"
mask_dir_ubiris = cfg.dataset_root+"/Masks/ubiris"



class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class EmptyDataset(torch.utils.data.Dataset):
    """A dataset that returns no data (empty dataset)."""
    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError("Empty dataset, no data available.")



def return_datasets_Concat(image_dirs, mask_dirs):
    labeled_datasets = []
    unlabeled_datasets = []

    # Loop through each pair of image and mask directories
    for img_dir, msk_dir in zip(image_dirs, mask_dirs):
        # Debug print to check the number of files in each directory
        print(f"Images in {img_dir}: {len(os.listdir(img_dir))}")
        print(f"Masks in {msk_dir}: {len(os.listdir(msk_dir))}")

        if len(os.listdir(img_dir)) == 0:
            print(f"Warning: {img_dir} is empty, skipping.")
            continue

        # Labeled dataset (images with corresponding masks)
        dataset = ImageMaskDataset(image_dir=img_dir, mask_dir=msk_dir, transform=transform, semi_supervised=cfg.semisupervised)
        if len(dataset) > 0:
            labeled_datasets.append(dataset)
        else:
            print(f"Warning: No labeled data found in {img_dir}, skipping.")

        # Unlabeled dataset (images only) - no masks provided for these images
        unlabeled_dataset = ImageMaskDataset(image_dir=img_dir, mask_dir=None, transform=transform, semi_supervised=cfg.semisupervised)
        if len(unlabeled_dataset) > 0:
            unlabeled_datasets.append(unlabeled_dataset)
        else:
            print(f"Warning: No unlabeled data found in {img_dir}, proceeding with empty unlabeled dataset.")

    # Check if we have any valid datasets to combine for labeled data
    if len(labeled_datasets) == 0:
        raise ValueError("No labeled datasets found. Check your image and mask directories.")

    # Combine all labeled datasets into one dataset
    combined_labeled_dataset = ConcatDataset(labeled_datasets)

    # Ensure we have data before proceeding to the split
    if len(combined_labeled_dataset) == 0:
        raise ValueError("Combined labeled dataset is empty. Check your dataset.")

    # Load the test filenames from the given text file
    test_filenames = []
    with open('test_set/Final_Test.txt', 'r') as file:
        test_filenames = [line.strip() for line in file.readlines()]

    # Create a list of test samples by ensuring they are in the labeled dataset
    test_samples = []
    print(test_filenames)
    for img_filename in test_filenames:
        img_path = None
        mask_path = None

        # Search for the image file in the image directories
        for img_dir in image_dirs:
            possible_img_path = os.path.join(img_dir, img_filename)
            if os.path.exists(possible_img_path):
                img_path = possible_img_path
                break

        # Search for the corresponding mask file in the mask directories (3 subdirectories)
        if img_path is not None:
            for msk_dir in mask_dirs:
                possible_mask_path = os.path.join(msk_dir, img_filename)
                if os.path.exists(possible_mask_path):
                    mask_path = possible_mask_path
                    break

        # Only add to test_samples if both image and mask exist
        if img_path and mask_path:
            test_sample = {'image': img_path, 'mask': mask_path, 'path': img_filename, 'labeled': True}
            test_samples.append(test_sample)

    # Create DataLoader for the test set (only labeled data)
    test_dataloader = DataLoader(test_samples, batch_size=1, shuffle=False)

    # Step 2: Remove test samples from the labeled dataset to form the train set
    seen_paths = set()  # To keep track of added sample paths
    train_samples = []

    # Add labeled samples to the train set, but skip if the path already exists
    for i in range(len(combined_labeled_dataset)):
        sample = combined_labeled_dataset[i]

        if sample['path'] not in seen_paths and sample['path'] not in test_filenames:
            train_samples.append(sample)
            seen_paths.add(sample['path'])


    # Add unlabeled data to the train set (no duplicate checking needed for unlabeled data)
    for unlabeled_dataset in unlabeled_datasets:
        for i in range(len(unlabeled_dataset)):
            sample = unlabeled_dataset[i]
            if sample['path'] not in seen_paths and sample['path'] not in test_filenames:
                train_samples.append(sample)
                seen_paths.add(sample['path'])

    # Print out the sizes
    print("Train Size Labeled:" + str(len([s for s in train_samples if 'mask' in s])))  # Count labeled samples
    print("Train Size Unlabeled:" + str(len([s for s in train_samples if 'mask' not in s])))  # Count unlabeled samples
    print("Test Size: " + str(len(test_samples)))

    return train_samples, test_dataloader

# Load datasets for IITD, openEDS, and uBIRIS with labeled and unlabeled data
# iitd_train_dataloader, iitd_val_dataloader, iitd_test_dataloader, iitd_train_dataloader_ul = return_datasets(image_dir_iitd, mask_dir_iitd)
# openEDS_train_dataloader, openEDS_val_dataloader, openEDS_test_dataloader, openEDS_train_dataloader_ul = return_datasets(image_dir_openEDS, mask_dir_openEDS)
# ubiris_train_dataloader, ubiris_val_dataloader, ubiris_test_dataloader, ubiris_train_dataloader_ul = return_datasets(image_dir_ubiris, mask_dir_ubiris)

"""# Models

Unet
"""

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNetModel(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

"""Attention Unet"""

import torch.nn as nn
class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        # number of input channels is a number of filters in the previous layer
        # number of output channels is a number of filters in the current layer
        # "same" convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out


class AttentionUNet(nn.Module):

    def __init__(self, img_ch=3, output_ch=1):
        super(AttentionUNet, self).__init__()

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(img_ch, 64)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 256)
        self.Conv4 = ConvBlock(256, 512)
        self.Conv5 = ConvBlock(512, 1024)

        self.Up5 = UpConv(1024, 512)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
        self.UpConv5 = ConvBlock(1024, 512)

        self.Up4 = UpConv(512, 256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.UpConv4 = ConvBlock(512, 256)

        self.Up3 = UpConv(256, 128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv3 = ConvBlock(256, 128)

        self.Up2 = UpConv(128, 64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv2 = ConvBlock(128, 64)

        self.Conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        e : encoder layers
        d : decoder layers
        s : skip-connections from encoder layers to decoder layers
        """
        e1 = self.Conv1(x)

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)

        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)

        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)

        s4 = self.Att5(gate=d5, skip_connection=e4)
        d5 = torch.cat((s4, d5), dim=1) # concatenate attention-weighted skip connection with previous layer output
        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)

        out = self.Conv(d2)

        return out

"""Nested Unet"""

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=1, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

"""## Loss Functions"""

# Unet
def dice_loss(pred, target, smooth=1e-6):
    """
    Dice loss for segmentation tasks.
    """
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

class UNetLoss(nn.Module):
    def __init__(self):
        super(UNetLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss (logits)

    def forward(self, pred, target):
        # Apply BCE Loss
        bce = self.bce_loss(pred, target)

        # Convert logits to probability and apply Dice Loss
        pred_probs = torch.sigmoid(pred)
        dice = dice_loss(pred_probs, target)

        return bce + dice  # Combined BCE and Dice Loss

# Nested Unet
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice

class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss

# Attention Unet
def normalize(x):
    return x / 255.0


def dice_coeff(prediction, target):
    """Calculate dice coefficient from raw prediction."""

    mask = np.zeros_like(prediction)
    mask[prediction >= 0.5] = 1

    inter = np.sum(mask * target)
    union = np.sum(mask) + np.sum(target)
    epsilon = 1e-6
    result = np.mean(2 * inter / (union + epsilon))
    return result
def f1_iou_scores(prediction, target):
    # Flatten the arrays to treat each pixel equally
    prediction=prediction.cpu().numpy()
    target=target.cpu().numpy()
    target_flat = target.flatten()
    prediction_flat = prediction.flatten()

    # Calculate True Positive (TP), False Positive (FP), and False Negative (FN)
    TP = np.sum((target_flat == 1) & (prediction_flat == 1))
    FP = np.sum((target_flat == 0) & (prediction_flat == 1))
    FN = np.sum((target_flat == 1) & (prediction_flat == 0))
    TN = np.sum((target_flat == 0) & (prediction_flat == 0))

    # Calculate Precision and Recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Calculate F1 Score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0

    # Calculate IoU
    intersection = TP
    union = TP + FP + FN
    iou_score = intersection / union if union > 0 else 0

    return f1_score, iou_score

class FocalLoss(nn.modules.loss._WeightedLoss):

    def __init__(self, gamma=0, size_average=None, ignore_index=-100,
                 reduce=None, balance_param=1.0):
        super(FocalLoss, self).__init__(size_average)
        self.gamma = gamma
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, input, target):
        # inputs and targets are assumed to be BatchxClasses
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)

        # compute the negative likelyhood
        logpt = - F.binary_cross_entropy_with_logits(input, target)
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Flatten the input and target tensors to calculate intersection and union

        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice  # Dice Loss is 1 - Dice Coefficient

class SegmentationLoss(nn.Module):
    def __init__(self, seg_weight=1.0, dice_weight=1.0):
        super(SegmentationLoss, self).__init__()
        self.seg_loss = nn.BCEWithLogitsLoss()  # BCEWithLogitsLoss for binary segmentation
        self.dice_loss = DiceLoss()
        self.seg_weight = seg_weight
        self.dice_weight = dice_weight

    def forward(self, seg_pred, seg_target):
        # For BCEWithLogitsLoss, the target must be a float tensor with the same shape as the prediction
        # Calculate BCEWithLogitsLoss
        bce_loss = self.seg_loss(seg_pred, seg_target)

        # Calculate Dice Loss (we assume seg_pred is not passed through sigmoid)
        dice_loss = self.dice_loss(torch.sigmoid(seg_pred), seg_target)

        # Combine both losses
        total_loss = self.seg_weight * bce_loss + self.dice_weight * dice_loss
        return total_loss

"""Save/Load Checkpoints"""

def save_checkpoint(model, optimizer, epoch, fold, checkpoint_dir="checkpoints"):
    """
    Save the model and optimizer state dict to a file.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_fold{fold}_data-{cfg.dataset}_mode-{cfg.semisupervised}_model-{cfg.model}.pth")
    torch.save({
        'epoch': epoch,
        'fold':fold,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path, device='cuda'):
    """
    Load a checkpoint to resume training.

    Args:
        model (torch.nn.Module): The model instance.
        optimizer (torch.optim.Optimizer): The optimizer instance.
        checkpoint_path (str): Path to the checkpoint file.
        device (str): The device to load the model and optimizer to ('cuda' or 'cpu').

    Returns:
        epoch (int): The epoch at which training was last saved.
        fold (int): The fold associated with the checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file does not exist at {checkpoint_path}")
        return None, None

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Restore model and optimizer state dicts
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    fold = checkpoint_path.split('fold')[1].split('_')[0]  # Extract fold number from filename

    print(f"Checkpoint loaded from {checkpoint_path}. Resuming from epoch {epoch}, fold {fold}.")

    return epoch, int(fold)  # Return epoch and fold number

"""Train"""

def train_model(labeled_dataloader, model, optimizer,fold, criterion, epochs,pseudo_label_threshold=0.9):
    checkpoint_filename = f"checkpoint_fold{fold}_data-{cfg.dataset}_mode-{cfg.semisupervised}_model-{cfg.model}.pth"
    checkpoint_full_path = os.path.join(cfg.checkpoint_save, checkpoint_filename)
    model.train()
    epoch_start = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(checkpoint_full_path):
        epoch_start, fold= load_checkpoint(model, optimizer, checkpoint_full_path, device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move model to GPU or CPU

    for epoch in range(epoch_start+1,epochs):
        running_loss = 0.0

        correct = 0
        total = 0

        for i, labeled_sample in enumerate(labeled_dataloader):
            loss_1=0
            loss_2=0
            images = labeled_sample['image'].to(device)
            labels = labeled_sample['mask'].to(device)
            outputs = model(images)

            # Create a mask for labeled samples
            is_labeled = labels.sum(dim=(1, 2, 3)) > 0  # Sum along the spatial dimensions (H, W) and check if it's > 0

            # Now, process labeled samples
            if is_labeled.any():  # Only process if there are labeled samples in the batch

                labeled_labels = labels[is_labeled]
                labeled_outputs = outputs[is_labeled]

                # Apply sigmoid for binary segmentation
                labeled_outputs = torch.sigmoid(labeled_outputs).unsqueeze(1)
                labeled_labels = labeled_labels.unsqueeze(1)
                labeled_labels=labeled_labels.float()

                loss = criterion(labeled_outputs, labeled_labels)
                loss_2 += loss

                predicted = (labeled_outputs > 0.5).float()
                total += labeled_labels.view(-1).numel()
                correct += (predicted == labeled_labels).sum().item()

            # Now, process unlabeled samples (where the label sum is 0)
            if (~is_labeled).any() and cfg.semisupervised==True:  # Only process if there are unlabeled samples in the batch

                unlabeled_outputs = outputs[~is_labeled]

                with torch.no_grad():
                    sigmoid_outputs = torch.sigmoid(unlabeled_outputs)
                    confidence = sigmoid_outputs * (1 - sigmoid_outputs)
                    pseudo_labels = sigmoid_outputs
                    pseudo_labels = pseudo_labels.float()

                loss_unlabeled = criterion(unlabeled_outputs, pseudo_labels)
                weighted_loss = (loss_unlabeled * confidence).mean()
                loss_1 += weighted_loss

            final_loss=loss_1+loss_2
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            running_loss+=final_loss.item()

            # Print status
            if i % 400 == 0:

                if total > 0:

                    print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(labeled_dataloader)}] | Loss: {running_loss / (i+1)}, Accuracy: {correct / total}")
                else:
                    print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(labeled_dataloader)}] | Loss: {running_loss / (i+1)}, Accuracy: Not computed (no data)")

                if(cfg.verbose):
                    non_zero_label_indices = (labels != 0).nonzero(as_tuple=True)[0]

                    outputs = model(images)
                    outputs = torch.sigmoid(outputs)
                    predicted = (outputs > 0.5).float()
                    print(non_zero_label_indices)
                    if(non_zero_label_indices is not None):
                        pred_mask = predicted[non_zero_label_indices[0]].cpu().detach()
                        true= labels[non_zero_label_indices[0]].cpu().detach()
                        # Save the predicted mask
                        out=cfg.output_dir+'_semi-supervised-'+str(cfg.semisupervised)+'_model-'+str(cfg.model)

                        if not os.path.exists(out):
                            os.makedirs(out)
                        save_image(pred_mask, os.path.join(out, f'epoch_{epoch}_iter_{i}_fold_{fold}_pred.png'))
                        save_image(true, os.path.join(out, f'epoch_{epoch}_iter_{i}_fold_{fold}_true.png'))

        if total > 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(labeled_dataloader)}] | Loss: {running_loss / (i+1)}, Accuracy: {correct / total}")
        else:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(labeled_dataloader)}] | Loss: {running_loss / (i+1)}, Accuracy: Not computed (no data)")

        save_checkpoint(model, optimizer, epoch, fold,checkpoint_dir=cfg.checkpoint_save)

"""# Run loop for cross validation"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = cfg.checkpoint_save
transform = ToTensorAndNormalize(cfg.img_size)
def cross_validate_labeled_unlabeled(image_dir, mask_dir, transform, batch_size, num_epochs=10, pseudo_label_threshold=0.9, k_folds=3):

    dataset_train,dataset_test=return_datasets_Concat(image_dir,mask_dir)
    dataset_size = len(dataset_train)

    # K-Fold Cross-Validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(dataset_size))):
        print(f"Training on fold {fold+1}/{k_folds}...")

        # Create subsets for training and validation
        train_subset = Subset(dataset_train, train_idx)
        val_subset = Subset(dataset_train, val_idx)

        # Create data loaders for training and validation sets
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        # Pick model
        if(cfg.model=="unet"):
           model=UNetModel(n_channels=1,n_classes=1)
           criterion=UNetLoss()
        elif(cfg.model=="attention_unet"):
          model = AttentionUNet(img_ch=1)
          criterion = DiceLoss()
        elif(cfg.model=="nested_unet"):
          model = NestedUNet(num_classes=1)
          criterion = BCEDiceLoss()
        model=model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        train_model(train_loader, model, optimizer,fold+1, criterion, num_epochs)
        # Evaluate the model on the validation fold
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for sample in val_loader:
                images = sample['image'].cuda()
                labels = sample['mask'].cuda()
                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.view(-1).numel()
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        fold_accuracies.append(accuracy)
        print(f"Fold {fold+1} accuracy: {accuracy}")
    avg_accuracy = np.mean(fold_accuracies)
    print(f"Average cross-validation accuracy: {avg_accuracy}")
if(cfg.dataset=='openEDS'):
   data_img=image_dir_openEDS
   data_mask=mask_dir_openEDS
elif(cfg.dataset=='iitd'):
   data_img=image_dir_iitd
   data_mask=mask_dir_iitd
elif(cfg.dataset=='ubiris'):
   data_img=image_dir_ubiris
   data_mask=mask_dir_ubiris
elif(cfg.dataset=='combined'):
   data_img=[image_dir_openEDS, image_dir_iitd, image_dir_ubiris]
   data_mask=[mask_dir_openEDS, mask_dir_iitd, mask_dir_ubiris]

cross_validate_labeled_unlabeled(data_img,data_mask,transform,cfg.batch_size,num_epochs=50)

"""# Metrics"""

""" Dice F1 and IoU
"""
def calculate_metrics(prediction, target, smooth=1e-6):
    """
    Calculate Dice Score, IoU, and F1 Score for binary masks using PyTorch.

    Args:
        prediction (torch.Tensor): Predicted binary mask (0 or 1 values).
        target (torch.Tensor): Ground truth binary mask (0 or 1 values).
        smooth (float): Small constant to avoid division by zero.

    Returns:
        dict: Dictionary containing Dice Score, IoU, and F1 Score.
    """
    # Flatten tensors
    prediction_flat = prediction.view(-1)
    target_flat = target.view(-1)
    # Create a binary mask from the prediction using a threshold of 0.5
    mask = (prediction >= 0.5).float()

    # Calculate intersection and union
    inter = torch.sum(mask * target).item()
    union = torch.sum(mask).item() + torch.sum(target).item()

    # Avoid division by zero using epsilon
    epsilon = 1e-6
    dice_score = (2 * inter) / (union + epsilon)
    print('dice score:',dice_score)

    # Calculate intersection and union
    intersection = (prediction * target).sum().float()
    total = prediction.sum().float() + target.sum().float()
    union = total - intersection

    # Dice Score
    dice = (2. * intersection + smooth) / (total + smooth)
    print('dice: ',dice)

    # Calculate TP, FP, FN, TN
    TP = torch.sum((target_flat == 1) & (prediction_flat == 1)).item()
    FP = torch.sum((target_flat == 0) & (prediction_flat == 1)).item()
    FN = torch.sum((target_flat == 1) & (prediction_flat == 0)).item()
    TN = torch.sum((target_flat == 0) & (prediction_flat == 0)).item()

     # Calculate IoU
    intersection = TP
    union = TP + FP + FN
    iou_score = intersection / union if union > 0 else 0
    print('iou: ',iou_score)

    # Calculate Precision and Recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Calculate F1 Score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
    print('f1: ',f1_score)

    return {"Dice Score": dice_score, "IoU": iou_score, "F1 Score": f1_score}

"""# Test

"""

# Run through 3 models and produce test results to file
def test(image_dir, mask_dir, transform, checkpoint1, checkpoint2, checkpoint3):
    dataset_train,dataset_test=return_datasets_Concat(image_dir,mask_dir)
    dataset_size = len(dataset_test)  # Converts to tensor and normalizes to [0, 1] range
    transform = ToTensorAndNormalize(cfg.img_size)
    accuracies=[]
    total=0
    correct=0
    # Ensemble of voting for 3 models
    if(cfg.model=="unet"):
        # model=UNetModel(n_channels=1,n_classes=1)
        model1=UNetModel(n_channels=1,n_classes=1).to(device)
        # model2=UNetModel(n_channels=1,n_classes=1).to(device)
        # model3=UNetModel(n_channels=1,n_classes=1).to(device)
    elif(cfg.model=="attention_unet"):
        model1 = AttentionUNet(img_ch=1).to(device)  # Using GPU
        model2 = AttentionUNet(img_ch=1).to(device)
        model3 = AttentionUNet(img_ch=1).to(device)
    elif(cfg.model=="nested_unet"):
        model1 = NestedUNet(num_classes=1).to(device)  # Using GPU
        model2 = NestedUNet(num_classes=1).to(device)
        model3 = NestedUNet(num_classes=1).to(device)
    # elif(cfg.model=="MTANet"):
    #     model=MTANet(num_classes=1)
    model1.eval()
    # model2.eval()
    # model3.eval()

    print('loading model checkpoints ')
    checkpoint1=torch.load(checkpoint1)
    # checkpoint2=torch.load(checkpoint2)
    # checkpoint3=torch.load(checkpoint3)
    model1.load_state_dict(checkpoint1['model_state_dict'])
    # model2.load_state_dict(checkpoint2['model_state_dict'])
    # model3.load_state_dict(checkpoint3['model_state_dict'])

    with torch.no_grad():
        for i, labeled_sample in enumerate(dataset_test):
            metrics={}
            # Filter out test set only images that have masks
            # Output test results to file
            out=cfg.output_dir+'_semi-supervised-'+str(cfg.semisupervised)+'_model-'+str(cfg.model)
            name=labeled_sample['path']
            print('***************',name,'\n')

            name, _ = os.path.splitext(name)  # Remove existing extension

            image = Image.open(labeled_sample['image']).convert('L')  # Convert to grayscale
            mask = Image.open(labeled_sample['mask']).convert('L')    # Convert to grayscale


            # Create the sample dictionary
            sample = {'image': image, 'mask': mask}

            # Apply the transformation
            transformed_sample = transform(sample)
            print('transformed sample: ',transformed_sample['image'].shape)

            # Apply the transformation

            images = transformed_sample['image'].unsqueeze(0).to(device)

            # print(f'{i}images',images.shape,torch.max(images),torch.min(images),'\n',images)
            labels = transformed_sample['mask'].unsqueeze(0).to(device)
            print('labels: ',labels)
            true= labels[0].cpu().detach()

            outputs1=model1(images)
            outputs1 = torch.sigmoid(outputs1)
            print('outputs1: ',outputs1,'\n')
            predicted1 = (outputs1 > 0.5).float().cpu().detach()
            pred1_metrics=calculate_metrics(predicted1,true)
            pred1_squeeze=predicted1.squeeze()
            predicted1_np=pred1_squeeze.numpy()
            predicted1_np = (predicted1_np * 255).astype(np.uint8)
            img1 = Image.fromarray((predicted1_np.squeeze()))

            outputs2 = torch.sigmoid(model2(images))
            print('outputs2: ',outputs2,'\n')
            predicted2 = (outputs2 > 0.5).float().cpu().detach()
            pred2_metrics=calculate_metrics(predicted2,true)
            pred2_squeeze=predicted2.squeeze()
            predicted2_np=pred2_squeeze.numpy()
            predicted2_np = (predicted2_np * 255).astype(np.uint8)
            img2 = Image.fromarray((predicted2_np.squeeze()))

            outputs3 = torch.sigmoid(model3(images))
            print('outputs3: ',outputs3,'\n')
            predicted3 = (outputs3 > 0.5).float().cpu().detach()
            pred3_metrics=calculate_metrics(predicted3,true)
            pred3_squeeze=predicted3.squeeze()
            predicted3_np=pred3_squeeze.numpy()
            predicted3_np = (predicted3_np * 255).astype(np.uint8)
            img3 = Image.fromarray((predicted3_np.squeeze()))

            outputs = (outputs1 + outputs2 + outputs3)

            print('outputs: ',outputs,'\n')
            predicted = (outputs > 1).float()
            pred_mask = predicted[0].cpu().detach()  # Move to CPU and detach from the graph

            _, predicted = torch.max(outputs, 1)
            total += labels.view(-1).numel()
            correct += (predicted == labels).sum().item()
            accuracy = correct / total
            accuracies.append(accuracy)


            # Calculate Dice, f1, IOU
            print('calculating pred_mask dice f1 and iou')
            pred_metrics = calculate_metrics(pred_mask, true)
            name = labeled_sample['path'] if isinstance(labeled_sample['path'], str) else labeled_sample['path'][0]
            # Output test results to file
            name, _ = os.path.splitext(name)  # Remove existing extension
            if not os.path.exists(out):
                os.makedirs(out)
            img1.save(os.path.join(out, f'{name}_test_pred_model1.png'))
            img2.save(os.path.join(out, f'{name}_test_pred_model2.png'))
            img3.save(os.path.join(out, f'{name}_test_pred_model3.png'))

            pred = Image.fromarray((pred_mask.squeeze().cpu().numpy()*255).astype(np.uint8))
            pred.save(os.path.join(out, f'{name}_test_pred.png'))
            img = Image.fromarray((true.squeeze().cpu().numpy()*255).astype(np.uint8))
            img.save(os.path.join(out, f'{name}_test_true.png'))
            image.save(os.path.join(out, f'{name}_test.png'))
            print('metrics:',pred_metrics)

            acc_out=cfg.output_dir+'_semi-supervised-'+str(cfg.semisupervised)+'_model-'+str(cfg.model)+'_accuracy'
            ave_acc_out=cfg.output_dir+'_semi-supervised-'+str(cfg.semisupervised)+'_model-'+str(cfg.model)+'_ave_accuracy'
            dice_f1_iou_out=cfg.output_dir+'_semi-supervised-'+str(cfg.semisupervised)+'_model-'+str(cfg.model)+'_accuracy'
            metrics={'pred_1:':pred1_metrics,'pred_2:':pred2_metrics,'pred_3:':pred3_metrics,'pred_combined:':pred_metrics}
            log_files(acc_out,accuracy)
            log_files(dice_f1_iou_out,metrics)
    ave_accuracy=np.mean(accuracies)
    log_files(ave_acc_out,ave_accuracy)
    print(f"Average cross-validation accuracy: {ave_accuracy}")

def log_files(out_dir,input):
    if not os.path.exists(os.path.dirname(out_dir)):
        os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    # Write or append to the file
    with open(out_dir, 'a') as log_file:
        log_file.write(f"{input}\n")


# image_dir_iitd = cfg.dataset_root+"/Images/IITD"
# mask_dir_iitd = cfg.dataset_root+"/Masks/IITD"

# image_dir_openEDS = cfg.dataset_root+"/Images/openEDS"
# mask_dir_openEDS = cfg.dataset_root+"/Masks/openEDS"

# image_dir_ubiris = cfg.dataset_root+"/Images/ubiris"
# mask_dir_ubiris = cfg.dataset_root+"/Masks/ubiris"

# image_dir_combined = [image_dir_openEDS, image_dir_iitd, image_dir_ubiris]
# mask_dir_combined = [mask_dir_openEDS, mask_dir_iitd, mask_dir_ubiris]

# if(cfg.dataset=='openEDS'):
#     data_img=image_dir_openEDS
#     data_mask=mask_dir_openEDS
# elif(cfg.dataset=='iitd'):
#     data_img=image_dir_iitd
#     data_mask=mask_dir_iitd
# elif(cfg.dataset=='ubiris'):
#     data_img=image_dir_ubiris
#     data_mask=mask_dir_ubiris
# elif(cfg.dataset=='combined'):
#     data_img=image_dir_combined
#     data_mask=mask_dir_combined

data_mask = ''
data_img = cfg.dataset_root + '/images'
transform=ToTensorAndNormalize(cfg.img_size)

checkpoint1="/Users/mollieyin/VSCode_workspace/model1.pth"
# checkpoint2="../checkpoints/attention-unet-supervised/checkpoint_fold2_data-combined_mode-False_model-attention_unet.pth"
# checkpoint3="../checkpoints/attention-unet-supervised/checkpoint_fold3_data-combined_mode-False_model-attention_unet.pth"
print('Testing**********')
# test(data_img,data_mask,transform,checkpoint1,checkpoint2,checkpoint3)
test(data_img,data_mask,transform,checkpoint1)

"""## Average Test Scores"""

import json

# Function to calculate average Dice Score, IoU, and F1 Score for each set of predictions
def calculate_prediction_averages(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()  # Read the file line by line

    # Initialize accumulators for each prediction type
    total_dice_pred_1 = 0.0
    total_iou_pred_1 = 0.0
    total_f1_pred_1 = 0.0
    count_pred_1 = 0

    # total_dice_pred_2 = 0.0
    # total_iou_pred_2 = 0.0
    # total_f1_pred_2 = 0.0
    # count_pred_2 = 0

    # total_dice_pred_3 = 0.0
    # total_iou_pred_3 = 0.0
    # total_f1_pred_3 = 0.0
    # count_pred_3 = 0

    # total_dice_pred_combined = 0.0
    # total_iou_pred_combined = 0.0
    # total_f1_pred_combined = 0.0
    # count_pred_combined = 0

    # Iterate through the lines in the file
    for i in range(0, len(lines), 2):  # We are processing pairs (value + dictionary)
        # Read the first line (value)
        value = float(lines[i].strip())

        # Read the second line (metrics dictionary)
        metrics_str = lines[i + 1].strip()
        metrics = eval(metrics_str)  # Convert string representation of dictionary into a dict

        # Extracting the metrics from predictions (pred_1, pred_2, pred_3, pred_combined)
        for pred_key, pred_metrics in metrics.items():
            if pred_key == 'pred_1':
                total_dice_pred_1 += pred_metrics['Dice Score']
                total_iou_pred_1 += pred_metrics['IoU']
                total_f1_pred_1 += pred_metrics['F1 Score']
                count_pred_1 += 1
            elif pred_key == 'pred_2':
                total_dice_pred_2 += pred_metrics['Dice Score']
                total_iou_pred_2 += pred_metrics['IoU']
                total_f1_pred_2 += pred_metrics['F1 Score']
                count_pred_2 += 1
            elif pred_key == 'pred_3':
                total_dice_pred_3 += pred_metrics['Dice Score']
                total_iou_pred_3 += pred_metrics['IoU']
                total_f1_pred_3 += pred_metrics['F1 Score']
                count_pred_3 += 1
            elif pred_key == 'pred_combined':
                total_dice_pred_combined += pred_metrics['Dice Score']
                total_iou_pred_combined += pred_metrics['IoU']
                total_f1_pred_combined += pred_metrics['F1 Score']
                count_pred_combined += 1

    # Calculate the averages for each prediction type
    avg_dice_pred_1 = total_dice_pred_1 / count_pred_1 if count_pred_1 > 0 else 0
    avg_iou_pred_1 = total_iou_pred_1 / count_pred_1 if count_pred_1 > 0 else 0
    avg_f1_pred_1 = total_f1_pred_1 / count_pred_1 if count_pred_1 > 0 else 0

    # avg_dice_pred_2 = total_dice_pred_2 / count_pred_2 if count_pred_2 > 0 else 0
    # avg_iou_pred_2 = total_iou_pred_2 / count_pred_2 if count_pred_2 > 0 else 0
    # avg_f1_pred_2 = total_f1_pred_2 / count_pred_2 if count_pred_2 > 0 else 0

    # avg_dice_pred_3 = total_dice_pred_3 / count_pred_3 if count_pred_3 > 0 else 0
    # avg_iou_pred_3 = total_iou_pred_3 / count_pred_3 if count_pred_3 > 0 else 0
    # avg_f1_pred_3 = total_f1_pred_3 / count_pred_3 if count_pred_3 > 0 else 0

    # avg_dice_pred_combined = total_dice_pred_combined / count_pred_combined if count_pred_combined > 0 else 0
    # avg_iou_pred_combined = total_iou_pred_combined / count_pred_combined if count_pred_combined > 0 else 0
    # avg_f1_pred_combined = total_f1_pred_combined / count_pred_combined if count_pred_combined > 0 else 0

    # Print the averages for each prediction type
    print(f"Average for pred_1:")
    print(f"  Dice: {avg_dice_pred_1:.4f}, IoU: {avg_iou_pred_1:.4f}, F1: {avg_f1_pred_1:.4f}")

    # print(f"\nAverage for pred_2:")
    # print(f"  Dice: {avg_dice_pred_2:.4f}, IoU: {avg_iou_pred_2:.4f}, F1: {avg_f1_pred_2:.4f}")

    # print(f"\nAverage for pred_3:")
    # print(f"  Dice: {avg_dice_pred_3:.4f}, IoU: {avg_iou_pred_3:.4f}, F1: {avg_f1_pred_3:.4f}")

    # print(f"\nAverage for pred_combined:")
    # print(f"  Dice: {avg_dice_pred_combined:.4f}, IoU: {avg_iou_pred_combined:.4f}, F1: {avg_f1_pred_combined:.4f}")

# Example usage
file_path = 'test/combined_supervised-model-nested_unet_accuracy.txt'  # Path to your file
calculate_prediction_averages(file_path)