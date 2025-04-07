import os
import torch
import cv2
import numpy as np
import json
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn

class arguments:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Config(object):
  """Change the args in the "args" variable to change the config"""

  def __init__(self):
    args = arguments(
        # root path on linux
        # root_path = '/run/media/wvubiometrics/My Passport/IRIS_ANNOTATION',
        root_path = 'C:/Users/siyuan/Documents/IRIS_DATA',
        # json file path
        json_file_path = 'iris.json',
        images_dir = 'images',
        masks_dir = 'annotations',
        # number of classes
        num_classes = 17,
        img_size = 256,
        # device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        # model
        model='SAM2',
        # checkpoint_dir='/home/wvubiometrics/Documents/Molly/IRIS_SEG_SAM2/checkpoints',
        checkpoint_dir='C:/path/to/new/virtual/environment/IRIS_SEG_SAM2',
        # output_dir="/home/wvubiometrics/Documents/Molly/IRIS_SEG_SAM2/SAM2_output",
        # test_output_dir="/home/wvubiometrics/Documents/Molly/IRIS_SEG_SAM2/SAM2_test_pred",
        output_dir="C:/path/to/new/virtual/environment/IRIS_SEG_SAM2/output",
        test_output_dir="C:/path/to/new/virtual/environment/IRIS_SEG_SAM2/SAM2_new_test",
        test_size=0.15,
        val_size=0.15,
        print_images=True,
        train=True
    )
    self.model = args.model
    self.device = args.device
    self.checkpoint_dir = args.checkpoint_dir
    self.output_dir = args.output_dir
    self.test_output_dir = args.test_output_dir
    self.train = args.train
    self.print_images=args.print_images
    self.num_classes = args.num_classes
    self.img_size = args.img_size
    self.test_size = args.test_size
    self.val_size = args.val_size
    self.images_dir = args.images_dir
    self.masks_dir = args.masks_dir
    self.json_file_path = args.json_file_path
    self.root_path = args.root_path
cfg = Config()

class CreateDataset(Dataset):

    """
    create a dataset from list of images and masks
    Return a dict of image, mask, and file_name
    """

    def __init__(self,image_list,mask_list,file_names,transform):
        self.image_list = image_list
        self.mask_list = mask_list
        self.file_names = file_names
        self.transform = transform
    def __len__(self):
        return len(self.mask_list)
    def __getitem__(self, index):
        image = self.image_list[index]
        mask = self.mask_list[index]
        file_name = self.file_names[index]
        if image is None:
            raise ValueError("Error loading image. Check the file path.")
        if mask is None:
            raise ValueError("Error loading mask. Check the file path.")
        
        if self.transform:
            image,mask = self.transform((image, mask))  # Pass as tuple
            sample = {'image': image, 'mask': mask, 'file_name': file_name}
        else:
            sample = {'image': image, 'mask': mask, 'file_name': file_name}
        return sample
    
 
class ReturnDataset(Dataset):
    '''
    convert subset to dataset
    '''   
    def __init__(self, subset):
        self.subset = subset
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        sample = self.subset[idx]
        return {'image': sample['image'], 'mask': sample['mask'], 'file_name': sample['file_name']}

# transform
class ToTensorAndNormalize:
    '''
    Tranform images and masks
    Return transformed image tensor and mask tensor
    '''
    def __init__(self, img_size):
        self.img_size = img_size  # Set the desired image size
        self.random_flip = transforms.RandomHorizontalFlip(0.5)

    def __call__(self,sample):

        image, mask = sample
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask,(self.img_size, self.img_size) )
        ### histogram equalization
        # Apply Gaussian Blur to reduce noise while preserving edges
        blurred_image = cv2.GaussianBlur(image, (1, 1), 0)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced_image = clahe.apply(blurred_image)

        image = np.expand_dims(enhanced_image,axis=0) # color dim expected 3 channels in image encoder
        image = np.repeat(image,3,axis=0)

        # convert to tensor
        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)
        # print(image_tensor.shape,image_tensor.shape)
        return image_tensor, mask_tensor
    
### split the full dataset     
def split_subset(dataset):
    num_data = len(dataset)
    print('dataset len:',num_data)
    # calculate the train, validation and test size
    val_data_size = np.ceil(cfg.val_size*num_data).astype(int)
    test_data_size = np.ceil(cfg.test_size*num_data).astype(int)
    train_data_size = num_data - val_data_size - test_data_size
    # random split the dataset 
    train_subset, val_subset, test_subset = random_split(
        dataset, [train_data_size,val_data_size,test_data_size],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    )
    return train_subset, val_subset, test_subset
# Create the datasets
def return_dataset(image_dir, mask_dir,batch_size):

    # Get sorted file paths and names
    img_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))
    
    img_paths = [os.path.join(image_dir, f).replace('\\','/') for f in img_files]
    mask_paths = [os.path.join(mask_dir, f).replace('\\','/') for f in mask_files]

    # img_paths = img_paths[:40]
    # mask_paths = mask_paths[:40]

    # Validate images and keep track of valid files
    valid_img_files = []
    valid_img_paths = []
    for img_file, img_path in zip(img_files, img_paths):
        try:
            image = Image.open(img_path)
            image.verify()
            valid_img_files.append(img_file)
            valid_img_paths.append(img_path)
        except (IOError, SyntaxError):
            print("Invalid image, skipping:", img_path)

      # Validate masks and keep track of valid files
    valid_mask_files = []
    valid_mask_paths = []
    for mask_file, mask_path in zip(mask_files, mask_paths):
        try:
            msk = Image.open(mask_path)
            msk.verify()
            valid_mask_files.append(mask_file)
            valid_mask_paths.append(mask_path)
        except (IOError, SyntaxError):
            print("Invalid mask, skipping:", mask_path)

    # Read only valid images and masks
    images = [cv2.imread(path, 0) for path in valid_img_paths]  # grayscale
    masks = [cv2.imread(path,1) for path in valid_mask_paths]
    masks = [cv2.cvtColor(mask,cv2.COLOR_BGR2RGB) for mask in masks]
    # Get base filenames without extension for saving predictions later
    file_names = [os.path.splitext(f)[0] for f in valid_img_files]

    if cfg.train:
        transform = ToTensorAndNormalize(cfg.img_size)
        dataset = CreateDataset(images, masks, file_names, transform=transform)
    else:
        transform = ToTensorAndNormalize(cfg.img_size)
        dataset = CreateDataset(images, masks, file_names, transform=transform)

    # Load iris subset
    train_dataset, val_dataset, test_dataset = split_subset(dataset)

    train_sample = ReturnDataset(train_dataset)
    train_loader = DataLoader(
        train_sample,
        batch_size=batch_size,
        shuffle=True,    # Critical for training
        num_workers=1,  # 2 subprocesses
        pin_memory=True
    )
    val_sample = ReturnDataset(val_dataset)
    val_loader = DataLoader(
        val_sample,
        batch_size=batch_size,
        shuffle=False,   # No need to shuffle validation
        num_workers=1
    )
    test_sample = ReturnDataset(test_dataset)
    test_loader = DataLoader(
        test_sample,
        batch_size=1,    # Batch size 1 for per-image evaluation
        shuffle=False
    )
    return train_loader, val_loader, test_loader

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    """
    Convert torch Tensor to numpy image.
    Accepts: 3D(H,W,C), or 2D(H,W)
    Returns: numpy image (H,W,C) or (H,W)
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    if tensor.dim() not in [2, 3]:
        raise ValueError(f"Unsupported tensor dimension: {tensor.dim()} (expected 2 or 3)")

    tensor = tensor.detach().cpu().float()
    tensor = torch.clamp(tensor, min_max[0], min_max[1])  # Clamp values
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # Normalize to [0,1]
    tensor = (tensor * 255).round().to(torch.uint8)  # Scale to [0,255]

    img_np = tensor.numpy()
    
    if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:  # CHW -> HWC
        img_np = np.transpose(img_np, (1, 2, 0))

    img = np.clip(img_np, 0, 255).astype(np.uint8)
    # print('in tensor to img \n',img)

    return img.astype(out_type)

def save_img(img, img_path,upscale=4):
    """
    Save a numpy image to disk. Automatically handles RGB to BGR conversion for OpenCV.

    Args:
        img (np.ndarray): Image in HWC (RGB) or HW format.
        img_path (str): Path to save image.
    """
    print(f"Saving image to {img_path}, shape: {img.shape}, dtype: {img.dtype}")
    
    if img.ndim == 3:
        if img.shape[2] == 3:
            rgb_image = img.astype(np.uint8)  # ensure dtype is uint8
            image = Image.fromarray(rgb_image)
            if upscale > 1:
                new_size = (image.width * upscale, image.height * upscale)
                image = image.resize(new_size, Image.NEAREST) 
            image.save(img_path)
        elif img.shape[2] not in [1, 3, 4]:
            raise ValueError(f"Unsupported number of channels: {img.shape[2]}")
    elif img.ndim != 2:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    # cv2.imwrite(img_path, img)

def load_json(filename):
    """Reads a JSON file and returns the data as a dictionary."""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)  # Load JSON data into a dictionary
        return data  # Return the dictionary with keys and values
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file '{filename}' is not a valid JSON file.")
        return None

# def get_mask_mappings():

#     json_data = load_json(cfg.root_path + '/'+cfg.json_file_path)
#     keys = []
#     values = {}
#     mappings = {}
#     for key, value in json_data.items():
#         keys.append(key)
#         values.update(value)
#     for key, value in values.items():
#         keys.append(key)
#         mappings.setdefault(value['name'],(value['id'],value['color']))
#     # print(id_color_dict.values())
#     cfg.num_classes = len(mappings)
#     return mappings
def get_mask_mappings():
    # Load the JSON data
    json_data = load_json(cfg.root_path + '/' + cfg.json_file_path)
    
    mappings = {}
    # Iterate through the 'labels' part of the JSON data
    for key, value in json_data['labels'].items():
        mappings[value['name']] = (value['id'], value['color'])  # Store the id and color for each label
    
    # Update the number of classes
    cfg.num_classes = len(mappings)
    
    return mappings

# def label_to_rgb(pred_mask,mapping):
#     """
#     Convert a tensor of class indices to RGB colors based on a predefined color map
#     Args:
#         label_tensor: (H, W) tensor of class indices (int)
#     Returns:
#         rgb_tensor: (H, W,3) tensor of RGB values (0-255)
#     """
#     # Initialize RGB
#     h, w = pred_mask.shape
#     rgb = np.zeros((h, w, 3), dtype=np.uint8)
#     pred_mask = pred_mask.cpu().numpy()
#     unique_labels = np.unique(pred_mask)
#     print(f"Unique labels in sample: {unique_labels}")

#     # Map each class to its color
#     for class_idx, color in mapping.values():
#         mask = (pred_mask == class_idx)
#         rgb[mask] = color 
     
#     return torch.from_numpy(rgb).to(torch.int8)
def label_to_rgb(pred_mask, mapping):
    """
    Convert a tensor of class indices to RGB colors based on a predefined color map.
    Args:
        pred_mask: (H, W) tensor of class indices (int)
        mapping: a dict with class ids and their corresponding RGB colors
    Returns:
        rgb_tensor: (H, W, 3) tensor of RGB values (0-255)
    """
    # Initialize RGB image with zero values (for background or NotYetLabeled)
    h, w = pred_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    pred_mask = pred_mask.cpu().numpy()  # Convert to numpy for processing
    
    # Check for unique classes in pred_mask
    unique_labels = np.unique(pred_mask)
    print(f"Unique labels in sample: {unique_labels}")  # Debugging: display the unique labels
    
    # Assign color for each class based on mapping
    for class_idx, color in mapping.values():
        mask = (pred_mask == class_idx)
        rgb[mask] = color  # Assign corresponding color to matching mask indices

    # Handle `NotYetLabeled` pixels (class 0, treated as background) separately
    if 0 in mapping:  # If background/NotYetLabeled is in the mapping
        rgb[pred_mask == 0] = mapping[0][1]  # Get the color for class 0 (background or NotYetLabeled)
    else:
        # If `NotYetLabeled` (class 0) is not in the mapping, manually assign background color
        rgb[pred_mask == 0] = [0, 0, 0]  # Background: black

    return torch.from_numpy(rgb).to(torch.int8)  # Convert the RGB image back to tensor
# def replace_with_labels(masks, mappings):
#     '''
#     Args: masks: (B,H,W,C)
#           mappings: a dict of mask labels and colors
            
#     Return: np array of masks in labels ((B,H,W))
#     '''
#     masks = masks.numpy()  ## convert it to numpy array
#     B,H,W,_=masks.shape
#     segment_masks = np.zeros((B,H,W),dtype=np.int64)
#     for id,color in mappings.values():
#         for b in range(B):
#             coord = np.all(masks[b]==np.array(color),axis=-1)
#             segment_masks[b][coord]=id
#     return segment_masks
def replace_with_labels(masks, mappings):
    '''
    Args: 
        masks: Tensor of shape (B, H, W, C), representing one-hot or RGB masks
        mappings: A dictionary mapping class names to (id, color) pairs
        
    Returns:
        A tensor of shape (B, H, W) with label-based mask values
    '''
    B, H, W, C = masks.shape  # Get batch size, height, width, and channels
    
    # Initialize an empty tensor for the segment masks (label-based masks)
    segment_masks = torch.zeros((B, H, W), dtype=torch.int64, device=masks.device)
    
    # Convert mappings to a tensor for faster color comparison
    color_dict = torch.tensor([color for _, color in mappings.values()], device=masks.device)
    
    for idx, color in enumerate(color_dict):
        # Create a mask where the color matches
        mask = torch.all(masks == color, dim=-1)  # (B, H, W), True where the color matches
        
        # Assign the corresponding label id to the segment_masks
        segment_masks[mask] = idx
    
    return segment_masks
def generate_prompts_from_masks(segmented_mask,total_points=50):
    """Generate random label-point from masks.
        Args:
            masks: np array (B,H,W) labeled masks.
            total_points: Number of points to sample.
        Returns: 
            points:(B,N,2) normalized coordinates (N is number of points)
            labels: (B,N) point labels (0 unlabeled, 1 eye, 2....etc)
    """

    B,H,W = segmented_mask.shape
    device = segmented_mask.device
    points1=[]
    labels1=[]
    for b in range(B):
        batch_points=[] # tensor
        batch_labels=[] # tensor
        unique_labels = torch.unique(segmented_mask[b].cpu()) # the number of unique ids
        # print('in batch ',b,'labels number ',unique_labels)
        unique_labels = unique_labels[unique_labels != 0] # exclude unlabeled
        labels = unique_labels.cpu().numpy()
        # Vectorized extraction of coordinates for all labels at once
        coords = torch.nonzero(segmented_mask[b] != 0, as_tuple=False)  # Get all non-background coordinates
        coords = coords.cpu().numpy()
        if coords.shape[0] > 0:
            # Iterate through unique labels to assign proper labels to points
            for label in labels:
                # Mask for the current label
                label_mask = (segmented_mask[b] == label)
                # Filter out the coordinates for this label
                label_coords = coords[(label_mask[coords[:, 0], coords[:, 1]]), :]
                if label_coords.size == 0:  # Skip if no points match this label
                    continue
                # label_coords_normalized = label_coords.float() / torch.tensor([H, W], device=device)
                label_coords_normalized = label_coords.astype(np.float32) / np.array([H, W], dtype=np.float32)
                # Store coordinates and corresponding label
                batch_points.append(torch.from_numpy(label_coords_normalized))
                batch_labels.append(torch.full((label_coords.shape[0],), label, device=device, dtype=torch.long))

        if batch_points:
            
            # Concatenate points/labels for this batch
            all_coords =(torch.cat(batch_points, dim=0))
            all_labels=(torch.cat(batch_labels, dim=0))                 
                
        else:
            # If no points found, pad with zeros (or any placeholder)
            all_coords=(torch.zeros((0, 2), device=device))
            all_labels=(torch.full((0,),0, device=device, dtype=torch.long))

       # If fewer points than required, pad with zeros and label -1
        num_points = all_coords.shape[0]
        if num_points >= total_points:
            indices = torch.randperm(num_points)[:total_points]
            points = all_coords[indices]
            labels = all_labels[indices]
        else:
            print('not find required number of points')
            points = torch.cat([all_coords, torch.zeros((total_points - num_points, 2), device=device)], dim=0)
            labels = torch.cat([all_labels, torch.full((total_points - num_points,), -1, device=device, dtype=torch.long)], dim=0)
        
    labels1.append(labels)
    # scale them back to the image size (H, W)
    points_scaled = points * torch.tensor([segmented_mask.shape[1], segmented_mask.shape[2]], dtype=torch.float)
    points1.append(points_scaled.to(torch.int32))
    return torch.stack(points1),torch.stack(labels1)


# Loss functions
def loss_function(pred, target):
    # pred: (B, C, H, W), target: (B, H, W) with class indices
    target = torch.from_numpy(target)
    ce_loss = torch.nn.CrossEntropyLoss()(pred, target.long())
    dice_loss = 1 - dice_score(pred.softmax(dim=1), target)
    return ce_loss + dice_loss


class DiceLoss(nn.Module):
    def __init__(self, num_classes=17, smooth=1e-6,class_weights=None):
        super().__init__()
        self.smooth = smooth  # Prevents division by zero
        self.num_classes = num_classes
        self.smooth = smooth  # Prevents division by zero
        self.num_classes = num_classes

        self.class_weight = class_weights if class_weights else torch.ones(num_classes)
        self.alpha = 0.5
        self.l1_loss = nn.L1Loss()
    def forward(self, logits, targets):
        """
        Args:
            logits: Model output (B, C, H, W) (raw scores, before softmax)
            targets: Ground truth (B, H, W) with class indices (0-16)
        Returns:
            dice_loss: Scalar
        """
        targets = targets.long()  # Cast targets to an integer type

        # Convert targets to one-hot encoding (B, C, H, W)
        targets_onehot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()

        # Apply softmax to logits to get probabilities
        probs = F.softmax(logits, dim=1)

        dice_loss = 0.0
        l1_loss=0.0
        # Calculate Dice loss for each class
        for class_idx in range(self.num_classes):
            # print("class "+str(class_idx)+": "+str(targets_onehot[:, class_idx].sum()/(2562563)100)+"%")
            # Intersection and Union for class_idx
            intersection = (probs[:, class_idx]*targets_onehot[:, class_idx]).sum()
            union = probs[:, class_idx].sum() + targets_onehot[:, class_idx].sum()
            l1_loss += self.class_weight[class_idx] * self.l1_loss(logits[:, class_idx], targets_onehot[:, class_idx])
            # Compute Dice loss for the class, using class weights
            dice_loss += self.class_weight[class_idx] * (1.0 - (2.0 * intersection) / (union + self.smooth))
            combined_loss = self.alpha * (dice_loss / self.num_classes) + (1 - self.alpha) * l1_loss
        # Return the average Dice loss over all classes
        return combined_loss

def dice_score(pred_class, target,num_classes, smooth=1e-6):
    """
    Args: 
        pred: B,N,H,W, N: default number of binary masks
        target: B, H, W in class indices
    Returns:
        mean_dice: Scalar mean Dice score across all classes
        dice_per_class: Tensor of shape [B, num_classes] with per-class Dice
    """
    # One-hot encode predictions and targets: [B, C, H, W]
    pred_onehot = F.one_hot(pred_class, num_classes).permute(0, 3, 1, 2).float()
    target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    # Calculate intersection and union
    intersection = (pred_onehot * target_onehot).sum(dim=(2, 3))
    union = pred_onehot.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))

    dice = (2. * intersection + smooth) / (union + smooth)  # [B, C]
    mean_dice = dice.mean()  # Mean over batch and class

    return mean_dice, dice  # [scalar], [B, C]

def IoU(pred_class, target, num_classes, smooth=1e-6):
    # print(torch.unique(pred_class))  # See what values it has
    pred_onehot = F.one_hot(pred_class, num_classes).permute(0, 3, 1, 2).float()
    target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    
    intersection = (pred_onehot * target_onehot).sum(dim=(2, 3))
    union = pred_onehot.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3)) - intersection

    iou = intersection / (union + 1e-6)
    mean_iou = iou.mean(dim=1)
    return mean_iou, iou


