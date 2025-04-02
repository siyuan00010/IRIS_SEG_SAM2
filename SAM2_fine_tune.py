import os
import torch
import cv2
import numpy as np
import json
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torch.nn.functional as F

print(torch.cuda.get_device_name(0))

class arguments:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Config(object):
  """Change the args in the "args" variable to change the config"""

  def __init__(self):
    args = arguments(
        # root path on linux
        root_path = '/run/media/wvubiometrics/My Passport/IRIS_ANNOTATION',
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
        checkpoint_dir='/home/wvubiometrics/Documents/Molly/IRIS_SEG_SAM2/checkpoints',
        output_dir="/home/wvubiometrics/Documents/Molly/IRIS_SEG_SAM2/SAM2_output",
        test_output_dir="/home/wvubiometrics/Documents/Molly/IRIS_SEG_SAM2/SAM2test_pred",
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
    
    img_paths = [os.path.join(image_dir, f) for f in img_files]
    mask_paths = [os.path.join(mask_dir, f) for f in mask_files]

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
    masks = [cv2.imread(path, 1) for path in valid_mask_paths]
    
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
    val_sample = ReturnDataset(val_dataset)
    test_sample = ReturnDataset(test_dataset)

    train_loader = DataLoader(
        train_sample,
        batch_size=batch_size,
        shuffle=True,    # Critical for training
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_sample,
        batch_size=batch_size,
        shuffle=False,   # No need to shuffle validation
        num_workers=2
    )

    test_loader = DataLoader(
        test_sample,
        batch_size=1,    # Batch size 1 for per-image evaluation
        shuffle=False
    )
    # print('a test loader:', test_loader.dataset[0])

    return train_loader, val_loader, test_loader

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

def get_mask_mappings():

    json_data = load_json(cfg.root_path + '/'+cfg.json_file_path)
    keys = []
    values = {}
    mappings = {}
    for key, value in json_data.items():
        keys.append(key)
        values.update(value)
    for key, value in values.items():
        keys.append(key)
        mappings.setdefault(value['name'],(value['id'],value['color']))
    # print(id_color_dict.values())
    cfg.num_classes = len(mappings)
    return mappings

def replace_with_labels(masks, mappings):
    '''
    Args: masks: (B,C,H,W)
          mappings: a dict of mask labels and colors
            
    Return: arrays with labels ((B,H,W))
    '''
    masks = masks.numpy()  ## convert it to numpy array
    B,C,H,W=masks.shape
    i=0
    segment_masks = np.zeros((B,H,W),dtype=np.uint8)
    for id,color in mappings.values():
        for b in range(B):
            coord = np.all(masks[b].transpose(1,2,0)==color,axis=-1)
            segment_masks[b][coord]=id
    return torch.from_numpy(segment_masks)

def generate_prompts_from_masks(segmented_mask,total_points=10):
    """Generate random label-point from masks.
        Args:
            masks: (B,H,W) labeled masks.
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
        batch_points=[]
        batch_labels=[]
        unique_labels = torch.unique(segmented_mask[b]) # the number of unique ids
        unique_labels = unique_labels[unique_labels != 0] # exclude unlabeled

        # Vectorized extraction of coordinates for all labels at once
        coords = torch.nonzero(segmented_mask[b] != 0, as_tuple=False)  # Get all non-background coordinates
        if coords.shape[0] > 0:
            # Iterate through unique labels to assign proper labels to points
            for label in unique_labels:
                # Mask for the current label
                label_mask = (segmented_mask[b] == label)
                # Filter out the coordinates for this label
                label_coords = coords[(label_mask[coords[:, 0], coords[:, 1]]), :]
                label_coords_normalized = label_coords.float() / torch.tensor([H, W], device=device)
                # Store coordinates and corresponding label
                batch_points.append(label_coords_normalized)
                batch_labels.append(torch.full((label_coords.shape[0],), label, device=device, dtype=torch.long))

        if batch_points:
            
            # Concatenate points/labels for this batch
            all_coords =(torch.cat(batch_points, dim=0))
            all_labels=(torch.cat(batch_labels, dim=0))                 
                
        else:
            # If no points found, pad with zeros (or any placeholder)
            all_coords=(torch.zeros((0, 2), device=device))
            all_labels=(torch.full((0,), device=device, dtype=torch.long))

        # If fewer points than required, pad with zeros and label -1
        num_points = all_coords.shape[0]
        if num_points >= total_points:
            indices = torch.randperm(num_points)[:total_points]
            points = all_coords[indices]
            labels = all_labels[indices]
        else:
            points = torch.cat([all_coords, torch.zeros((total_points - num_points, 2), device=device)], dim=0)
            labels = torch.cat([all_labels, torch.full((total_points - num_points,), -1, device=device, dtype=torch.long)], dim=0)
        
        labels1.append(labels)
        # scale them back to the image size (H, W)
        points_scaled = points * torch.tensor([segmented_mask.shape[1], segmented_mask.shape[2]], dtype=torch.float)
        points1.append(points_scaled)

    return torch.stack(points1),torch.stack(labels1)

# Loss functions
def loss_function(pred, target):
    # pred: (B, C, H, W), target: (B, H, W) with class indices
    ce_loss = torch.nn.CrossEntropyLoss()(pred, target)
    dice_loss = 1 - dice_score(pred.softmax(dim=1), target)
    return ce_loss + dice_loss

def dice_score(pred, target, smooth=1e-6):
    # Ensure target is of the correct dtype and shape for one-hot encoding
    if target.dim() == 4:  # target shape is (1, C, H, W)
        target = target.squeeze(0)  # Now target is of shape (C, H, W)
    num_classes = pred.shape[0]

    pred = pred.softmax(dim=0)  # Apply softmax to get probabilities per class (C, H, W)

    # Ensure that target values are in the valid range [0, num_classes-1]
    target = torch.clamp(target, min=0, max=num_classes - 1)
    target=target.long()
    # One-hot encode the target (target shape: (B, H, W) => (B, H, W, num_classes))
    target_onehot = torch.nn.functional.one_hot(target, num_classes=num_classes)  # Shape: (B, H, W, num_classes)

    # Remove the last dimension (from (B, H, W, num_classes) to (B, num_classes, H, W))
    target_onehot = target_onehot.permute(0, 3, 1, 2)  # Now it becomes (B, C, H, W)

    # Initialize Dice scores for each class
    dice_scores = np.zeros(num_classes)
    for class_id in range(num_classes):
        true_mask = (target == class_id)
        pred_mask = (pred == class_id)
        intersection = torch.sum(true_mask*pred_mask)
        union = torch.sum(true_mask)+torch.sum(pred_mask)

        dice_scores[class_id] = (2. * intersection + smooth) / (union + smooth)

    return dice_scores

def IoU(pred, target, smooth=1e-6):

    # print("pred shape in IoU: ", pred.shape, pred[0,1,:])
    # print("target shape:",target.shape)
    pred = pred[0]
    pred = pred.argmax(dim=0)  # Convert to class indices
    num_classes = pred.shape[0]
    IoU_scores = np.zeros(num_classes)
    for class_id in range(num_classes):
        # print(pred)
        intersection = torch.sum((target==class_id)&(pred==class_id))
        union = abs(torch.sum((target==class_id)|(pred==class_id)))
        if torch.sum(union == 0) > 0:
            IoU_scores[class_id] = np.nan
        else:
            IoU_scores[class_id] = intersection.item() / (union.item() + smooth)
    return IoU_scores

def save_img(img, img_path):
    cv2.imwrite(img_path, img)
def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    # if n_dim == 4:
    #     n_img = len(tensor)
    #     img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
    #     img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    if n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def main():
    batch_size = 8  # Small batch size for limited data
    images_dir = os.path.join(cfg.root_path,cfg.images_dir)
    masks_dir = os.path.join(cfg.root_path,cfg.masks_dir)

    train_dataloader, val_dataloader, test_dataloader = return_dataset(images_dir, masks_dir,batch_size)
   
    # Load SAM2 and freeze the encoder
    sam = sam_model_registry["vit_b"](checkpoint=f"{cfg.checkpoint_dir}/sam_vit_b_01ec64.pth")
    predictor = SamPredictor(sam)

    # training
    if cfg.train:

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
        patience = 10  # Stop if no improvement for 10 epochs
        epochs_without_improvement = 0

        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, verbose=True
        )
        mapping = get_mask_mappings()
        epoch_size = 100
        for epoch in range(epoch_size):
            # Training phase
            loss = 0.0
            sam.train()
            i=0
            for images, masks, file_name in train_dataloader:

                # Adjust tensor shape to n,64,64,c
                images = np.repeat(images,2,axis=3)
                images = np.repeat(images,2,axis=2)

                images = torch.tensor(images,dtype=torch.float32)
                # Forward pass
                image_embeddings = sam.image_encoder(images)
                
                masks_label = replace_with_labels(masks,mapping)        
                points,labels = generate_prompts_from_masks(masks_label)
                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    points = (points,labels),
                    boxes = None,
                    masks = None
                )

                outputs,_ = sam.mask_decoder(image_embeddings=image_embeddings,
                                            image_pe=sam.prompt_encoder.get_dense_pe(),
                                            sparse_prompt_embeddings=sparse_embeddings,
                                            dense_prompt_embeddings=dense_embeddings,
                                            multimask_output=True)
                masks=masks.permute(0,3,1,2)
                masks=F.interpolate(masks, size=(256, 256), mode='bilinear', align_corners=False)
                masks = masks.float()

                loss = criterion(outputs, masks)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i+=1
                print(f'step {i}/{np.round(len(train_dataloader)/batch_size)}, loss: {loss:.4f}')
            loss += loss
            print(f'epoch {epoch+1}/epoch_size,\nloss: {loss:.4f}')

            # Validation phase
            if (cfg.val_size>0):
                sam.eval()
                val_iou = 0.0
                with torch.no_grad():
                    for images, masks, file_names in val_dataloader:
                        # Adjust tensor shape to n,64,64,c
                        images = np.repeat(images,2,axis=3)
                        images = np.repeat(images,2,axis=2)
                        images = torch.tensor(images,dtype=torch.float32)
                        # Forward pass
                        image_embeddings = sam.image_encoder(images)
                        masks_label = replace_with_labels(masks,mapping)        
                        points,labels = generate_prompts_from_masks(masks_label)
                        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                            points = (points,labels),
                            boxes = None,
                            masks = None
                        )
                        outputs,_ = sam.mask_decoder(image_embeddings=image_embeddings,
                                                    image_pe=sam.prompt_encoder.get_dense_pe(),
                                                    sparse_prompt_embeddings=sparse_embeddings,
                                                    dense_prompt_embeddings=dense_embeddings,
                                                    multimask_output=True)
                        # print('OUT SHAPE: ',outputs.shape)
                        val_iou += IoU(outputs, masks)
                        val_dice += dice_score(outputs,masks)
                        os.makedirs(cfg.root_path+ '\\'+cfg.test_output_dir +'\\' , exist_ok=True)
                        save_img(tensor2img(masks[0].detach()[i].float().cpu()), cfg.root_path+ '\\'+cfg.test_output_dir +'\\'  + str(epoch)+'_' + f'{file_name[:-5]}Mask.png')
                        save_img(tensor2img(outputs[0].detach()[i].float().cpu()), cfg.root_path+ '\\'+cfg.test_output_dir +'\\'  + str(epoch)+'_' + f'{file_name[:-5]}Pred.png')
                        save_img(tensor2img(images[0].detach()[i].cpu()/255), cfg.root_path+ '\\'+cfg.test_output_dir +'\\'  + str(epoch)+'_' + f'{file_name[:-5]}.png')
                    
                val_iou /= len(val_dataloader)
                val_dice /= len(val_dice)
                val_iou_mean = val_iou.mean()  # Compute the mean across all elements
                val_dice_mean = val_dice.mean()  # Compute the mean across all elements            
                print(f'epoch {epoch}/{epoch_size},\nval_IoU: {val_iou_mean.item():.2f}, val_dice: {val_dice_mean.item():.2f}')
                val_iou_current=torch.nanmean(torch.tensor(val_iou)).item()
                scheduler.step(val_iou_current)  # Adjust LR based on validation IoU

                # Check validation IoU
                current_val_iou = val_iou
                if current_val_iou > best_val_iou:
                    best_val_iou = current_val_iou
                    epochs_without_improvement = 0
                    # best iou checkpoint
                    torch.save(sam.mask_decoder.state_dict(), f"{cfg.checkpoint_dir}/best_iris_sam2.pt")
                    print(f"Saved best model at epoch {epoch} with IoU {val_iou:.4f}")
                
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

            # Save periodic checkpoint
            if epoch % 20 == 0:
                torch.save({
                    "epoch": epoch,
                    "model": sam.mask_decoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, f"{cfg.checkpoint_dir}/epoch_{epoch}.pt")

            # Load the best model
            # checkpoint = torch.load(f"{cfg.checkpoint_dir}/epoch{epoch_size}.pt")
            # sam.mask_decoder.load_state_dict(checkpoint["model"])

            # optimizer.load_state_dict(checkpoint["optimizer_state"])
            # start_epoch = checkpoint["epoch"] + 1  # Resume training from next epoch
            if (cfg.test_size>0):

                best_model = torch.load(f"{cfg.checkpoint_dir}/best_iris_sam2.pt")
                sam.mask_decoder.load_state_dict(best_model["model_state"])
                optimizer.load_state_dict(best_model["optimizer_state"])

                # Evaluation
                sam.eval()
                with torch.no_grad():
                    for images, masks, filenames in test_dataloader:
                        image_embeddings = sam.image_encoder(images)
                        logits = sam.mask_decoder(image_embeddings)
                        preds = logits.argmax(dim=1)  # (H, W) class indices
                        for pred, name in zip(preds, filenames):
                            save_path = f"{cfg.test_output_dir}/{name}.png"
                            save_img(tensor2img(pred.detach()[i].float().cpu()), save_path)

if __name__ == '__main__':
    # Necessary for multiprocessing in Windows
    from torch.multiprocessing import set_start_method
    try:
        set_start_method('spawn')  # Ensure spawn method is used on Windows
    except RuntimeError:
        pass  # If the start method is already set, ignore the error

    main()  # Start the main function
