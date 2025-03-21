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
        img_size = 512,
        # device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        # model
        model='SAM2',
        checkpoint_dir='/home/wvubiometrics/Documents/Molly/IRIS_SEG_SAM2/checkpoints',
        output_dir="/home/wvubiometrics/Documents/Molly/IRIS_SEG_SAM2/SAM2_output",
        test_output_dir="/home/wvubiometrics/Documents/Molly/IRIS_SEG_SAM2/SAM2test_pred",
        test_size=0.15,
        val_size=0.15,
        train=True
    )
    self.model = args.model
    self.device = args.device
    self.checkpoint_dir = args.checkpoint_dir
    self.output_dir = args.output_dir
    self.test_output_dir = args.test_output_dir
    self.train = args.train
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
    create a dataset from list of paths
    """

    def __init__(self,image_list,mask_list,transform):
        self.image_list = image_list
        self.mask_list = mask_list
        self.transform = transform
    def __len__(self):
        return len(self.mask_list)
    def __getitem__(self, index):
        image = self.image_list[index]
        mask = self.mask_list[index]
        if image is None:
            raise ValueError("Error loading image. Check the file path.")
        if mask is None:
            raise ValueError("Error loading mask. Check the file path.")
        
        if self.transform:
            sample = self.transform((image, mask))  # Pass as tuple
        else:
            # convert to tensor
            sample = {'image': torch.tensor(image), 'mask': torch.tensor(mask)}
            

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
        return {'image': sample['image'], 'mask': sample['mask']}

# transform
class ToTensorAndNormalize:
    def __init__(self, img_size):
        self.img_size = img_size  # Set the desired image size
        self.random_flip = transforms.RandomHorizontalFlip(0.5)

    def __call__(self,sample):

        image, mask = sample
        # convert to tensor
        image_tensor = torch.tensor(image)
        mask_tensor = torch.tensor(mask)

        # resize image and mask
        image = cv2.resize(image_tensor.numpy(), (self.img_size, self.img_size))
        mask = cv2.resize(mask_tensor.numpy(), (self.img_size, self.img_size))
        # print('mask tensor shape; ', mask.shape)

        ### histogram equalization

        # Apply Gaussian Blur to reduce noise while preserving edges
        blurred_image = cv2.GaussianBlur(image, (1, 1), 0)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced_image = clahe.apply(blurred_image)

        image = np.expand_dims(enhanced_image,axis=0) # color dim expected 3 channels in image encoder
        image = np.repeat(image,3,axis=0)
        image_tensor = torch.tensor(image)
        mask_tensor = torch.tensor(mask)
        # print(image_tensor.shape,image_tensor.shape)

        # normalize
        image = transforms.Normalize(mean=[0.5], std=[0.5])
        
        normalized = image(image_tensor.to(torch.float32))

        return {'image': normalized, 'mask': mask_tensor}
    
##############################
### split the full dataset     
def split_subset(dataset):
    num_data = len(dataset)
    print('dataset len:',num_data)
    val_data_size = np.ceil(cfg.val_size*num_data).astype(int)
    test_data_size = np.ceil(cfg.test_size*num_data).astype(int)
    train_data_size = num_data - val_data_size - test_data_size

    # print('the training size is :',train_data_size)
    # print('the val and test size is : ',val_data_size,test_data_size)

    train_subset, val_subset, test_subset = random_split(
        dataset, [train_data_size,val_data_size,test_data_size],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    )
    # print('a train subset contains: ', train_subset[0])

    return train_subset, val_subset, test_subset

# Create the datasets
def return_dataset(image_dir, mask_dir,batch_size):

    img_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
    mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
 
    bad = "/run/media/wvubiometrics/My Passport/IRIS_ANNOTATION/annotations/06234d25.png"
    while bad in mask_paths:
        mask_paths.remove(bad)
    mask_paths.sort()

    # read image
    images = [cv2.imread(os.path.join(image_dir, path),0) for path in img_paths] # grayscale
    masks = [cv2.imread(os.path.join(mask_dir, path), 1) for path in mask_paths]

    # print('masks: ',masks[0].shape)


    if cfg.train:
        transform = ToTensorAndNormalize(cfg.img_size)
        dataset = CreateDataset(images,masks,transform=transform)
        # # calculate class distributions

    else:
        dataset = CreateDataset(images,masks,None)

    # Load iris subset
    train_dataset, val_dataset, test_dataset = split_subset(dataset)

    train_sample = ReturnDataset(train_dataset)
    val_sample = ReturnDataset(val_dataset)
    test_sample = ReturnDataset(test_dataset)
    # print('train len: ', len(train_sample))
    # for batch in train_sample:
    #     print(batch['mask'].shape)


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

def extract_mask_regions(masks, mappings):
    '''
    Args: mask is (B,H,W,C)
          mappings is a dict of mask labels and colors
            
    Return: binary masks ((B,H,W))
    '''
    masks = masks.numpy()  ## convert it to numpy array
    segment_masks = np.zeros(masks.shape,dtype=np.uint8)

    for key,value in mappings.items():
        id,color = value
        color=np.uint8(color)
        if np.all(color == [0,0,0]): # if not labeled
            continue
        print(color)
        find_color = np.any(masks==color)
        if find_color:
            match = np.where(masks==color)
            segment_masks[match] = id
            print(match)


    return segment_masks

'''
convert mask into points
'''
###############
def mask_to_points(binary_mask,num_points=10):
    """
    Args: mask (numpy.ndarray): binary masks of shape (h,w)

    Returns: array of points (N,2) where each point is (x,y)
    """
    y,x=np.where(binary_mask == 1)
    points=np.column_stack((x,y))
    return points

def prep_prompts(segm_masks=dict):
    '''
    Args: segm_masks is a dict contains labels and binary masks array
    Return: a prompt contains label and points
    '''
    prompt = {}
    for label,mask in segm_masks:
        mask_p = mask_to_points(mask)
        if len(mask_p)>0:
            prompt[label]=np.array(mask_p)
    return prompt

def generate_prompts_from_masks(mask,num_points=10):
    """Generate random foreground/background points from masks.
        Args:
            masks: (B,H,W) ground-truth binary masks.
            num_points: Number of points to sample.
        Returns: 
            points:(B,N,2) normalized coordinates
            labels: (B,N) point labels (1=foreground, 0=background)
    """
    B,H,W = mask.shape
    points =[]
    labels = []
    for i in range(B):
        fg_coords = torch.nonzero(mask[i])
        fg_indices = torch.randperm(fg_coords.shape[0])[:num_points//2]
        fg_points = fg_coords[fg_indices].float()/torch.tensor([H,W])
        bg_coords = torch.nonzero(mask[i]==0)
        bg_indices = torch.randperm(bg_coords.shape[0])[:num_points//2]
        bg_points = fg_coords[bg_indices].float()/torch.tensor([H,W])
    # combine
    points.append(torch.cat([fg_points,bg_points]))
    labels.append(torch.cat([torch.ones(num_points//2,torch.zeros(num_points//2))]))
    
    return torch.stack(points),torch.stack(labels)

def segm_with_prmpt(image,mask):
    segment_masks = extract_mask_regions(mask)
    prompt = prep_prompts(segment_masks)
    predictor.set_image(image)
    masks,_,_ = predictor.predict(
        point_coords=prompt,
        point_labels=np.ones(len(prompt)),
        multimask_output=False # single mask per prompt
    )
    return masks

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

'''
main loop
'''
# training

batch_size = 8  # Small batch size for limited data
images_dir = os.path.join(cfg.root_path,cfg.images_dir)
masks_dir = os.path.join(cfg.root_path,cfg.masks_dir)

train_dataloader, val_dataloader, test_dataloader = return_dataset(images_dir, masks_dir,batch_size)
mappings = get_mask_mappings()

# for batch in train_dataloader:
#     image,mask = batch["image"],batch["mask"]
#     print('batch in train loader: ',image.shape, mask.shape)

# Load SAM2 and freeze the encoder
sam = sam_model_registry["vit_b"](checkpoint=f"{cfg.checkpoint_dir}/sam_vit_b_01ec64.pth")
predictor = SamPredictor(sam)

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
patience = 10  # Stop if no improvement for 15 epochs
epochs_without_improvement = 0

scheduler = ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5, verbose=True
)
mapping = get_mask_mappings()
for epoch in range(100):
    # Training phase
    loss = 0.0
    sam.train()
    i=0
    for batch in train_dataloader:
        images,masks = batch["image"],batch["mask"]
        # print(masks.shape)
        ##################

        # Adjust tensor shape to n,64,64,c
        images = np.repeat(images,2,axis=3)
        images = np.repeat(images,2,axis=2)

        images = torch.tensor(images,dtype=torch.float32)
        # Forward pass
        image_embeddings = sam.image_encoder(images)
        masks = extract_mask_regions(masks,mapping)        
        points,labels = generate_prompts_from_masks(masks)#########
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points = (points,labels),
            boxes = None,
            masks = None
        )

        outputs,_ = sam.mask_decoder(image_embeddings=image_embeddings,
                                     image_pe=sam.prompt_encoder.get_dense_pe(),
                                     sparse_prompt_embeddings=sparse_embeddings,
                                     dense_prompt_embeddings=dense_embeddings)
        loss = criterion(outputs, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i+=1
        print(f'step {i}/8, loss: {loss:.4f}')
    loss += loss
    print(f'epoch {epoch}/100,\nloss: {loss:.4f}')

    # Validation phase
    sam.eval()
    val_iou = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            images,masks = batch["image"],batch["mask"]
            outputs = sam.mask_decoder(sam.image_encoder(images))
            val_iou += IoU(outputs, masks)
            val_dice += dice_score(outputs,masks)
    val_iou /= len(val_dataloader)
    val_dice /= len(val_dice)
    print(f'epoch {epoch}/100,\nval_IoU: {val_iou:.2f},val_dice: {val_dice:.2f}')

    scheduler.step(val_iou)  # Adjust LR based on validation IoU

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
    if epoch % 5 == 0:
        torch.save({
            "epoch": epoch,
            "model": sam.mask_decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, f"{cfg.checkpoint_dir}/epoch_{epoch}.pt")

# Load the best model
checkpoint = torch.load(f"{cfg.checkpoint_dir}/epoch_10.pt")
sam.mask_decoder.load_state_dict(checkpoint["model"])

optimizer.load_state_dict(checkpoint["optimizer_state"])
# start_epoch = checkpoint["epoch"] + 1  # Resume training from next epoch
best_model = torch.load(f"{cfg.checkpoint_dir}/best_iris_sam2.pt")
sam.mask_decoder.load_state_dict(best_model["model_state"])
optimizer.load_state_dict(best_model["optimizer_state"])
# Evaluation
predictions = []
sam.eval()
with torch.no_grad():
    for batch in test_dataloader:
      test_image, test_mask = batch['image'],batch['mask']
      image_embeddings = sam.image_encoder(test_image)
      logits = sam.mask_decoder(image_embeddings)
      pred_mask = logits.argmax(dim=1)  # (H, W) class indices
      predictions.append(pred_mask)
cv2.imshow("Prediction", predictions[0].cpu().numpy())
cv2.imshow("Test Mask",test_dataloader[0]['mask'].numpy())
# cv2.imwrite('test_pred.png', predictions[0].cpu().numpy())
# # cv2.imwrite('test_mask.png', test_dataloader[0]['mask'].numpy())
# Save the predictions
for i, pred_mask in enumerate(predictions):
    pred_mask = pred_mask.cpu().numpy()
    cv2.imwrite(f"test_pred/{i}.png", pred_mask)
    print(f"Saved prediction {i}")
print("All predictions saved!")