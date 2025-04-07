
from SAM2_fine_tune import Config, replace_with_labels,return_dataset, generate_prompts_from_masks,get_mask_mappings,IoU,dice_score,label_to_rgb,save_img,tensor2img
import torch
import torch.nn as nn
import numpy as np
import os
from segment_anything import sam_model_registry
from segment_anything.modeling.mask_decoder import MLP
from torch.optim.lr_scheduler import ReduceLROnPlateau

def main():
    cfg = Config()
    batch_size = 2  # Small batch size for limited data
    images_dir = os.path.join(cfg.root_path,cfg.images_dir).replace('\\','/')
    masks_dir = os.path.join(cfg.root_path,cfg.masks_dir).replace('\\','/')
    train_dataloader, val_dataloader, test_dataloader = return_dataset(images_dir, masks_dir,batch_size)
   
    # # Load SAM2 and freeze the encoder
    # Load the checkpoint state_dict
    model_state_dict = torch.load(f"{cfg.checkpoint_dir}/sam_vit_b_01ec64.pth")

    # Extract only the encoder state_dict
    encoder_state_dict = {k: v for k, v in model_state_dict.items() if 'encoder' in k}

    # Initialize the SAM model (encoder and decoder)
    sam = sam_model_registry["vit_b"](checkpoint=None)  # Don't load the full checkpoint here

    # Load the encoder weights into the model
    sam.image_encoder.load_state_dict(encoder_state_dict, strict=False)  # Strict=False will ignore missing keys

    # Adjust the mask_tokens and IOU head to match the number of classes (17)
    sam.mask_decoder.mask_tokens = nn.Embedding(cfg.num_classes, 256)  # Change to 17 tokens
    sam.mask_decoder.iou_prediction_head =  nn.Sequential(
                                            nn.Linear(256, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, cfg.num_classes)  # Update for 17 classes
                                            )
    
    # training
    if cfg.train:
        # Freeze the image encoder (no gradients)
        # for param in sam.image_encoder.parameters():
        #     param.requires_grad = False # <--- unfreeze for >10k images
        for param in sam.image_encoder.blocks[-1].parameters():
            param.requires_grad = True
        # Unfreeze the mask decoder (fine-tune it)
        for param in sam.mask_decoder.parameters():
            param.requires_grad = True  # <--- Only change from earlier!

        # Loss and optimizer (only affects the decoder)
        criterion = torch.nn.CrossEntropyLoss()
        l1_loss = nn.L1Loss()
        optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=1e-5) # small lr for fine tuning

        best_val_iou = 0.0
        patience = 10  # Stop if no improvement for 10 epochs
        epochs_without_improvement = 0

        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, verbose=True
        )
        mapping = get_mask_mappings()
        epoch_size = 2
        print(f"epoch 0/{epoch_size}")
        for epoch in range(epoch_size):
            # Training phase
            loss = 0.0
            sam.train()
            i=0
            for batch in train_dataloader:
                images, masks, file_name = batch['image'],batch['mask'],batch['file_name']
                # Adjust tensor shape to n,64,64,c
                images = np.repeat(images,cfg.img_size/64,axis=3)
                images = np.repeat(images,cfg.img_size/64,axis=2)
                images = images.clone().detach().to(torch.float32) #float for image encoder
                # Forward pass
                image_embeddings = sam.image_encoder(images)
                
                masks_label = replace_with_labels(masks,mapping)     
                masks_label = torch.from_numpy(masks_label).clone().detach()  
                points,labels = generate_prompts_from_masks(masks_label,total_points=5)
                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    points=(points, labels),
                    boxes=None,
                    masks=None
                )

                outputs,_ = sam.mask_decoder(image_embeddings=image_embeddings,
                                            image_pe=sam.prompt_encoder.get_dense_pe(),
                                            sparse_prompt_embeddings=sparse_embeddings,
                                            dense_prompt_embeddings=dense_embeddings,
                                            multimask_output=True)
                
                masks_label = masks_label.long()
                loss = criterion(outputs, masks_label) + l1_loss(outputs, masks_label)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i+=1
                print(f'step {i}/{np.round(len(train_dataloader.dataset)/batch_size).astype(int)}, loss: {loss:.3f}')
            loss += loss
            print(f'epoch {epoch+1}/{epoch_size}, loss: {loss:.3f}')

            # Validation phase
            if (cfg.val_size>0):
                sam.eval()
                val_iou = 0.0
                val_dice = 0.0
                with torch.no_grad():
                    for batch in val_dataloader:
                        images, masks, file_name = batch['image'],batch['mask'],batch['file_name']
                        # Adjust tensor shape to n,64,64,c
                        img = np.repeat(images,cfg.img_size/64,axis=3)
                        img = np.repeat(img,cfg.img_size/64,axis=2)
                        img = img.clone().detach().to(torch.float32)
                        # Forward pass
                        image_embeddings = sam.image_encoder(img)
                        masks_label = replace_with_labels(masks,mapping)  
                        masks_label = torch.from_numpy(masks_label).clone().detach()    
                        points,labels = generate_prompts_from_masks(masks_label)
                        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                            points = (points,labels),
                            boxes = None,
                            masks = None
                        )
                        raw_logits,_ = sam.mask_decoder(image_embeddings=image_embeddings,
                                                    image_pe=sam.prompt_encoder.get_dense_pe(),
                                                    sparse_prompt_embeddings=sparse_embeddings,
                                                    dense_prompt_embeddings=dense_embeddings,
                                                    multimask_output=True)
                        # print(raw_logits.shape)##[B,C,H,W]
                        num_classes = raw_logits.shape[1]
                        # Apply softmax to convert logits to probabilities for each class
                        probabilities = torch.softmax(raw_logits, dim=1)
                        # Convert to predicted class by taking argmax along the class dimension
                        pred_class = torch.argmax(probabilities, dim=1) ### [B,H,W]
                        # Compute IoU for each class by comparing the predicted class and ground truth class
                        mean_iou, _ = IoU(pred_class,masks_label,num_classes)
                        val_iou += mean_iou
                        mean_dice,dice = dice_score(pred_class,masks_label,num_classes)
                        val_dice += mean_dice
                        os.makedirs(cfg.test_output_dir +'/' , exist_ok=True)

                        for b in range(pred_class.shape[0]):
                            outputs = label_to_rgb(pred_class[b],mapping)
                            image = images[b].permute(1, 2, 0).cpu().numpy()
                            output_img = tensor2img(outputs)
                            mask_img = tensor2img(masks[b])

                            save_img(output_img, f"{cfg.test_output_dir}/{epoch}_{file_name[b]}_pred.png")
                            save_img(image, f"{cfg.test_output_dir}/{epoch}_{file_name[b]}.png")
                            save_img(mask_img, f"{cfg.test_output_dir}/{epoch}_{file_name[b]}_Mask.png")
                val_iou /= len(val_dataloader.dataset)
                val_dice /= len(val_dataloader.dataset)
                val_iou_mean = val_iou.mean()  # Compute the mean across all elements
                val_dice_mean = val_dice.mean()  # Compute the mean across all elements            
                print(f' val_IoU: {val_iou_mean.item():.2f}, val_dice: {val_dice_mean.item():.2f}')
                current_val_iou=torch.nanmean(val_iou.clone().detach()).item()
                scheduler.step(current_val_iou)  # Adjust LR based on validation IoU

                # Check validation IoU
                best_val_iou = val_iou_mean.cpu().numpy()
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
            if epoch % 2 == 0:
                torch.save({
                    "epoch": epoch,
                    "model": sam.mask_decoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, f"{cfg.checkpoint_dir}/epoch_{epoch}.pt")

    if (cfg.test_size>0):

        pretrained = torch.load(f"{cfg.checkpoint_dir}/epoch_0.pt")
        sam.mask_decoder.load_state_dict(pretrained["model"])
        optimizer.load_state_dict(pretrained["optimizer"])

        # Evaluation
        sam.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                images, masks, file_name = batch['image'],batch['mask'],batch['file_name']
                # Adjust tensor shape to n,64,64,c
                img = np.repeat(images,cfg.img_size/64,axis=3)
                img = np.repeat(img,cfg.img_size/64,axis=2)
                img = img.clone().detach().to(torch.float32) #float for image encoder
                image_embeddings = sam.image_encoder(img)
                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                            points = None,
                            boxes = None,
                            masks = None
                        )
                logits,_ = sam.mask_decoder(image_embeddings=image_embeddings,
                                        image_pe=sam.prompt_encoder.get_dense_pe(),
                                        sparse_prompt_embeddings=sparse_embeddings,
                                        dense_prompt_embeddings=dense_embeddings,
                                        multimask_output=True)
                logits = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)  # (B,H, W) class indices
                for b in range(preds.shape[0]):
                            rgb_pred = label_to_rgb(preds[b],mapping)
                            save_path = f"{cfg.test_output_dir}/{file_name[b]}_pred.png"
                            save_img(tensor2img(rgb_pred), save_path)

if __name__ == '__main__':
    # Necessary for multiprocessing in Windows
    from torch.multiprocessing import set_start_method
    try:
        set_start_method('spawn')  # Ensure spawn method is used on Windows
    except RuntimeError:
        pass  # If the start method is already set, ignore the error

    main()  # Start the main function