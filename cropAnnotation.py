import os
import cv2
import numpy as np

def crop_with_circular_mask(image_path, mask_path, output_dir):
    # Read the image and the mask
    image = cv2.imread(image_path) #annotation
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Mask should be a grayscale image
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
  
    cropped_image = cv2.bitwise_and(image, image, mask=mask)
    # print(f"Mask unique values: {np.unique(cropped_image)}")

    # Prepare output filename
    base_filename = os.path.basename(image_path)
    output_filename = os.path.join(output_dir, f"{base_filename}")

    # Save the cropped image
    cv2.imwrite(output_filename, cropped_image)
    print(f"Saved: {output_filename}")

def process_images_and_masks(image_dir, mask_dir, output_dir):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all image and mask filenames
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.tiff'))]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.tiff'))]

    # Process each image and mask
    for image_filename in image_files:
        # Check if a mask exists with the same base name (excluding extensions)
        mask_filename = os.path.splitext(image_filename)[0] + os.path.splitext(mask_files[0])[1]  # Assume same extension for both files
        if mask_filename in mask_files:
            image_path = os.path.join(image_dir, image_filename)
            mask_path = os.path.join(mask_dir, mask_filename)

            # Call the cropping function
            crop_with_circular_mask(image_path, mask_path, output_dir)
        else:
            print(f"Mask not found for image: {image_filename}")

# Example usage
dir = 'C:/Users/siyuan/Documents/IRIS_DATA'
mask_dir = dir+'/binary_cropped_iris'
image_dir = dir+'/annotations'
output_dir =dir+'/cropped_annotations'

process_images_and_masks(image_dir, mask_dir, output_dir)