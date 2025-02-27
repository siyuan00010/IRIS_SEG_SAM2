import os
import cv2
import numpy as np

def crop_with_circular_mask(image_path, mask_path, output_dir):
    # Read the image and the mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Mask should be a grayscale image

    # Ensure the mask is binary (0 or 255 values)
    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Get the center and radius of the circle in the mask
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the bounding box of the largest contour (circle mask)
        (x, y), radius = cv2.minEnclosingCircle(contours[0])
        center = (int(x), int(y))
        radius = int(radius)

        # Create a mask with the same size as the original image
        circular_mask = np.zeros_like(mask_binary)

        # Draw a filled circle on the mask
        cv2.circle(circular_mask, center, radius, 255, thickness=-1)

        # Crop the region of interest from the original image
        # This will give a region the same size as the mask
        cropped_image = cv2.bitwise_and(image, image, mask=circular_mask)

        # Prepare output filename
        base_filename = os.path.basename(image_path)
        output_filename = os.path.join(output_dir, f"cropped_{base_filename}")

        # Save the cropped image
        cv2.imwrite(output_filename, cropped_image)
        print(f"Saved: {output_filename}")
    else:
        print(f"Mask not valid for: {image_path}")

def process_images_and_masks(image_dir, mask_dir, output_dir):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all image and mask filenames
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

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
dir = '/Volumes/IrisAnnotation'
image_dir = dir+'/annotations'
mask_dir = r"/Users/mollieyin/VSCode_workspace/predictions"
output_dir = r"/Users/mollieyin/VSCode_workspace/cropped_annotations"

process_images_and_masks(image_dir, mask_dir, output_dir)