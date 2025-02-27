import os
import cv2
import numpy as np

def crop_with_mask(image_path, mask_path, output_dir):
    # Read the iris binary mask and the annotation
    image = cv2.imread(image_path) #annotation
    # print(image.shape)
    resize_image = cv2.resize(image, (256, 256)) # the same size as mask
    # cv2.imshow(resize_image)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Mask should be a grayscale image


    # --- METHOD 1: Morphological Closing (Best for small gaps/jagged edges) ---
    kernel_size = 3  # Increase kernel size for stronger smoothing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    processed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Ensure the mask is binary (0 or 255 values)
    _, mask_binary = cv2.threshold(processed_mask, 127, 255, cv2.THRESH_BINARY)

    # Get the center and radius of the circle in the mask
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Select largest contour
    largest_contour = max(contours, key=lambda c: cv2.contourArea(c))

    # Method 1: Minimal Enclosing Circle (OpenCV built-in)
    center_min, radius_min = cv2.minEnclosingCircle(largest_contour)
    center_min = tuple(map(int, center_min))
    radius_min = int(radius_min)

    # Method 2: Least Squares Circle Fit (mathematical best fit)
    points = largest_contour.squeeze().astype(np.float32)
    x = points[:, 0]
    y = points[:, 1]

    # Set up and solve linear system for circle parameters
    A = np.vstack([2*x, 2*y, np.ones_like(x)]).T
    b = x**2 + y**2
    a, c, d = np.linalg.lstsq(A, b, rcond=None)[0]
    center_lsq = (int(a), int(c))
    radius_lsq = int(np.sqrt(d + a**2 + c**2))

    # Method 3: Ellipse Fit (for near-circular shapes)
    ellipse = cv2.fitEllipse(largest_contour)
    (ell_center, (major, minor), angle) = ellipse
    is_circle = abs(major - minor) < 6  # Threshold for circularity check

    # Visualization
    result = cv2.cvtColor(processed_mask, cv2.COLOR_GRAY2BGR)

    # # Draw minimal enclosing circle (red)
    # cv2.circle(result, center_min, radius_min, (0, 0, 255), 2)

    # # Draw least squares circle (green)
    # cv2.circle(result, center_lsq, radius_lsq, (0, 255, 0), 2)

    # Draw ellipse fit if nearly circular (blue)
    if not is_circle:
        # Create a mask with the same size as the original image
        ell_result = np.zeros_like(resize_image[:, :,0])
        cv2.ellipse(result, ellipse, (255, 0, 0), 2)

        # Draw a filled circle on the mask
        cv2.ellipse(ell_result, ellipse, 255, -1)

        cropped_result = cv2.bitwise_and(resize_image, resize_image, mask=ell_result)
        cv2.imshow(cropped_result)
        print('ecllipse')
        cv2.waitKey(0)
    else:
        circular_mask = np.zeros_like(resize_image[:, :, 0])
        # Draw a filled circle on the mask
        # cv2.circle(circular_mask, center_min, radius_min, (0, 0, 255), 2)
        cv2.circle(circular_mask, center_lsq, radius_lsq, 255, -1)
        cropped_result = cv2.bitwise_and(resize_image, resize_image, mask=circular_mask)
        cv2.imshow(cropped_result)
        print('circle lsq')
        cv2.waitKey(0)


    # Show results
    cv2.imshow(result)
    cv2.waitKey(0)

    # Prepare output filename
    base_filename = os.path.basename(image_path)
    output_filename = os.path.join(output_dir, f"{base_filename}")

    # Save the cropped image
    cv2.imwrite(output_filename, cropped_result)
    print(f"Saved: {output_filename}")

# dir = '/Volumes/My Passport/IRIS_ANNOTATION'
dir = '/Users/mollieyin/Downloads'
img_file_p = dir+'/001'
masks_dir = r"/Users/mollieyin/VSCode_workspace/predictions"
# annotation_dir = dir+'/annotations'
# output_dir = dir+'/path/train/cropped_annotations'
# os.makedirs(output_dir, exist_ok=True)
# crop_with_mask(img_file_p,mask_file_p,output_filename)

    # # Create a mask with the same size as the original image
    # circular_mask = np.zeros_like(resize_image[:, :,0])

    # # Draw a filled circle on the mask
    # cv2.circle(circular_mask, center_lsq, radius_lsq, 255, thickness=-1)

    # # Make sure the mask has the same data type as the image
    # circular_mask = circular_mask.astype(np.uint8)

    # # Crop the region of interest from the original image
    # # This will give a region the same size as the mask
    # cropped_image = cv2.bitwise_and(resize_image, resize_image, mask=circular_mask)

    # cv2.imshow(cropped_image)
    # cv2.waitKey(0)


    # # Ensure the mask is binary (0 or 255 values)
    # _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # # Get the center and radius of the circle in the mask
    # contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # if contours:
    #     # Get the bounding box of the largest contour (circle mask)
    #     (x, y), radius = cv2.minEnclosingCircle(contours[0])
    #     center = (int(x), int(y))
    #     radius = int(radius)

    #     # Create a mask with the same size as the original image
    #     circular_mask = np.zeros_like(resize_image[:, :, 0])

    #     # Draw a filled circle on the mask
    #     cv2.circle(circular_mask, center, radius, 255, thickness=-1)

    #     # Make sure the mask has the same data type as the image
    #     circular_mask = circular_mask.astype(np.uint8)

    #     # Crop the region of interest from the original image
    #     # This will give a region the same size as the mask
    #     cropped_image = cv2.bitwise_and(resize_image, resize_image, mask=circular_mask)

    #     cv2.imshow(cropped_image)


    #     # Prepare output filename
    #     base_filename = os.path.basename(image_path)
    #     output_filename = os.path.join(output_dir, f"cropped_anno_{base_filename}")

    #     # Save the cropped image
    #     # cv2.imwrite(output_filename, cropped_image)
    #     print(f"Saved: {output_filename}")
    # else:
    #     print(f"Mask not valid for: {image_path}")

# def process_images_and_masks(image_dir, mask_dir, output_dir):
#     # Ensure output directory exists
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # Get all image and mask filenames
#     image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.bmp','.tiff'))]
#     mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.tiff'))]

#     # Sort images and masks
#     image_files.sort()
#     mask_files.sort()

#     # Get mask extension
#     ext_mask = mask_files[0].split('.')[-1]

#     # Process each image and mask
#     for image_filename in image_files:
#         # Check if a mask exists with the same base name (excluding extensions)
#         ID = os.path.splitext(image_filename)[0]
#         mask_filename = "pred_" + ID + f".{ext_mask}"
#         if mask_filename in mask_files:
#             image_path = os.path.join(image_dir, image_filename)
#             # print(image_path)
#             mask_path = os.path.join(mask_dir, mask_filename)
#             # print(mask_path)

#             # Call the cropping function
#             crop_with_mask(image_path, mask_path, output_dir)
#         else:
#             print(f"Mask not found for image: {image_filename}")

# dir = '/Volumes/My Passport/IRIS_ANNOTATION'
# masks_dir = dir+"/path/train/pred_iris_masks"
# annotation_dir = dir+'/annotations'
# output_dir = dir+'/path/train/cropped_annotations'
# os.makedirs(output_dir, exist_ok=True)

# process_images_and_masks(annotation_dir,masks_dir, output_dir)

"""*crop iris*"""

import numpy as np
from PIL import Image, ImageDraw
import cv2
import os
import matplotlib.pyplot as plt
# dir = '/Volumes/My Passport/IRIS_ANNOTATION'
# img_file_p='/images'
# mask_file_p='/annotations'
cropped_dir = "/Users/mollieyin/VSCode_workspace/cropped_iris"
os.makedirs(cropped_dir, exist_ok=True)
# image_files = [os.path.join(img_file_p, f) for f in os.listdir(dir+img_file_p) if f.endswith(('.bmp', '.png', '.jpg', '.jpeg','.tiff'))]
# mask_files = [os.path.join(mask_file_p, f) for f in os.listdir(dir+mask_file_p) if f.endswith(('.bmp', '.png', '.jpg', '.jpeg', '.tiff'))]
image_files = []
# store all files into image list
for f in os.listdir(img_file_p):
  if f.endswith(('.bmp', '.png', '.jpg', '.jpeg','.tiff')):
    image_files.append(os.path.join(img_file_p, f))
mask_files = []
for f in os.listdir(masks_dir):
  if f.endswith(('.bmp', '.png', '.jpg', '.jpeg', '.tiff')):
    mask_files.append(os.path.join(masks_dir, f))

# sort image files and masks
image_files.sort()
mask_files.sort()

for img,mask in zip(image_files,mask_files):
  img_org = cv2.imread(img)
  img_mask = cv2.imread(mask)
  print(img_org.shape)
  print(img_mask.shape)
  base_filename = os.path.basename(img)
  resize_image = cv2.resize(img_org, (512,512), interpolation = cv2.INTER_AREA)
  img_mask = cv2.resize(img_mask, (512,512), interpolation = cv2.INTER_AREA)

  # --- METHOD 1: Morphological Closing (Best for small gaps/jagged edges) ---
  kernel_size = 3  # Increase kernel size for stronger smoothing
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
  processed_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
  processed_mask = cv2.cvtColor(processed_mask, cv2.COLOR_BGR2GRAY)

  # Ensure the mask is binary (0 or 255 values)
  _, mask_binary = cv2.threshold(processed_mask, 127, 255, cv2.THRESH_BINARY)

  # Get the center and radius of the circle in the mask
  contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  # Select largest contour
  largest_contour = max(contours, key=lambda c: cv2.contourArea(c))

  # Method 1: Minimal Enclosing Circle (OpenCV built-in)
  center_min, radius_min = cv2.minEnclosingCircle(largest_contour)
  center_min = tuple(map(int, center_min))
  radius_min = int(radius_min)

  # Method 2: Least Squares Circle Fit (mathematical best fit)
  points = largest_contour.squeeze().astype(np.float32)
  x = points[:, 0]
  y = points[:, 1]

  # Set up and solve linear system for circle parameters
  A = np.vstack([2*x, 2*y, np.ones_like(x)]).T
  b = x**2 + y**2
  a, c, d = np.linalg.lstsq(A, b, rcond=None)[0]
  center_lsq = (int(a), int(c))
  radius_lsq = int(np.sqrt(d + a**2 + c**2))

  # Method 3: Ellipse Fit (for near-circular shapes)
  ellipse = cv2.fitEllipse(largest_contour)
  (ell_center, (major, minor), angle) = ellipse
  is_circle = abs(major - minor) < 6  # Threshold for circularity check

  # Visualization
  result = cv2.cvtColor(processed_mask, cv2.COLOR_GRAY2BGR)

  # # Draw minimal enclosing circle (red)
  # cv2.circle(result, center_min, radius_min, (0, 0, 255), 2)

  # # Draw least squares circle (green)
  # cv2.circle(result, center_lsq, radius_lsq, (0, 255, 0), 2)

  # Draw ellipse fit if nearly circular (blue)
  if not is_circle:
      # Create a mask with the same size as the original image
      ell_result = np.zeros_like(resize_image[:, :,0])
      cv2.ellipse(result, ellipse, (255, 0, 0), 2)

      # Draw a filled circle on the mask
      cv2.ellipse(ell_result, ellipse, 255, -1)

      cropped_result = cv2.bitwise_and(resize_image, resize_image, mask=ell_result)
    #   cv2.imshow("",cropped_result)
      print('ecllipse')
    #   cv2.waitKey(0)
  else:
      circular_mask = np.zeros_like(resize_image[:, :, 0])
      # Draw a filled circle on the mask
      # cv2.circle(circular_mask, center_min, radius_min, (0, 0, 255), 2)
      cv2.circle(circular_mask, center_lsq, radius_lsq, 255, -1)
      cropped_result = cv2.bitwise_and(resize_image, resize_image, mask=circular_mask)
    #   cv2.imshow("",cropped_result)
      print('circle lsq')
    #   cv2.waitKey(0)


  # # Show results
  # cv2.imshow(result)
  # cv2.waitKey(0)

  # ##Resizing images
  # img_org = cv2.resize(img_org, (1024,1024), interpolation = cv2.INTER_NEAREST)
  # img_mask = cv2.resize(img_mask, (1024,1024), interpolation = cv2.INTER_NEAREST)
  # # cv2.imshow(img_org)
  # # cv2.imshow(img_mask)

  new=cv2.bitwise_and(cropped_result, resize_image)
  # cv2.imshow(new)
  base_filename = os.path.basename(img)
  output_filename = os.path.join(cropped_dir, f"{base_filename}")

  # Save the cropped image
  cv2.imwrite(output_filename, new)
  print(f"Saved: {output_filename}")
  # plt.imshow(new)