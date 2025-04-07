 
"""*crop iris*"""

import numpy as np 
import cv2
import os 
def crop_with_mask(image_path, mask_path, output_dir, method):
    # Read the iris binary mask and the annotation
    image = cv2.imread(image_path) #annotation
    # print(image.shape)
    resize_image = cv2.resize(image, (256, 256)) # the same size as mask
    # cv2.imshow(resize_image)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Mask should be a grayscale image

    if method == 1:
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

      center_min, radius_min = cv2.minEnclosingCircle(largest_contour)
      center_min = tuple(map(int, center_min))
      radius_min = int(radius_min)
    if method ==2:
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

    if method ==3:
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

dir = 'C:/Users/siyuan/Documents/IRIS_DATA'
img_file_p = dir + '/images'
masks_dir = dir+'/binary_predictions'
cropped_dir = "C:/Users/siyuan/Documents/IRIS_DATA/binary_cropped_iris"
os.makedirs(cropped_dir, exist_ok=True)
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
  base_filename = os.path.basename(img)
  resize_image = cv2.resize(img_org, (256,256), interpolation = cv2.INTER_AREA)
  img_mask = cv2.resize(img_mask, (256,256), interpolation = cv2.INTER_AREA)

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

  new=cv2.bitwise_and(cropped_result, resize_image)
  
  # x, y, w, h = cv2.boundingRect(largest_contour)
    
  # # Crop the image using the bounding box
  # cropped_object = new[y:y+h, x:x+w]

  # new = cv2.resize(cropped_object,(256,256))
  # print(new.shape)

  # cv2.imshow(new)
  base_filename = os.path.basename(img)
  output_filename = os.path.join(cropped_dir, f"{base_filename}")

  # Save the cropped image
  cv2.imwrite(output_filename, new)
  print(f"Saved: {output_filename}")
  # plt.imshow(new)