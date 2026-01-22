import cv2
import os

def blur_average(img, ksize=5):
    return cv2.blur(img, (ksize, ksize))

def blur_gaussian(img, ksize=5, sigma=0):
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

def blur_median(img, ksize=5):
    return cv2.medianBlur(img, ksize)

# Output directory
output_dir = "/home/leprieto/dev/a2-vision/smooth-filter/output"
os.makedirs(output_dir, exist_ok=True)

# Load images (BGR)
img1 = cv2.imread('/home/leprieto/dev/a2-vision/practica2/data/industrial/01_large.jpg')
img2 = cv2.imread('/home/leprieto/dev/a2-vision/practica2/data/medical/CXR165_IM-0427-1001.png')

if img1 is None or img2 is None:
    raise FileNotFoundError("One or more images could not be loaded.")

# Apply filters
img1_avg  = blur_average(img1, ksize=25)
img1_gaus = blur_gaussian(img1, ksize=25)

img2_avg  = blur_average(img2, ksize=7)
img2_gaus = blur_gaussian(img2, ksize=7)

# Save images (original resolution preserved)
cv2.imwrite(f"{output_dir}/img1_original.png", img1)
cv2.imwrite(f"{output_dir}/img1_avg.png", img1_avg)
cv2.imwrite(f"{output_dir}/img1_gaussian.png", img1_gaus)

cv2.imwrite(f"{output_dir}/img2_original.png", img2)
cv2.imwrite(f"{output_dir}/img2_avg.png", img2_avg)
cv2.imwrite(f"{output_dir}/img2_gaussian.png", img2_gaus)
