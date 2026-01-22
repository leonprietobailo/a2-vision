import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy

def blur_average(img, ksize=5):
    return cv2.blur(img, (ksize, ksize))

def blur_gaussian(img, ksize=5, sigma=0):
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

def blur_median(img, ksize=5):
    return cv2.medianBlur(img, ksize)

def compute_psnr(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    return 20 * np.log10(255.0 / np.sqrt(mse))


def compute_ssim(img1, img2):
    gray1 = to_gray(img1)
    gray2 = to_gray(img2)
    return ssim(gray1, gray2)

def compute_entropy(img):
    gray = to_gray(img)
    return shannon_entropy(gray)

def compute_epi(original, transformed):
    gray_o = to_gray(original)
    gray_t = to_gray(transformed)

    edges_o = cv2.Canny(gray_o, 100, 200)
    edges_t = cv2.Canny(gray_t, 100, 200)

    intersection = np.logical_and(edges_o > 0, edges_t > 0).sum()
    total_edges = np.count_nonzero(edges_o)

    if total_edges == 0:
        return 0.0

    return intersection / total_edges


def to_gray(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

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

img2_avg  = blur_average(img2, ksize=5)
img2_gaus = blur_gaussian(img2, ksize=5)

# Save images (original resolution preserved)
cv2.imwrite(f"{output_dir}/img1_original.png", img1)
cv2.imwrite(f"{output_dir}/img1_avg.png", img1_avg)
cv2.imwrite(f"{output_dir}/img1_gaussian.png", img1_gaus)

pnsr1_avg = compute_psnr(img1, img1_avg)
ssim1_avg = compute_ssim(img1, img1_avg)
entropy1_avg = compute_entropy(img1_avg)
epi1_avg = compute_epi(img1, img1_avg)

pnsr1_gaus = compute_psnr(img1, img1_gaus)
ssim1_gaus = compute_ssim(img1, img1_gaus)
entropy1_gaus = compute_entropy(img1_gaus)
epi1_gaus = compute_epi(img1, img1_gaus)

cv2.imwrite(f"{output_dir}/img2_original.png", img2)
cv2.imwrite(f"{output_dir}/img2_avg.png", img2_avg)
cv2.imwrite(f"{output_dir}/img2_gaussian.png", img2_gaus)

pnsr2_avg = compute_psnr(img2, img2_avg)
ssim2_avg = compute_ssim(img2, img2_avg)
entropy2_avg = compute_entropy(img2_avg)
epi2_avg = compute_epi(img2, img2_avg)

pnsr2_gaus = compute_psnr(img2, img2_gaus)
ssim2_gaus = compute_ssim(img2, img2_gaus)
entropy2_gaus = compute_entropy(img2_gaus)
epi2_gaus = compute_epi(img2, img2_gaus)

print(f"PNSR1 AVG: {pnsr1_avg}, SSIM1 AVG: {ssim1_avg}, ENTROP1 AVG: {entropy1_avg}, EPI1 AVG: {epi1_avg}")
print(f"PNSR1 GAUS: {pnsr1_gaus}, SSIM1 GAUS: {ssim1_gaus}, ENTROP1 GAUS: {entropy1_gaus}, EPI1 GAUS: {epi1_gaus}")

print(f"PNSR2 AVG: {pnsr2_avg}, SSIM2 AVG: {ssim2_avg}, ENTROP2 AVG: {entropy2_avg}, EPI2 AVG: {epi2_avg}")
print(f"PNSR2 GAUS: {pnsr2_gaus}, SSIM2 GAUS: {ssim2_gaus}, ENTROP2 GAUS: {entropy2_gaus}, EPI2 GAUS: {epi2_gaus}")