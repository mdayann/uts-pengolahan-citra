#####Soal nomor 2
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization(img):
    img = img.astype('uint8')
    
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    cdf = hist.cumsum()
    
    cdf_normalized = cdf * float(hist.max()) / cdf.max()

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    img_equalized = cdf[img]
    
    return img_equalized, hist, cdf_normalized

url = "https://www.shutterstock.com/image-photo/low-contrast-horizontal-key-image-260nw-1195770070.jpg"
img = imageio.imread(url)

if len(img.shape) == 3:
    img_gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
else:
    img_gray = img.astype(np.uint8)

img_equalized, hist_original, cdf = histogram_equalization(img_gray)

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.title('Gambar Asli')
plt.imshow(img_gray, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Histogram Gambar Asli')
plt.hist(img_gray.flatten(), 256, [0, 256])
plt.xlabel('Intensitas Piksel')
plt.ylabel('Jumlah Piksel')

plt.subplot(2, 2, 3)
plt.title('Gambar Setelah Histogram Equalization')
plt.imshow(img_equalized, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title('Histogram Setelah Equalization')
plt.hist(img_equalized.flatten(), 256, [0, 256])
plt.xlabel('Intensitas Piksel')
plt.ylabel('Jumlah Piksel')

plt.tight_layout()
plt.show()

#####Soal nomor 3
def contrast_scaling(img, level=1.5):

    img_norm = img.astype(float) / 255
    img_contrast = np.clip(img_norm * level, 0, 1)
    return (img_contrast * 255).astype(np.uint8)

img_contrast = contrast_scaling(img_gray, 1.5)

plt.figure(figsize=(15, 12))

plt.subplot(3, 2, 1)
plt.title('Gambar Asli')
plt.imshow(img_gray, cmap='gray')
plt.axis('off')

plt.subplot(3, 2, 2)
plt.title('Histogram Gambar Asli')
plt.hist(img_gray.flatten(), 256, [0, 256], color='black')
plt.xlabel('Intensitas Piksel')
plt.ylabel('Jumlah Piksel')

plt.subplot(3, 2, 3)
plt.title('Contrast Scaling (level 1.5)')
plt.imshow(img_contrast, cmap='gray')
plt.axis('off')

plt.subplot(3, 2, 4)
plt.title('Histogram Contrast Scaling')
plt.hist(img_contrast.flatten(), 256, [0, 256], color='blue')
plt.xlabel('Intensitas Piksel')
plt.ylabel('Jumlah Piksel')

plt.subplot(3, 2, 5)
plt.title('Histogram Equalization')
plt.imshow(img_equalized, cmap='gray')
plt.axis('off')

plt.subplot(3, 2, 6)
plt.title('Histogram Setelah Equalization')
plt.hist(img_equalized.flatten(), 256, [0, 256], color='red')
plt.xlabel('Intensitas Piksel')
plt.ylabel('Jumlah Piksel')

plt.tight_layout()
plt.show()

print("\nStatistik Perbandingan:")
print(f"Gambar Asli - Mean: {img_gray.mean():.2f}, Std: {img_gray.std():.2f}")
print(f"Contrast Scaling - Mean: {img_contrast.mean():.2f}, Std: {img_contrast.std():.2f}")
print(f"Histogram Equalization - Mean: {img_equalized.mean():.2f}, Std: {img_equalized.std():.2f}")

def calculate_psnr(original, modified):
    mse = np.mean((original - modified) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

print("\nPSNR (Peak Signal-to-Noise Ratio):")
print(f"Contrast Scaling vs Original: {calculate_psnr(img_gray, img_contrast):.2f} dB")
print(f"Histogram Equalization vs Original: {calculate_psnr(img_gray, img_equalized):.2f} dB")
