import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load the image
image = np.array(Image.open('vedant.bmp'))

# Normalize the image to [0, 1] range
image_norm = image / 255.0

# Split the image into its RGB channels
r = image_norm[:, :, 0]
g = image_norm[:, :, 1]
b = image_norm[:, :, 2]

# Perform histogram equalization on each RGB channel
r_hist, _ = np.histogram(r.flatten(), bins=256, range=(0, 1))
g_hist, _ = np.histogram(g.flatten(), bins=256, range=(0, 1))
b_hist, _ = np.histogram(b.flatten(), bins=256, range=(0, 1))

r_cdf = np.cumsum(r_hist) / (r.shape[0] * r.shape[1])
g_cdf = np.cumsum(g_hist) / (g.shape[0] * g.shape[1])
b_cdf = np.cumsum(b_hist) / (b.shape[0] * b.shape[1])

r_eq = np.interp(r.flatten(), np.linspace(0, 1, 256), r_cdf)
g_eq = np.interp(g.flatten(), np.linspace(0, 1, 256), g_cdf)
b_eq = np.interp(b.flatten(), np.linspace(0, 1, 256), b_cdf)

r_eq = (255 * r_eq).astype(np.uint8).reshape(r.shape)
g_eq = (255 * g_eq).astype(np.uint8).reshape(g.shape)
b_eq = (255 * b_eq).astype(np.uint8).reshape(b.shape)

# Combine the equalized RGB channels into a single RGB image_normalized
image_eq = np.stack((r_eq, g_eq, b_eq), axis=2)

r = image_eq[:, :, 0]
g = image_eq[:, :, 1]
b = image_eq[:, :, 2]

# Compute the histograms of each RGB channel
hist_r, bins_r = np.histogram(r.flatten(), bins=256, range=(0, 255))
hist_g, bins_g = np.histogram(g.flatten(), bins=256, range=(0, 255))
hist_b, bins_b = np.histogram(b.flatten(), bins=256, range=(0, 255))

# Overlay the histograms of each RGB channel with their respective colors
plt.figure()
plt.title('Overlayed Histograms of RGB Channels')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.bar(bins_r[:-1], hist_r, color='red', alpha=0.5, width=1)
plt.bar(bins_g[:-1], hist_g, color='green', alpha=0.5, width=1)
plt.bar(bins_b[:-1], hist_b, color='blue', alpha=0.5, width=1)
plt.show()

# Display the original and equalized images side by side
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image)
ax[0].set_title('Original Image')
ax[1].imshow(image_eq)
ax[1].set_title('Equalized Image')
plt.show()
