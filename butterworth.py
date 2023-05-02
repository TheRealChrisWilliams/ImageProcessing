import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Load the image
image = plt.imread('macaw.jpg')

# Convert the image to grayscale
gray = np.mean(image, axis=-1)


# Define the Butterworth lowpass filter
def butterworth_lp(cutoff, order, shape):
    rows, cols = shape
    x, y = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))
    d = np.sqrt(x ** 2 + y ** 2)
    b_lp = 1 / (1 + (d / cutoff) ** (2 * order))
    return b_lp


# Apply the filter to the image
cutoff = 0.4
order = 2
lp_filter = butterworth_lp(cutoff, order, gray.shape)
filtered = ndimage.convolve(gray, lp_filter, mode='mirror')

# Plot the original and filtered images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(gray, cmap='gray')
ax1.set_title('Original Image')
ax2.imshow(filtered, cmap='gray')
ax2.set_title('Filtered Image')
plt.show()
