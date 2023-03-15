import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

# Load the bitmap image and convert it to a NumPy array
image = np.array(Image.open('bird.bmp').convert('L'))

# Apply a low-pass Gaussian filter to the image
sigma = 6.0
filtered_image = gaussian_filter(image, sigma)

# Convert the filtered image to uint8 for display
filtered_image = np.uint8(filtered_image)

# Display the original and filtered images side by side

gaussian_image = Image.fromarray(filtered_image)
gaussian_image.save((os.path.join(os.path.expanduser("~"), "Desktop", "gaussian.bmp")))

fig, axs = plt.subplots(1, 2)
axs[0].imshow(image, cmap='gray')
axs[0].set_title('Original Image')
axs[1].imshow(filtered_image, cmap='gray')
axs[1].set_title('Filtered Image')
plt.show()
