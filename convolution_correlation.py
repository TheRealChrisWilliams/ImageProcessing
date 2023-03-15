import os

import numpy as np
from PIL import Image
from scipy.signal import convolve2d

# Load the image and convert it to a grayscale NumPy array
# noinspection PyTypeChecker
image = np.array(Image.open('pelvis.bmp').convert('L'))

# Define a 3x3 filter
kernel = np.array([[4, 0, -5],
                   [2, 0, -2],
                   [3, 0, -4]])[::-1, ::-1]

# Compute the spatial convolution
conv_result = convolve2d(image, kernel, mode='same') + image

# Compute the spatial correlation
corr_result = convolve2d(image, kernel, mode='same', boundary='symm')

# Save the convolution result to a new bitmap image
conv_image = Image.fromarray(conv_result.astype('uint8'))
conv_image.save(os.path.join(os.path.expanduser("~"), "Desktop", "conv_result.bmp"))

# Save the correlation result to a new bitmap image
corr_image = Image.fromarray(corr_result.astype('uint8'))
corr_image.save(os.path.join(os.path.expanduser("~"), "Desktop", "corr_result.bmp"))
