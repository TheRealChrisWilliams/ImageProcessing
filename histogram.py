import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load the image and convert it to a grayscale NumPy array
image = np.array(Image.open('pelvis.bmp').convert('L'))
image2 = image + 40

# Compute the histogram of pixel intensities
hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 255))

hist2, bins2 = np.histogram(image2.flatten(), bins=256, range=(0, 255))

# Plot the histogram
plt.figure()
plt.title('Histogram of Pixel Intensities')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.bar(bins[:-1], hist, width=1)
plt.show()


plt.figure()
plt.title('Histogram of Pixel Intensities + 40')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.bar(bins2[:-1], hist2, width=1)
plt.show()


plt.title('Overlayed Histograms')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.hist(image.flatten(), bins=256, range=(0, 255), alpha=0.5)
plt.hist(image2.flatten(), bins=256, range=(0, 255), alpha=0.5)
plt.show()
