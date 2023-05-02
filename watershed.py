import numpy as np
import skimage.io
import skimage.segmentation
import skimage.color
import skimage.transform

# Load the image
image = skimage.io.imread('holes.jpg')

# Convert the image to grayscale
image = skimage.color.rgb2gray(image)
print(image)
# Threshold the image to obtain markers
threshold = 10
markers = skimage.filters.threshold_local(image, block_size=11, method='gaussian') > threshold

# Resize the markers array to the same shape as the image
markers = skimage.transform.resize(markers, image.shape, anti_aliasing=False)

# Apply watershed segmentation
labels = skimage.segmentation.watershed(-image, markers)

# Show the result
skimage.io.imshow(labels)
skimage.io.show()
