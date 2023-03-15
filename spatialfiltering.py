import os

import numpy as np
from PIL import Image

# Load the image and convert it to a grayscale NumPy array
image = np.array(Image.open('pelvis.bmp').convert('L'))


first_order = image[:+1]-image[1:]
second_order = image[:-1]+image[:+1]-2*image[1:]

combine = first_order+second_order

first_order_image = Image.fromarray(first_order.astype('uint8'))
first_order_image.save((os.path.join(os.path.expanduser("~"), "Desktop", "first_order.bmp")))

second_order_image = Image.fromarray(second_order.astype('uint8'))
second_order_image.save((os.path.join(os.path.expanduser("~"), "Desktop", "second_order.bmp")))

combined = Image.fromarray(combine.astype('uint8'))
combined.save((os.path.join(os.path.expanduser("~"), "Desktop", "combined.bmp")))
