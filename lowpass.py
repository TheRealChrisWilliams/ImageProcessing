import numpy as np
from PIL import Image


def apply_lowpass_filter(image, kernel_size):
    # Convert the image to a NumPy ndarray
    image_array = np.array(image)

    # Create a 2D Gaussian kernel
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    sigma = 8.0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)

    # Apply the kernel to each channel of the image
    filtered_image = np.zeros_like(image_array)
    for c in range(3):
        padded_image = np.pad(image_array[:, :, c], ((center, center), (center, center)), mode='edge')
        for i in range(image_array.shape[0]):
            for j in range(image_array.shape[1]):
                patch = padded_image[i:i + kernel_size, j:j + kernel_size]
                filtered_image[i, j, c] = np.sum(patch * kernel)

    # Convert the filtered image back to an Image object
    filtered_image = Image.fromarray(filtered_image.astype(np.uint8))
    return filtered_image


if __name__ == '__main__':
    # Read the input image
    image = Image.open('chris.jpeg')

    # Apply the lowpass filter with a kernel size of 5
    filtered_image = apply_lowpass_filter(image, 5)
    filtered_image.save('filtered_image.jpg')

    # Show the original and filtered images
    image.show()
    filtered_image.show()

