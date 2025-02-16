import cv2
import numpy as np
import matplotlib.pyplot as plt


def resize_image_constant_median_size(image, target_median_size=100, max_resize=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    th, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = (gray < th).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    cc_sizes
    median_size = np.median(stats[:, 4])

    resize_ratio = min(max(target_median_size / median_size, 1), max_resize)
    resize_shape = [int(np.rint(i * resize_ratio)) for i in image.shape[:2]][::-1]
    assert resize_shape[0]<100_000 and resize_shape[1]<100_000
    resized_image = cv2.resize(image, resize_shape)
    return resized_image


def enhance_soft_binarization(image: str, blur_resolution=50, remove_bg_strength=10, verbose=False) -> np.ndarray:
    """
    Enhances an image for better OCR performance.

    Steps:
    - Convert to grayscale
    - Apply noise reduction
    - Perform soft binarization

    :param image_path: Path to the input image
    :return: Enhanced image as a NumPy array
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if verbose:
        plt.figure(figsize=(24, 16))
        plt.imshow(gray)
        plt.show()
    # Estimate kernel size for Gaussian blur based on image size
    kernel_size = max(3, (min(gray.shape[:2]) // blur_resolution) | 1)  # Ensure odd kernel size

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    if verbose:
        plt.figure(figsize=(24, 16))
        plt.imshow(blurred)
        plt.show()
    #     Apply adaptive thresholding for binarization
    th, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    gray_n = gray.astype(float)
    min_max_norm = lambda x: (x - x.min()) / (x.max() - x.min())
    th_n = (th - gray_n.min()) / (gray_n.max() - gray_n.min())
    gray_n = min_max_norm(gray)
    enhanced = 1 - np.log(1 + np.exp(remove_bg_strength * (-gray_n + th_n)))
    enhanced = (min_max_norm(enhanced) * 255).astype(np.uint8)

    if verbose:
        print(f"threshold :{th}")
        plt.figure(figsize=(24, 16))
        plt.imshow(enhanced)
        plt.show()
    return enhanced


def align_text(image: np.ndarray, angle_resolution=0.1, max_angle=45, line_threshold=0.2, canny_thresholds=(50, 150),
               verbose=False) -> np.ndarray:
    """
    Corrects the skew of text in an image by detecting and rotating it to the correct orientation.

    Uses the Hough Line Transform to estimate the best rotation angle.

    :param image: Input binary image (output of enhance_image)
    :return: Deskewed image as a NumPy array
    """
    # Invert colors for better line detection (white text on black background)
    inverted = cv2.bitwise_not(image)

    # Detect edges for Hough Line Transform
    edges = cv2.Canny(inverted, canny_thresholds[0], canny_thresholds[
        1])  # TODO: this is suppose to be very easy to estimate if there is an enhancement step - replace with automatic
    if verbose:
        plt.figure(figsize=(24, 16))
        plt.imshow(edges)
        plt.show()
    # Perform Hough Line Transform
    lines = cv2.HoughLines(edges, 1, angle_resolution * np.pi / 180, int(line_threshold * image.shape[1]))
    if lines is None:
        print("No lines detected for alignment.")
        return image

    # Calculate the average angle of detected lines
    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta - np.pi / 2) * (180 / np.pi)  # Convert from radians to degrees
        #         angle = (angle + 45)%90 - 45
        if max_angle > angle > -max_angle:
            angles.append(angle)
    median_angle = np.median(angles)
    if verbose:
        plt.hist(angles, bins=np.arange(-max_angle, max_angle, angle_resolution * np.pi / 180))
        # Compute the median angle to avoid outliers
        print(median_angle)

    # Get image dimensions
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Perform the rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    deskewed = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return deskewed


class preprocessConfig():
    def __init__(self):
        self.resize = True
        self.resize_params = {"target_median_size": 100,
                              "max_resize":10}
        self.enhance = True
        self.enhance_params = {"blur_resolution": 50,
                               "remove_bg_strength": 10,
                               }
        self.align = True
        self.align_params = {"angle_resolution": 0.1,
                             "max_angle": 45,
                             "line_threshold": 0.2,
                             "canny_thresholds": (50, 150)}


def preprocess(image, config: preprocessConfig):
    img_out = image.copy()
    if config.resize:
        img_out = resize_image_constant_median_size(image, **config.resize_params)
    if config.enhance:
        img_out = enhance_soft_binarization(img_out, **config.enhance_params)
    if config.align:
        img_out = align_text(img_out, **config.align_params)
    return img_out
