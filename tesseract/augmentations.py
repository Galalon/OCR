import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance


class AugmentationConfig:
    def __init__(self):
        self.apply_low_pass_noise = True
        self.low_pass_noise_kernel_size_range = (3, 7)
        self.low_pass_noise_std_range = (10, 30)

        self.apply_blur = True
        self.blur_kernel_size_range = (3, 7)

        self.apply_color_jitter = True
        self.color_jitter_brightness_range = (0.8, 1.2)  # Multiplicative factor
        self.color_jitter_brightness_bias_range = (-30, 30)  # Additive bias
        self.color_jitter_contrast_range = (0.8, 1.2)
        self.color_jitter_saturation_range = (0.8, 1.2)

        self.apply_rotation = True
        self.rotation_angle_range = 5

        self.augmentation_probabilities = {
            "low_pass_noise": 0.8,
            "blur": 0.8,
            "color_jitter": 0.8,
            "rotation": 0.8,
        }


def add_low_pass_noise(image, config):
    """Apply low-pass filtered noise to simulate spatially correlated noise."""
    if random.random() > config.augmentation_probabilities["low_pass_noise"]:
        return image
    kernel_size = random.choice(
        range(config.low_pass_noise_kernel_size_range[0], config.low_pass_noise_kernel_size_range[1] + 1, 2))
    noise_std = random.uniform(config.low_pass_noise_std_range[0], config.low_pass_noise_std_range[1])
    noise = np.random.normal(0, noise_std * kernel_size, image.shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (kernel_size, kernel_size), 0)
    noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_image


def apply_blur(image, config):
    """Apply Gaussian blur."""
    if random.random() > config.augmentation_probabilities["blur"]:
        return image
    kernel_size = random.choice(range(config.blur_kernel_size_range[0], config.blur_kernel_size_range[1] + 1, 2))
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_color_jitter(image, config):
    """Apply color jitter with brightness bias, contrast, and saturation adjustments."""
    if random.random() > config.augmentation_probabilities["color_jitter"]:
        return image

    brightness_factor = random.uniform(config.color_jitter_brightness_range[0], config.color_jitter_brightness_range[1])
    brightness_bias = random.uniform(config.color_jitter_brightness_bias_range[0],
                                     config.color_jitter_brightness_bias_range[1])
    contrast_factor = random.uniform(config.color_jitter_contrast_range[0], config.color_jitter_contrast_range[1])
    saturation_factor = random.uniform(config.color_jitter_saturation_range[0], config.color_jitter_saturation_range[1])

    image = image.astype(np.float32) * brightness_factor + brightness_bias
    image = np.clip(image, 0, 255).astype(np.uint8)

    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Apply contrast adjustment
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    # Apply saturation adjustment
    img = ImageEnhance.Color(img).enhance(saturation_factor)

    return np.array(img)


def rotate_image(image, config):
    """Apply small rotation within a limited range, adjusting the bounding box to prevent cropping."""
    if random.random() > config.augmentation_probabilities["rotation"]:
        return image
    h, w = image.shape[:2]
    angle = random.uniform(-config.rotation_angle_range, config.rotation_angle_range)

    # Compute the new bounding box size
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Adjust the rotation matrix to account for translation
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    # Perform the rotation
    return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=[255,255,255])


def augment_text_image(image, config):
    """Apply all augmentations to a given text image."""
    if config.apply_rotation:
        image = rotate_image(image, config)
    if config.apply_blur:
        image = apply_blur(image, config)
    if config.apply_color_jitter:
        image = apply_color_jitter(image, config)
    if config.apply_low_pass_noise:
        image = add_low_pass_noise(image, config)
    return image

