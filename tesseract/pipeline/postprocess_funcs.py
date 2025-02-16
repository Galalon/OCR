from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import matplotlib.pyplot as plt


# TODO: move to seperate file
def visualize_ocr(image, bboxes, words, wers, font_path="arial.ttf", font_size=20, right_box_color=(0, 255, 0),
                  wrong_box_color=(255, 0, 0), text_color=(10, 10, 10), thickness=2):
    """
    Visualizes OCR results with support for UTF-8 characters.

    Parameters:
        image (numpy.ndarray): The image (read via OpenCV).
        bboxes (list of list of int): List of bounding boxes, where each box is [x1, y1, x2, y2].
        words (list of str): List of words corresponding to each bounding box.
        font_path (str): Path to a TrueType font file (supports UTF-8 characters).
        font_size (int): Font size for the text.
        right_box_color (tuple): Color of the bounding box (default is green).
        right_box_color (tuple): Color of the bounding box if the prediction is wrong(default is red).
        text_color (tuple): Color of the text (default is black).
        thickness (int): Thickness of the bounding box.

    Returns:
        None: Displays the image with OCR annotations.
    """
    # Convert the image to a PIL image
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # Load the specified font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        raise IOError(f"Font not found at {font_path}. Please specify a valid TrueType font file.")

    # Draw bounding boxes and text
    for bbox, word, wer in zip(bboxes, words, wers):
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox  # Unpack the bounding box

        bbox_color = right_box_color if (wer is not None) and (wer <= 0) else wrong_box_color
        # Draw the bounding box
        draw.rectangle([x1, y1, x2, y2], outline=bbox_color, width=thickness)

        # Add the text above the bounding box
        draw.text((x1, y1 - font_size), word[::-1], fill=text_color, font=font)

    # Convert back to OpenCV format for displaying with Matplotlib
    result_image = np.array(pil_image)
    return result_image

import pandas as pd
def postprocess_ocr_data(ocr_data_pred, max_char_bbox_ratio=10,min_confidence=30):
    image_data_pred = pd.DataFrame(ocr_data_pred)
    image_data_pred['bbox'] = image_data_pred.apply(lambda r: (r['left'],r['top'],r['left'] + r['width'] ,r['top'] + r['height']),axis=1)
    image_data_pred = image_data_pred[image_data_pred['conf']!=-1]
    valid_box = image_data_pred["text"].apply(lambda x: bool(x.strip()))
    bbox_ratio_to_text_len = (image_data_pred["width"]/image_data_pred["height"])/image_data_pred["text"].apply(lambda x: len(x)+1e-8)
    valid_box = valid_box&(bbox_ratio_to_text_len<max_char_bbox_ratio)&(bbox_ratio_to_text_len>1/max_char_bbox_ratio)
    valid_box = valid_box&(image_data_pred["conf"]>=min_confidence)
    pred_bbox = list(image_data_pred[valid_box]['bbox'])
    pred_text = list(image_data_pred[valid_box]['text'])
    pred_conf = list(image_data_pred[valid_box]['conf'])
    return pred_bbox,pred_text,pred_conf



