import fitz  # PyMuPDF
import json
from PIL import Image
import os

import fitz
import os
import re
from PIL import Image
from pathlib import Path
import unicodedata


def remove_hebrew_diacritics(text):
    """Remove Hebrew diacritical marks (Nikud) from text."""
    text = unicodedata.normalize("NFKD", text)  # Decompose characters
    return re.sub(r"[\u0591-\u05C7]", "", text)  # Remove diacritics


def clean_text(text):
    text = remove_hebrew_diacritics(text)  # Remove diacritics
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')  # Remove control characters
    text = re.sub(r"�+", "", text)  # Remove repeated unknown characters (�)
    text = re.sub(r"\s+", "", text)  # Replace multiple spaces/tabs/newlines with a single space
    return text.strip()  # Trim leading/trailing spaces


def process_pdf_to_images_and_data(pdf_file, dpi=300, output_dir="output", delimiters=[" ", "\n"]):
    pdf_name = Path(pdf_file).name.removesuffix('.pdf')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    doc = fitz.open(pdf_file)
    scaling_factor = dpi / 72  # Convert PDF points to desired DPI
    output_data = {}

    # Precompile the regex for delimiter splitting
    delimiters_pattern = re.compile("|".join(map(re.escape, delimiters)))

    for page_number in range(len(doc)):
        page = doc[page_number]

        # 1. Extract full text and split it with delimiters
        full_text = page.get_text("text", flags=fitz.TEXT_INHIBIT_SPACES)
        expected_words = [clean_text(word) for word in delimiters_pattern.split(full_text) if word.strip()]

        # 2. Extract words with bounding boxes
        extracted_words = page.get_text("words")
        bboxes = []
        texts = []
        current_word = ""
        current_bbox = None

        # Pointer for expected words comparison
        expected_index = 0

        for bbox_data in extracted_words:
            x0, y0, x1, y1, word = bbox_data[:5]

            # Map BBox to image coordinates
            x0_img = int(x0 * scaling_factor)
            y0_img = int(y0 * scaling_factor)
            x1_img = int(x1 * scaling_factor)
            y1_img = int(y1 * scaling_factor)
            word_bbox_img = (x0_img, y0_img, x1_img, y1_img)

            if expected_index >= len(expected_words):
                raise ValueError(f"Too many extracted words on page {page_number + 1}.")

            target_word = expected_words[expected_index]

            word = clean_text(word)
            target_word = clean_text(target_word)

            # Build current composite word and expand bounding box
            if not current_word:
                current_word = word
                current_bbox = word_bbox_img
            else:
                current_word += word
                current_bbox = (
                    min(current_bbox[0], x0_img),
                    min(current_bbox[1], y0_img),
                    max(current_bbox[2], x1_img),
                    max(current_bbox[3], y1_img)
                )
                assert len(current_word) <= len(target_word), f"unable to resolve word: {target_word}"

            # Check if the built word matches the target
            if current_word == target_word:
                # Save the merged word and bbox
                texts.append(current_word)
                bboxes.append(current_bbox)
                current_word = ""
                current_bbox = None
                expected_index += 1

        # Ensure all expected words were matched
        if current_word:
            raise ValueError(
                f"Unresolved word on page {page_number + 1}: '{current_word}'"
            )
        if expected_index < len(expected_words):
            raise ValueError(
                f"Missing expected words on page {page_number + 1}: "
                f"Remaining: {expected_words[expected_index:]}"
            )
        merged_texts = []
        merged_bbox = []
        prev_word = None
        prev_bbox = None
        merge_bboxes = lambda x, y: (min(x[0], y[0]), min(x[1], y[1]), max(x[2], y[2]), max(x[3], y[3]))
        for curr_word, curr_bbox in zip(texts, bboxes):
            if (prev_word is None) and (prev_bbox is None):
                prev_word = curr_word
                prev_bbox = curr_bbox
            elif prev_bbox[0] - curr_bbox[2] == 0 and max(prev_bbox[1], curr_bbox[1]) < min(prev_bbox[3], curr_bbox[
                3]):  # words in the same line with 0 difference between bboxes -> need to be merged
                prev_word += curr_word
                prev_bbox = merge_bboxes(prev_bbox, curr_bbox)
            else: #new_bbox - push to lists
                merged_texts.append(prev_word)
                merged_bbox.append(prev_bbox)
                prev_bbox = curr_bbox
                prev_word = curr_word

        merged_texts.append(prev_word)
        merged_bbox.append(prev_bbox)

        # Save page image and data
        img_path = f"{os.path.join(output_dir, pdf_name)}_page_{page_number + 1}.png"
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img.save(img_path)

        # Save the page data
        output_data[f"page_{page_number + 1}"] = {
            "image_path": img_path,
            "bboxes": bboxes,
            "words": texts,
        }

    # 3. Save the JSON file
    file_name = Path(pdf_file).name.removesuffix('.pdf')
    json_path = f"{output_dir}/{file_name}_output_data.json"
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(output_data, json_file, indent=4, ensure_ascii=False)

    print(f"Processing completed. Images saved to '{output_dir}', and data saved to '{json_path}'.")
    return


if __name__ == "__main__":

    pdf_path = r"C:\Users\sgala\Downloads\pkudot.pdf"
    output_dir = r"C:\Users\sgala\OCR\tesseract_test_data_example"

    # Example Usage
    out = process_pdf_to_images_and_data(pdf_path, dpi=300, output_dir=output_dir)
    print(out)
