import os
import random
import cv2
import numpy as np
from pdf2image import convert_from_path
import fitz  # PyMuPDF
from tesseract.augmentations import augment_text_image, AugmentationConfig


# PDF to dataset
def pdf_to_train_images(pdf_path, output_dir,aug_config:AugmentationConfig , dpi=300):
    """Convert a PDF to a dataset of images and texts."""
    pdf_name = pdf_path.split('\\')[-1].removesuffix('.pdf')
    scale = dpi / 72  # Scaling factor
    os.makedirs(output_dir, exist_ok=True)
    images = convert_from_path(pdf_path, dpi=dpi, poppler_path=r'C:\Program Files\Popler\poppler-24.08.0\Library\bin')

    for page_num, page_image in enumerate(images):
        # Convert PIL image to OpenCV format
        page_image_cv = np.array(page_image)
        page_image_cv = cv2.cvtColor(page_image_cv, cv2.COLOR_RGB2BGR)

        # Load the PDF page with PyMuPDF for text extraction
        pdf_document = fitz.open(pdf_path)
        page = pdf_document[page_num]

        # Extract lines of text and their bounding boxes
        n = 1
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                # Combine spans to form the full line text
                line_text = " ".join(span["text"] for span in line["spans"])
                if line_text.isspace() or line_text == '':
                    continue
                bbox = line["bbox"]  # (x0, y0, x1, y1) in PDF coordinate space

                # Convert bbox to image pixel space
                x0, y0, x1, y1 = [int(coord * scale) for coord in bbox]

                # Crop the image to the line bounding box
                line_image = page_image_cv[y0:y1, x0:x1]

                # Save the line image and text
                line_id = f"{pdf_name}_page{page_num + 1}_line{n}"
                line_image_path = os.path.join(output_dir, f"{line_id}.png")
                line_text_path = os.path.join(output_dir, f"{line_id}.gt.txt")
                n += 1

                if line_image.size > 0:  # Avoid saving empty crops
                    with open(line_text_path, "w", encoding="utf-8") as text_file:
                        text_file.write(line_text)

                    # Optionally augment the image

                    line_image = augment_text_image(line_image, aug_config)
                    cv2.imwrite(line_image_path, line_image)
                    print(f'Created image {line_image_path}')

        pdf_document.close()


if __name__ == "__main__":
    # Usage
    pdf_path = r"C:\Users\sgala\OCR\hebrew_dataset\hebrew_text_0.pdf"  # Path to the input PDF
    output_dir = r"C:\Users\sgala\OCR\tesseract_train_data_example"  # Directory to save the dataset
    pdf_to_train_images(pdf_path, output_dir, augment=True, dpi=300)
