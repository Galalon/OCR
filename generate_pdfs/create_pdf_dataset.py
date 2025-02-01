import os
import random
from generate_pdfs.generate_pdf import generate_pdf, pdf_to_image, load_hebrew_words, register_fonts


def create_pdf_dataset(corpus_path='top_500_words.csv', pdfs_path='dataset', images_path=None, n_files=100):
    font_files = register_fonts()
    os.makedirs(pdfs_path, exist_ok=True)
    if images_path is not None:
        os.makedirs(images_path, exist_ok=True)
    hebrew_words = load_hebrew_words(corpus_path)

    # Generate multiple PDFs with different fonts
    for i in range(n_files):
        font_name = random.choice(list(font_files.keys()))
        pdf_path = os.path.join(pdfs_path, f"text_{i}.pdf")
        generate_pdf(pdf_path, font_name, hebrew_words)
        if images_path is not None:
            image_path = os.path.join(images_path, f"image_{i}")
            pdf_to_image(pdf_path, image_path)
        print(f"Generating PDF with font {font_name}...")
    print(f"Dataset created in directory: {pdfs_path}")


if __name__ == '__main__':
    create_pdf_dataset(images_path='images')
