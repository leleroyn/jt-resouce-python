from typing import List
from fitz_new import fitz


def convent_page_to_image(pdf_bits) -> List:
    pdf_page_img_data = []
    pdf_doc = fitz.open("pdf", pdf_bits)
    pages = pdf_doc.pages()

    for page in pages:
        zoom_x = 1.33333333
        zoom_y = 1.33333333
        mat = fitz.Matrix(zoom_x, zoom_y)
        pix = page.get_pixmap(matrix=mat, dpi=None, alpha=False)
        img_bits = pix.tobytes()
        pdf_page_img_data.append(img_bits)
    return pdf_page_img_data
