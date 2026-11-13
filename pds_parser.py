from docx import Document
from openpyxl import load_workbook
import pytesseract
from pdf2image import convert_from_path

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
POPPLER_PATH = r'C:\poppler\Library\bin'

def extract_pdf(file_path):
    """Extract text from a PDF using OCR."""
    try:
        images = convert_from_path(file_path, dpi=300, poppler_path=POPPLER_PATH)
    except Exception as e:
        raise Exception(f"Could not read PDF: {e}")

    full_text = ""
    for i, img in enumerate(images):
        page_text = pytesseract.image_to_string(img, lang='eng')
        full_text += f"\n--- PAGE {i+1} ---\n{page_text}"

    return full_text

def extract_docx(file_path):
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])


def extract_xlsx(file_path):
    wb = load_workbook(file_path, data_only=True)

    text = ""

    for sheet in wb.sheetnames:
        ws = wb[sheet]

        for row in ws.iter_rows(values_only=True):
            row_text = " ".join([str(c) for c in row if c])
            text += row_text + "\n"

    return text