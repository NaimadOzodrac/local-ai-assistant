from pypdf import PdfReader


def load_pdf(path: str) -> str:
    """Load and extract text from a PDF file."""
    reader = PdfReader(path)

    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text