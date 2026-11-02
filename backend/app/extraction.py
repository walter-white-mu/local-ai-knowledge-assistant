import io
import re

from pypdf import PdfReader

from app.chunking import Section
from app.errors import AppError


def extract_sections(filename: str, content: bytes, max_chars: int) -> list[Section]:
    if filename.lower().endswith(".txt"):
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise AppError("TXT files must use UTF-8 encoding.") from exc
        if "\x00" in text:
            raise AppError("This does not appear to be a plain text file.")
        sections = [
            Section(text=part.strip(), section=index + 1)
            for index, part in enumerate(re.split(r"\n\s*\n", text))
            if part.strip()
        ]
    elif filename.lower().endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(content))
            if reader.is_encrypted:
                raise AppError("Encrypted PDFs are not supported. Upload an unlocked copy.")
            sections = []
            total = 0
            for index, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                total += len(text)
                if total > max_chars:
                    raise AppError("Extracted document is too large.", 413)
                if text.strip():
                    sections.append(Section(text=text, section=index + 1, page=index + 1))
        except AppError:
            raise
        except Exception as exc:
            raise AppError("Could not read this PDF. It may be malformed.") from exc
    else:
        raise AppError("Only PDF and TXT documents are supported.", 415)
    if not sections:
        raise AppError("No readable text found. Scanned PDFs require OCR, which is not supported.")
    if sum(len(section.text) for section in sections) > max_chars:
        raise AppError("Extracted document is too large.", 413)
    return sections
