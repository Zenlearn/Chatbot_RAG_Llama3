from config.config_env import CHUNK_SIZE
from fastapi import UploadFile
from PyPDF2 import PdfReader
import pandas as pd
import docx2txt
from io import BytesIO


def extract_text(file: UploadFile) -> str:
    """
    Extract text from an uploaded file (supports PDF and plain text).
    """
    if file.filename.endswith(".pdf"):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += " ".join(page.extract_text().replace("\n", ""))
        return text

    elif file.filename.endswith(".txt"):
        return " ".join(file.file.read().decode("utf-8").replace("\n", ""))

    elif file.filename.endswith(".xlsx"):
        data = pd.read_excel(file.file, sheet_name=None)
        text = ""
        for sheet_name, df in data.items():
            text += " ".join(
                df.fillna("").to_string(index=False, header=True).replace("\n", "")
            )
        return text
    elif file.filename.endswith(".docx"):
        file_content = file.file.read()
        content = " ".join(
            docx2txt.process(BytesIO(file_content)).replace("\n", "").split()
        )
        return content
    else:
        raise Exception("Unsupported file type. Please upload a PDF or text file.")


def chunk_text(text: str) -> list[str]:
    """
    Split text into smaller chunks for vectorization.
    """
    words = text.split()
    chunks = [
        " ".join(words[i : i + CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)
    ]
    return chunks
