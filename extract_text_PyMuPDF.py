import pymupdf
import os
import fitz


PDF_SOURCE_DIR = "E:/kaggle/gemma_data/filesNew"
TEXT_OUTPUT_DIR = "E:/kaggle/gemma-quechua/data/raw_text"

os.makedirs(TEXT_OUTPUT_DIR, exist_ok=True)


def extract_text_from_pdfs():
    """
    Recorre todos los PDFs en PDF_SOURCE_DIR, extrae su texto
    y lo guarda en archivos .txt en TEXT_OUTPUT_DIR.
    """
    pdf_files = [f for f in os.listdir(PDF_SOURCE_DIR) if f.endswith(".pdf")]
    print(f"Encontrados {len(pdf_files)} PDFs para procesar.")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_SOURCE_DIR, pdf_file)
        text_filename = os.path.splitext(pdf_file)[0] + ".txt"
        text_path = os.path.join(TEXT_OUTPUT_DIR, text_filename)

        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            with open(text_path, "w", encoding="utf-8-sig") as f:
                f.write(full_text)
            print(f"Texto extraído de {pdf_path} y guardado en {text_path}")
        except Exception as e:
            print(f"Error procesando {pdf_path}: {e}")


if __name__ == "__main__":
    extract_text_from_pdfs()
