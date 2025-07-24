import fitz  # PyMuPDF
import os
import json
from tqdm import tqdm

# --- CONFIGURACIÓN ---
PDF_SOURCE_DIR = "gemma_data/pdfs_quechua/"
OUTPUT_DIR = "data/vision_dataset/"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_JSONL_FILE = os.path.join(OUTPUT_DIR, "metadata.jsonl")
IMAGE_DPI = 300


def create_image_text_dataset():
    """
    Recorre los PDFs, convierte cada página a una imagen PNG de alta calidad,
    extrae su texto, y guarda todo en un archivo JSONL.
    """
    os.makedirs(IMAGE_DIR, exist_ok=True)
    pdf_files = [f for f in os.listdir(PDF_SOURCE_DIR) if f.lower().endswith(".pdf")]
    print(f"📚 Encontrados {len(pdf_files)} PDFs para procesar.")

    with open(OUTPUT_JSONL_FILE, "w", encoding="utf-8") as writer:
        for pdf_file in tqdm(pdf_files, desc="Procesando PDFs"):
            pdf_path = os.path.join(PDF_SOURCE_DIR, pdf_file)
            try:
                doc = fitz.open(pdf_path)
                for page_num in tqdm(
                    range(len(doc)), desc=f"   -> {pdf_file}", leave=False
                ):
                    page = doc.load_page(page_num)
                    page_text = page.get_text("text", sort=True).strip()
                    if len(page_text.split()) < 10:
                        continue

                    pix = page.get_pixmap(dpi=IMAGE_DPI)
                    image_filename = (
                        f"{os.path.splitext(pdf_file)[0]}_page_{page_num + 1}.png"
                    )
                    image_path = os.path.join(IMAGE_DIR, image_filename)
                    pix.save(image_path)

                    relative_image_path = os.path.join("images", image_filename)
                    record = {
                        "image_path": relative_image_path,
                        "text": page_text,
                    }
                    writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"❗️ Error procesando '{pdf_file}': {e}")

    print(f"\n🎉 ¡Proceso completado! Dataset de visión creado en '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    create_image_text_dataset()
