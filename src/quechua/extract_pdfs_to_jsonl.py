import os
import jsonlines
from tqdm import tqdm
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser


INPUT_FOLDER = "E:/kaggle/gemma_data/test2"
OUTPUT_FILE = "E:/kaggle/gemma-quechua/data/raw_jsonl/quechua_dataset_marker2.jsonl"


def extract_text_with_marker(pdf_path):

    config = {
        "format_lines": True,
        "force_ocr": False,
        "use_llm": False,
        "langs": ["qu", "es"],  # Quechua y Español
        "output_format": "markdown",
    }
    config_parser = ConfigParser(config)

    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )

    # Procesar el PDF
    rendered = converter(pdf_path)
    text, _, _ = text_from_rendered(rendered)
    print(f"📄 {os.path.basename(pdf_path)}: {len(text)} caracteres extraídos")
    return text.strip()


def main():
    pdf_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".pdf")]

    with jsonlines.open(OUTPUT_FILE, mode="w") as writer:
        for filename in tqdm(pdf_files, desc="Extrayendo PDFs con Marker"):
            pdf_path = os.path.join(INPUT_FOLDER, filename)
            try:
                text = extract_text_with_marker(pdf_path)
                if text:
                    writer.write(
                        {
                            "id": filename.replace(".pdf", ""),
                            "text": text,
                            "metadata": {"source": filename, "processor": "marker-pdf"},
                        }
                    )
            except Exception as e:
                print(f"Error al procesar {filename}: {e}")

    print(
        f"\n✅ Extracción completada con Marker. Archivo guardado como: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
