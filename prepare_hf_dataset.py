from datasets import load_dataset
import json
from tqdm import tqdm
import os

# --- Setup ---

DATASET_NAME_ON_HF = "somosnlp-hackathon-2022/spanish-to-quechua"

OUTPUT_DIR = "gemma_data/"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "quechua_hf_alpaca.jsonl")

NUM_EXAMPLES = 100


def main():
    try:
        dataset = load_dataset(DATASET_NAME_ON_HF, split="train")
    except Exception as e:
        print(f"🛑 Error al cargar el dataset: {e}")
        print("   - Revisa tu conexión a internet o el nombre del dataset.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"🔨 Convirtiendo al formato Alpaca y guardando en '{OUTPUT_FILE}'...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for row in tqdm(dataset, desc="Convirtiendo dataset"):
            spanish_text = row["es"]
            quechua_text = row["qu"]
            if not spanish_text or not quechua_text:
                continue
            entry = {
                "instruction": "Kay rimayta 'quechua simiman' tikray.",
                "input": spanish_text,
                "output": quechua_text,
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"✅ Dataset convertido y guardado en '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()
