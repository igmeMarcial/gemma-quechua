import json
from semhash import SemHash

INPUT_FILE = "quechua_dataset.jsonl"
OUTPUT_FILE = "quechua_dataset_clean.jsonl"


def load_texts_from_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f if line.strip()]


def save_texts_to_jsonl(texts, path):
    with open(path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


def main():
    print("📥 Cargando textos...")
    texts = load_texts_from_jsonl(INPUT_FILE)

    print(f"🧠 Textos cargados: {len(texts)}. Iniciando deduplicación semántica...")
    semhash = SemHash.from_records(records=texts)

    deduplicated = semhash.self_deduplicate()
    dedup_texts = deduplicated.selected
    print(f"✅ Deduplicados: {len(dedup_texts)} textos restantes.")

    print("🧹 Filtrando outliers...")
    filtered = semhash.self_filter_outliers()
    clean_texts = filtered.selected
    print(f"✅ Dataset limpio: {len(clean_texts)} textos finales.")

    print(f"💾 Guardando en: {OUTPUT_FILE}")
    save_texts_to_jsonl(clean_texts, OUTPUT_FILE)

    print("🎉 Proceso completo.")


if __name__ == "__main__":
    main()
