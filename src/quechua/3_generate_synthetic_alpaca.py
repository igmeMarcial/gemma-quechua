

import json
import random
import re


SOURCE_CORPUS_FILE = 'E:/kaggle/gemma-quechua/data/raw_text/biblia.txt'
OUTPUT_FILE = 'E:/kaggle/gemma-quechua/data/quechua_alpaca_dataset_v1.jsonl'

def create_summary_examples(paragraphs, max_examples=500):
    
    examples = []
    valid_paragraphs = [p for p in paragraphs if '.' in p and len(p) > 150]
    random.shuffle(valid_paragraphs)
    
    for p in valid_paragraphs[:max_examples]:
        # Heurística simple: la primera oración es el "resumen".
        first_sentence = p.split('.')[0] + '.'
        if len(first_sentence) < len(p) * 0.8: # Asegurarse de que el resumen es más corto
            entry = {
                "instruction": "Kay qillqasqata pisiyachiy.",
                "input": p,
                "output": first_sentence
            }
            examples.append(entry)
    return examples

   # --- FÁBRICA DE DATOS 2: EXTRACCIÓN DE INFORMACIÓN ---
def create_qa_from_patterns(paragraphs, max_examples=500):
    """
    Genera preguntas y respuestas buscando patrones.
    Esto es más complejo y requiere conocimiento del idioma.
    Ejemplo: Busca oraciones con nombres y lugares.
    """
    examples = []
    # Ejemplo muy simple: busca oraciones con "Inka"
    valid_paragraphs = [p for p in paragraphs if 'inka' in p.lower()]
    random.shuffle(valid_paragraphs)

    for p in valid_paragraphs[:max_examples]:
        # Busca la primera oración que contenga "inka"
        sentences = re.split(r'(?<=[.!?])\s+', p)
        target_sentence = next((s for s in sentences if 'inka' in s.lower()), None)

        if target_sentence:
            entry = {
                "instruction": "Pimantataq kay qillqasqa rimarin?", # ¿De quién habla este texto?
                "input": p,

                # La respuesta es la oración específica que encontraste
                "output": f"Kay qillqasqaqa Inkakunamantam rimarin. Nintaqmi: '{target_sentence}'"
            }
            examples.append(entry)
    return examples

# --- FÁBRICA DE DATOS 3: PREGUNTAS DE CONOCIMIENTO GENERAL ---
# Esto es más fácil de crear a mano, pero puedes tener una lista y expandirla.
def create_general_knowledge_examples():
    qa_pairs = [
        {"instruction": "Imataq 'wasi' ninayan?", "input": "", "output": "'Wasi'qa runap tiyananpaq wasim, pirqayuq, wasi qatayuq ima."},
        {"instruction": "Hayk'aqmi Peru suyup p'unchawnin?", "input": "", "output": "Peru suyup p'unchawninqa sapa 28 punchaw inti raymi killapim."},
        # ... añade más pares aquí ...
    ]
    return qa_pairs



# --- FUNCIÓN PRINCIPAL ---
def main():
    with open(SOURCE_CORPUS_FILE, 'r', encoding='utf-8') as f:
        # Asumimos que cada párrafo está separado por dos saltos de línea
        paragraphs = [p.strip() for p in f.read().split('\n\n') if p.strip()]

    print(f"Corpus cargado con {len(paragraphs)} párrafos.")

    # Generar ejemplos de cada tipo
    summary_data = create_summary_examples(paragraphs, max_examples=1000)
    qa_pattern_data = create_qa_from_patterns(paragraphs, max_examples=500)
    general_knowledge_data = create_general_knowledge_examples()
    
    # Combinar todos los datasets
    full_dataset = summary_data + qa_pattern_data + general_knowledge_data
    random.shuffle(full_dataset)

    # Guardar en formato JSONL
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in full_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"¡Éxito! Dataset sintético creado con {len(full_dataset)} ejemplos.")
    print(f"Archivo guardado en: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
    
