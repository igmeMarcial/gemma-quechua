
import json
import random


INSTRUCTION_TEMPLATES = [
    "¿Imamantataq kay qillqasqa rimarin?",  # ¿De qué habla este texto?
    "Kay qillqasqata pisiyachiy.",         # Resume este texto.
    "¿Ima yachayllatapas kaymanta horqoyta atinki?", # ¿Qué conocimiento puedes extraer de aquí?
    "Kay qillqasqamanta 5 tapuykunata ruway.", # Crea 5 preguntas sobre este texto.
]

# Asumimos que tienes un archivo de texto limpio con párrafos en quechua
SOURCE_TEXT_FILE = '../data/raw_text/quechua_corpus.txt' 
OUTPUT_JSONL_FILE = '../data/quechua_dataset.jsonl'
CHAT_TEMPLATE = "<s>[INST] {instruction} [/INST] {output}</s>"



def create_dataset_from_text():
    """
    Lee un corpus de texto y lo convierte en un dataset de instrucciones
    en formato JSONL, listo para Unsloth.
    
    Read a text corpus and convert it into an instruction dataset
    in JSONL format, ready for Unsloth.
    """
    with open(SOURCE_TEXT_FILE, 'r', encoding='utf-8') as f:
        paragraphs = [p.strip() for p in f.read().split('\n\n') if p.strip()]
    print(f"Procesando {len(paragraphs)} párrafos para crear el dataset de instrucciones.")

    dataset_entries = []
    
    # --- TIPO 1: Tareas de resumen y pregunta sobre un texto ---
    # Esto es solo un ejemplo, ¡aquí es donde tu creatividad y conocimiento entran en juego!
    for p in paragraphs:
        if len(p.split()) > 20: # Solo usar párrafos suficientemente largos
            # Tarea: Resumir el texto
            instruction = f"Kay qillqasqata pisillapi willay: '{p}'"
            # Para la respuesta 'output', idealmente tú crearías un resumen.
            # Como eso es mucho trabajo, un truco es usar la primera oración como "resumen".
            output = p.split('.')[0] + '.'
            formatted_entry = CHAT_TEMPLATE.format(instruction=instruction, output=output)
            dataset_entries.append({"text": formatted_entry})

    # --- TIPO 2: Traducción (si tienes pares español-quechua) ---
    # spanish_quechua_pairs = [("Buenos días", "Allin p'unchaw"), ...]
    # for esp, que in spanish_quechua_pairs:
    #     instruction = f"Imaynatan 'español simipi' kay rimayta nina: '{que}'"
    #     output = f"'{que}' nisqaqa 'español simipi' '{esp}' ninanmi."
    #     formatted_entry = CHAT_TEMPLATE.format(instruction=instruction, output=output)
    #     dataset_entries.append({"text": formatted_entry})
    
    # --- TIPO 3: Preguntas y respuestas generales (creadas por ti) ---
    # general_qa = [("¿Pikunataq Inka karqan?", "Inkakunaqa ñawpa pacha Tawantinsuyu suyu apuqkuna karqan..."), ...]
    # for question, answer in general_qa:
    #     instruction = question
    #     output = answer
    #     formatted_entry = CHAT_TEMPLATE.format(instruction=instruction, output=output)
    #     dataset_entries.append({"text": formatted_entry})


    # Guardar el dataset final en formato JSONL
    with open(OUTPUT_JSONL_FILE, 'w', encoding='utf-8') as f:
        for entry in dataset_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"✓ Dataset creado con {len(dataset_entries)} entradas y guardado en '{OUTPUT_JSONL_FILE}'")


if __name__ == '__main__':
    create_dataset_from_text()