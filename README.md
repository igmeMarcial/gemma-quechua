#### Supervised Fine-tuning (SFT)

gemma-quechua/
├── data/
│ └── quechua_corpus.txt ← corpus limpio en quechua (raw text)
├── notebooks/
│ └── pretrain_gemma_unsloth.ipynb ← notebook para entrenamiento
├── scripts/
│ ├── extract_text.py ← extrae texto de PDF/DOCX
│ └── clean_text.py ← limpieza básica del texto
├── requirements.txt ← dependencias: unsloth, datasets, PyMuPDF, nltk...
└── README.md ← descripción del proyecto y pasos

gemma-quechua/
├── src/
│ └── quechua/ ← tu paquete importable
│ ├── **init**.py
│ ├── extract_text.py ←
│ └── clean_text.py ← movido desde scripts/
├── data/
│ └── quechua_corpus.txt
├── notebooks/
│ └── pretrain_gemma_unsloth.ipynb
├── requirements.txt
├── README.md
├── setup.py

# Create Virtual Environment

python -m venv .venv

# Create, Activate Virtual Environment in Linux

python -m venv .venv && source .venv/bin/activate

# Create, Activate Virtual Environment in Windows

python -m venv .venv && .venv\Scripts\activate

# Instalar en modo desarrollo (editable)

pip install -r requirements.txt

🚀 ¿Qué deberías hacer ahora?
Paso 1: Extraer texto de tus PDFs en quechua
→ Te puedo dar un script en Python con PyMuPDF que lo haga.

Paso 2: Limpiar ese texto y convertirlo en .jsonl
→ Te ayudaré a hacer un script que genere {"text": "...", "language": "quechua"} por línea.

Paso 3: Usar synthetic-dataset-generator para crear instrucciones y respuestas.
→ Aquí usarás tus textos como "seed" para generar prompts del tipo:

json
Copy
Edit
{"instruction": "Resume este texto en quechua", "input": "text", "output": "..." }

{
"instruction": "Imapaqmi allin llama uywa, Inkakunap pachankupi?",
"input": "",
"output": "Llamaqa allinmi karqan llank'anapaq, aychanta mikhunapaq, chaymanta millmanwan p'achakunata ruwanapaq."
}

Objetivo: Hacer que Gemma "piense" en quechua. No le enseñas a ser un chatbot, le enseñas el idioma en profundidad: su gramática, vocabulario, y cómo se conectan las ideas.
