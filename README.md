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

# Crear entorno virtual (si no existe)

python -m venv .venv

# Create, Activate Virtual Environment in Linux

python -m venv .venv && source .venv/bin/activate

# Create, Activate Virtual Environment in Windows

python -m venv .venv && .venv\Scripts\activate

# Instalar en modo desarrollo (editable)

pip install -r requirements.txt
