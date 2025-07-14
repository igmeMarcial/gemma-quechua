from PyPDF2 import PdfReader, PdfWriter

INPUT_PATH = "E:/kaggle/gemma_data/biblia/bibliatest.pdf"
OUTPUT_PATH = "E:/kaggle/gemma_data/biblia/pdf_trimmed.pdf"
MAX_PAGES = 20

# Cargar el PDF
reader = PdfReader(INPUT_PATH)
writer = PdfWriter()

# Agregar solo las primeras 20 páginas (o menos si no tiene tantas)
num_pages = min(MAX_PAGES, len(reader.pages))
for i in range(num_pages):
    writer.add_page(reader.pages[i])

# Guardar nuevo archivo
with open(OUTPUT_PATH, "wb") as f:
    writer.write(f)

print(f"Guardado: {OUTPUT_PATH} ({num_pages} páginas)")
