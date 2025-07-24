import os

try:
    import torch

    torch._dynamo.config.cache_size_limit = 64
    print("✅ Límite de caché de TorchDynamo aumentado a 64.")
except ImportError:
    print(
        "⚠️  No se pudo importar torch para configurar el límite de caché. Asegúrate de que está instalado."
    )
    pass

from unsloth import FastModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

print("✅ (1/7) Configuración inicial...")
MODEL_NAME = "unsloth/gemma-3n-E4B-it"
DATASET_PATH = "gemma_data/quechua_hf_alpaca.jsonl"
OUTPUT_DIR = "outputs/gemma-3n-quechua"
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 2
GRAD_ACCUMULATION = 4
LR = 2e-4
EPOCHS = 1

# ==============================================================================
# 2. Cargar Modelo y Tokenizer
# ==============================================================================
print(f"✅ (2/7) Cargando el modelo base '{MODEL_NAME}'...")
model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)
print("   - Modelo y tokenizer cargados.")

# ==============================================================================
# 3. Preparar Modelo para Fine-tuning (LoRA)
# ==============================================================================
print("✅ (3/7) Aplicando adaptadores LoRA al modelo...")
model = FastModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
print("   - Adaptadores LoRA aplicados.")

# ==============================================================================
# 4. Cargar y Formatear el Dataset (MÉTODO SIMPLIFICADO Y DIRECTO)
# ==============================================================================
print(f"✅ (4/7) Cargando y formateando dataset desde '{DATASET_PATH}'...")

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input_text, output_text in zip(instructions, inputs, outputs):
        text = (
            alpaca_prompt.format(instruction, input_text, output_text)
            + tokenizer.eos_token
        )
        texts.append(text)
    return {
        "text": texts,
    }


dataset = load_dataset("json", data_files={"train": DATASET_PATH}, split="train")
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)
print(
    f"   - Dataset formateado directamente a la columna 'text'. {len(dataset)} ejemplos listos."
)

# ==============================================================================
# 5. Definir el Entrenador (SFTTrainer)
# ==============================================================================
print("✅ (5/7) Configurando el SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=None,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        warmup_steps=10,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs_temp",
        report_to="none",
    ),
)
print("   - Trainer configurado.")

# ==============================================================================
# 6. Mostrar Estadísticas de Memoria y Iniciar
# ==============================================================================
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

print(f"🚀 (6/7) ¡Iniciando entrenamiento por {EPOCHS} épocas!")
trainer.train()
print("   - Entrenamiento finalizado.")

# ==============================================================================
# 7. Guardar el Modelo Final
# ==============================================================================
print(f"💾 (7/7) Guardando el modelo final en '{OUTPUT_DIR}'...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"🎉 ¡Éxito! Modelo guardado en la carpeta '{OUTPUT_DIR}'.")
