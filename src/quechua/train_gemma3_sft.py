import torch
import os
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm

print("✅ (1/8) Importaciones y configuración inicial completas.")
MODEL_NAME = "unsloth/gemma-3n-E4B-it"
DATASET_PATH = "quechua_hf_alpaca.jsonl"
OUTPUT_DIR = "outputs/gemma-3n-quechua-translator"
MAX_SEQ_LENGTH = 1024
MAX_STEPS = 1200
BATCH_SIZE = 2
GRAD_ACCUMULATION = 4

# ==============================================================================
# 2. Cargar Modelo y Tokenizer
# ==============================================================================

print(f"✅ (2/8) Cargando el modelo base '{MODEL_NAME}' con Unsloth...")
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
print("✅ (3/8) Aplicando adaptadores LoRA al modelo...")
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
# 4. Cargar y Transformar el Dataset (LA PARTE MÁS IMPORTANTE)
# ==============================================================================
print(f"✅ (4/8) Cargando y transformando el dataset desde '{DATASET_PATH}'...")


# Función para convertir tu formato Alpaca al formato ShareGPT que espera el notebook.
def alpaca_to_sharegpt(examples):
    conversations = []
    for instruction, input_text, output_text in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        if not instruction or not output_text:
            continue
        user_content = instruction
        if input_text:
            user_content += f"\n{input_text}"
        convo = [
            {"from": "user", "content": user_content},
            {"from": "assistant", "content": output_text},
        ]
        conversations.append(convo)
    return {"conversations": conversations}


dataset = load_dataset("json", data_files={"train": DATASET_PATH}, split="train")
dataset = dataset.map(
    alpaca_to_sharegpt, batched=True, remove_columns=["instruction", "input", "output"]
)
print(
    f"   - Dataset convertido al formato ShareGPT. Ejemplos estimados: {dataset.num_rows if hasattr(dataset, 'num_rows') else 'desconocido'}"
)

# ==============================================================================
# 5. Aplicar la Plantilla de Chat de Gemma-3
# ==============================================================================
print("✅ (5/8) Aplicando la plantilla de chat 'gemma-3'...")
tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-3",
)


def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        )
        for convo in convos
    ]
    return {"text": texts}


dataset = dataset.map(formatting_prompts_func, batched=True)
print("   - Plantilla de chat aplicada. El dataset está listo para el SFTTrainer.")

# ==============================================================================
# 6. Definir el Entrenador (SFTTrainer)
# ==============================================================================
print("✅ (6/8) Configurando el SFTTrainer...")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        warmup_steps=10,
        max_steps=MAX_STEPS,
        learning_rate=2e-4,
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
# 7. Optimización: Entrenar solo en las Respuestas
# ==============================================================================
print("✅ (7/8) Configurando para entrenar solo en las respuestas del asistente...")
trainer = train_on_responses_only(
    trainer,
    instruction_part="<start_of_turn>user\n",
    response_part="<start_of_turn>model\n",
)
print("   - Máscara de entrenamiento aplicada.")

# ==============================================================================
# 8. Iniciar el Entrenamiento y Guardar
# ==============================================================================

print(f"🚀 (8/8) ¡Iniciando entrenamiento por {MAX_STEPS} pasos!")
trainer.train()
print("   - Entrenamiento finalizado.")

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"🎉 ¡Éxito! Modelo guardado en la carpeta '{OUTPUT_DIR}'.")
