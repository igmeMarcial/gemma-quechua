import torch
import os
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm
import torch._dynamo

torch._dynamo.disable()

print("✅ (1/8) Importaciones y configuración inicial completas.")
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
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
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
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output_text},
        ]
        conversations.append(convo)
    return {"conversations": conversations}


dataset = load_dataset("json", data_files={"train": DATASET_PATH}, split="train")
dataset = dataset.map(
    alpaca_to_sharegpt, batched=True, remove_columns=["instruction", "input", "output"]
)
if isinstance(dataset, dict):
    dataset = Dataset.from_dict(dataset)

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
        ).removeprefix("<bos>")
        for convo in convos
    ]
    return {"text": texts}


dataset = dataset.map(formatting_prompts_func, batched=True)
print("   - Plantilla de chat aplicada. El dataset está listo para el SFTTrainer.")
print("\n" + "=" * 60)
print("🔍 VERIFICACIÓN DEL FORMATO PROCESADO:")
print("=" * 60)
print("Ejemplo del primer registro procesado:")
first = next(iter(dataset))
print(first["text"][:300] + "..." if len(first["text"]) > 300 else first["text"])
print("=" * 60 + "\n")

# ==============================================================================
# 6. Definir el Entrenador (SFTTrainer)
# ==============================================================================
print("✅ (6/8) Configurando el SFTTrainer...")

trainer = SFTTrainer(
    model=model,
    eval_dataset=None,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        warmup_steps=10,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
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
# 7. Habilitar Entrenamiento solo en Respuestas
# ==============================================================================
print("✅ (7/9) Habilitando el entrenamiento solo en las respuestas del asistente...")
trainer = train_on_responses_only(
    trainer,
    instruction_part="<start_of_turn>user\n",
    response_part="<start_of_turn>model\n",
)
print("   - Máscara de entrenamiento aplicada.")

# ==============================================================================
# 8. Mostrar Estadísticas de Memoria
# ==============================================================================

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")
# ==============================================================================
# 9. Iniciar el Entrenamiento y Guardar
# ==============================================================================

print(f"🚀 (9/9) ¡Iniciando entrenamiento por {EPOCHS} pasos!")
trainer.train()
print("   - Entrenamiento finalizado.")

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"🎉 ¡Éxito! Modelo guardado en la carpeta '{OUTPUT_DIR}'.")
