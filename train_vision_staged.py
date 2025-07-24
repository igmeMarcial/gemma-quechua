import torch
import os
from unsloth import FastVisionModel, get_chat_template
from datasets import load_dataset, Dataset, Image as HFImage
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator
from tqdm import tqdm

print("✅ (1/8) Configuración inicial...")

BASE_VISION_MODEL = "unsloth/gemma-3-4b-pt"
TEXT_LORA_ADAPTER_PATH = "outputs/gemma-3n-quechua"
VISION_DATASET_PATH = "gemma_data/vision_dataset/metadata.jsonl"
VISION_DATASET_DIR = "gemma_data/vision_dataset/"
OUTPUT_DIR = "outputs/gemma-3n-quechua-multimodal"


MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 1
GRAD_ACCUMULATION = 4
LR = 1e-4
EPOCHS = 1

# ==============================================================================
# 2. Cargar Modelo Base y Adaptadores de Texto (Etapa 1)
# ==============================================================================
print(f"✅ (2/8) Cargando el modelo base de visión '{BASE_VISION_MODEL}'...")
model, processor = FastVisionModel.from_pretrained(
    model_name=BASE_VISION_MODEL,
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)
print("   - Modelo base cargado.")

print(
    f"🧠 (3/8) Cargando adaptadores LoRA de la Etapa 1 desde '{TEXT_LORA_ADAPTER_PATH}'..."
)
try:
    model.load_adapter(TEXT_LORA_ADAPTER_PATH)
    print(
        "   - ✅ Adaptadores de texto cargados exitosamente. El modelo ahora sabe traducir."
    )
except Exception as e:
    print(f"   - ❗️ ADVERTENCIA: No se pudieron cargar los adaptadores de texto: {e}")
    print("   - Continuando el entrenamiento desde el modelo base de visión.")

# ==============================================================================
# 4. Activar Nuevos Adaptadores LoRA para la Etapa 2
# ==============================================================================
print(
    "🚀 (4/8) Aplicando adaptadores LoRA para el entrenamiento multimodal (Etapa 2)..."
)
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,  # False if not finetuning vision layers
    finetune_language_layers=True,  # False if not finetuning language layers
    finetune_attention_modules=True,  # False if not finetuning attention layers
    finetune_mlp_modules=True,  # False if not finetuning MLP layers
    r=16,  # The larger, the higher the accuracy, but might overfit
    lora_alpha=16,  # Recommended alpha == r at least
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
    target_modules="all-linear",  # Optional now! Can specify a list if needed
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
)
print("   - Adaptadores LoRA para la Etapa 2 aplicados.")

# ==============================================================================
# 5. Preparar el Dataset de Visión
# ==============================================================================
print(f"✅ (5/8) Preparando el dataset de visión desde '{VISION_DATASET_PATH}'...")
instruction = "Transcribe todo el texto en quechua que ves en esta imagen. Sé preciso y mantén el formato original."

dataset = load_dataset("json", data_files=VISION_DATASET_PATH, split="train")
dataset = dataset.cast_column("image_path", HFImage(decode=True)).rename_column(
    "image_path", "image"
)
dataset = dataset.with_format("torch", image_dir=VISION_DATASET_DIR)

print(f"   - Dataset de visión cargado con {len(dataset)} pares de imagen-texto.")


def convert_to_conversation(sample):
    image_rgb = sample["image"].convert("RGB")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": image_rgb},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["text"]}],
        },
    ]
    return {"messages": conversation}


converted_list = [
    convert_to_conversation(sample)
    for sample in tqdm(dataset, desc="   -> Formateando para entrenamiento")
]
converted_dataset = Dataset.from_list(converted_list)

# ==============================================================================
# 6. Preparar Plantilla y Entrenador
# ==============================================================================
print("✅ (6/8) Aplicando plantilla de chat y configurando SFTTrainer...")
processor = get_chat_template(processor, "gemma-3")
FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    train_dataset=converted_dataset,
    processing_class=processor.tokenizer,
    data_collator=UnslothVisionDataCollator(model, processor),
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        # use reentrant checkpointing
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,
        max_steps=30,
        # num_train_epochs = 2,          # Set this instead of max_steps for full training runs
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="steps",
        optim="adamw_torch_fused",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs",
        report_to="none",  # For Weights and Biases
        # You MUST put the below items for vision finetuning:
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=2048,
    ),
)
print("   - Trainer configurado.")

# ==============================================================================
# 7. Iniciar el Entrenamiento (Etapa 2)
# ==============================================================================
print(f"🚀 (7/8) ¡Iniciando entrenamiento de la Etapa 2 por {EPOCHS} épocas!")
trainer.train()
print("   - Entrenamiento de la Etapa 2 finalizado.")

# ==============================================================================
# 8. Guardar el Modelo Final
# ==============================================================================
print(f"💾 (8/8) Guardando el modelo final multimodal en '{OUTPUT_DIR}'...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"🎉 ¡Éxito! Tu modelo multimodal ha sido entrenado y guardado.")
