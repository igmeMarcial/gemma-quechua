from unsloth import FastVisionModel, FastLanguageModel, get_chat_template
from datasets import load_dataset
import torch
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# =============================
# 1. Cargar modelo y procesador
# =============================


model, processor = FastVisionModel.from_pretrained(
    "unsloth/gemma-3-4b-pt",
    load_in_4bit=True,  # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
)

# =============================
# 2. Activar LoRA
# =============================

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


# =============================
# 3. Preparar dataset
# =============================
instruction = "Write the LaTeX representation for this image."


def convert_to_conversation(sample):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": sample["image"]},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": sample["text"]}]},
    ]
    return {"messages": conversation}


print("⏳ Descargando dataset...")
dataset = load_dataset(
    "unsloth/LaTeX_OCR", split="train"
)  # TODO: aqui tensria que trear dataset de imagenes en quechua
converted_dataset = Dataset.from_list(
    [convert_to_conversation(sample) for sample in dataset]
)

# =============================
# 4. Preparar template y modo entrenamiento
# =============================

processor = get_chat_template(processor, "gemma-3")
FastVisionModel.for_training(model)


# =============================
# 5. Configurar entrenamiento
# =============================

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

# =============================
# 6. Entrenamiento
# =============================

print("🚀 Entrenando modelo vision...")
trainer_stats = trainer.train()

# =============================
# 7. Guardar adaptadores LoRA
# =============================
print("💾 Guardando modelo...")
model.save_pretrained("lora_model")
processor.save_pretrained("lora_model")

print("✅ Entrenamiento finalizado.")
