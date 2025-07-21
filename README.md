# 🇵🇪 Llaqta-Simi: Revitalizing the Quechua Language with Gemma 3

**A project for the [Google - Gemma 3n Hackathon](https://www.kaggle.com/competitions/google-gemma-3n-hackathon)**

**Author:** Marcial Igme

---

## 🌟 Project Vision: A Digital Bridge for an Ancestral Language

**Quechua, the language of the Inca Empire, is spoken by millions yet faces a growing digital divide.** The lack of high-quality AI tools limits its presence in the modern world, accelerating its decline and isolating its speakers.

**Llaqta-Simi (The People's Voice)** is my answer to this challenge. This project does more than create a translator; it aims to **breathe digital life into Quechua**. Our mission is to leverage the power and efficiency of **Gemma 3** to build a language model that doesn't just _process_ Quechua, but _understands_ it with deep cultural and grammatical nuance.

This project directly addresses the core pillars of the hackathon:

- **Accessibility & Education:** We are creating a foundational tool that breaks down communication barriers for native speakers, researchers, and anyone wishing to learn the language.
- **Positive Impact:** By strengthening an indigenous language, we preserve an invaluable cultural heritage for future generations.

---

## 🚀 The Solution: A High-Fidelity Quechua-Spanish Translator

**Llaqta-Simi** is a language model, powered by **Gemma 3N**, that has been meticulously fine-tuned to become an expert translator between Spanish and Quechua.

**[🎥 WATCH THE 3-MINUTE VIDEO DEMO] (https://your-video-link-on-youtube.com)**
_(This is the most important link! Make it prominent.)_

### ✨ Key Features

- **High-Fidelity Bidirectional Translation:** Capable of accurately translating nuances, dialects, and cultural context.
- **Efficiency Powered by Unsloth:** The fine-tuning process was completed with remarkable speed and significantly reduced VRAM usage, proving the feasibility of tuning powerful models on accessible hardware.
- **A Foundation for the Future:** This model is more than just a translator. It is a foundational base upon which more complex tools can be built: tutoring chatbots, Quechua voice assistants, sentiment analysis tools, and more.

---

## 🛠️ Technical Implementation: From Data to Model

The success of this project lies in a rigorous data engineering and training pipeline.

### 1. Dataset Curation and Preparation

The greatest challenge for low-resource languages is the lack of quality data. To overcome this:

- **Leveraging Existing Resources:** The project utilized the `somosnlp-hackathon-2022/spanish-to-quechua` dataset, an invaluable corpus containing over **128,000 translation pairs**.
- **Data Transformation:** The dataset, originally in a simple format, was transformed into a conversational structure (ShareGPT/ChatML style), following best practices for modern model fine-tuning.
- **Chat Template Formatting:** The official `gemma-3` chat template was applied to ensure the model natively understands the question-and-answer structure.

### 2. Efficient Fine-Tuning with Unsloth

Training was performed using the **Unsloth** framework, which enabled:

- **4-bit Loading (QLoRA):** The `unsloth/gemma-3n-E4B-it` base model was loaded in 4-bit, drastically reducing VRAM consumption and making training on consumer-grade hardware possible.
- **PEFT (LoRA):** Low-Rank Adaptation adapters were used to train only a small fraction of the model's parameters, resulting in a training process that is up to 2x faster.
- **Training Optimization:** The `train_on_responses_only` technique was implemented to focus the model exclusively on learning to generate the correct Quechua responses, thereby improving accuracy.

The complete, reproducible script can be found in `train_quechua.py`.

### 3. Tech Stack

- **Base Model:** `unsloth/gemma-3n-E4B-it`
- **Training Framework:** `Unsloth`
- **Core Libraries:** `PyTorch`, `Hugging Face Transformers`, `TRL`, `Datasets`, `PEFT`
- **Execution Environment:** RunPod (GPU: NVIDIA RTX A4000)

---

## 🚀 Getting Started

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv && source .venv/bin/activate  # Linux/Mac
python -m venv .venv && .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

## 🗂️ 2. Prepare Your Quechua Dataset

### Required Dataset Format

The script assumes the training dataset is located at gemma_data/quechua_hf_alpaca.jsonl. If you have already generated it, place it in the project's root directory. Otherwise, you can generate it using the prepare_hf_dataset.py script.

```json
{
  "instruction": "Translate to Quechua",
  "input": "Good morning",
  "output": "Allin p'unchay"
}
```

## 🚀 3. Train Model

### Basic Command

```bash
python train_gemma3_sft.py
```

## 🧑‍💻 Author

This project was developed with passion by Marcial Igme for the Gemma 3n Impact Challenge. As a developer and native Quechua speaker, I created this project to show that cutting-edge AI can—and must—serve to preserve, protect, and empower our native languages and cultures. Quechua is my mother tongue, and that personal connection is what drives my commitment to building tools that support my community.

cd /workspace
tar -czvf gemma3_quechua_model.tar.gz gemma3_quechua_model
