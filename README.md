# 🦙 Quechua Language Fine-Tuning for Gemma 3

![Gemma and Quechua](https://via.placeholder.com/800x200?text=Gemma+3+%2B+Quechua+Language)

This project fine-tunes Google's Gemma 3 model to deeply understand Quechua language structure, grammar, and semantics using Unsloth for efficient training.

## 🌟 Project Vision

**Objective:** Make Gemma "think" in Quechua by teaching:

- Deep language understanding (grammar, vocabulary)
- How ideas connect in Quechua
- Natural language patterns (not just chatbot responses)

## 🛠️ Technical Implementation

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

Place your training files in `gemma-data/` directory using **Alpaca-style JSONL** format. Each line should contain:

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

Developed by Marcial Igme for the Gemma 3n Impact Challenge.
