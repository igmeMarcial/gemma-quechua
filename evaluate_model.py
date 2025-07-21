import torch
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from transformers.generation.streamers import TextStreamer
import os

# ==============================================================================
# SETUP
# ==============================================================================
MODEL_PATH = "outputs/gemma-3n-quechua"
BASE_MODEL = "unsloth/gemma-3n-E4B-it"

# ==============================================================================
# CARGAR MODELO FINE-TUNEADO
# ==============================================================================
print("🔄 Cargando modelo fine-tuneado...")

# Verificar si existe el modelo fine-tuneado
if os.path.exists(MODEL_PATH):
    print(f"✅ Encontrado modelo en: {MODEL_PATH}")
    try:
        # Cargar modelo con Unsloth
        model, tokenizer = FastModel.from_pretrained(
            model_name=MODEL_PATH,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        print("✅ Modelo cargado exitosamente con Unsloth")
    except Exception as e:
        print(f"❌ Error cargando con Unsloth: {e}")
        print("🔄 Intentando cargar modelo base...")
        model, tokenizer = FastModel.from_pretrained(
            model_name=BASE_MODEL,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        print("⚠️ Cargado modelo base (sin fine-tuning)")
else:
    print(f"❌ No se encontró modelo en: {MODEL_PATH}")
    print("🔄 Cargando modelo base...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    print("⚠️ Cargado modelo base (sin fine-tuning)")

# ==============================================================================
# CONFIGURAR CHAT TEMPLATE
# ==============================================================================
print("🔧 Configurando chat template...")
tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-3",
)


# ==============================================================================
# FUNCIÓN DE PRUEBA
# ==============================================================================
def test_model(prompt_text, max_tokens=128, show_stream=True):
    """
    Función para probar el modelo con un prompt dado.
    Args:
        prompt_text (str): Texto del prompt.
        max_tokens (int, optional): Número máximo de tokens a generar. Defaults to 128.
        show_stream (bool, optional): Mostrar salida en streaming. Defaults to True.
    """
    # Crear mensaje en formato correcto
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]

    # Preparar input
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🧪 Prompt: {prompt_text}")
    print("🤖 Respuesta: ", end="")

    if show_stream:
        # Generar con streaming
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                streamer=streamer,
                do_sample=True,
            )
    else:
        # Generar sin streaming
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                do_sample=True,
            )

        # Decodificar respuesta
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extraer solo la respuesta del modelo
        if "<start_of_turn>model" in response:
            response = response.split("<start_of_turn>model")[-1].strip()

        print(response)

    print("-" * 60)


# ==============================================================================
# EXMPLOS DE PRUEBA
# ==============================================================================
test_examples = [
    # Traducción español -> quechua
    "Traduce al quechua: Hola, ¿cómo estás?",
    "Traduce al quechua: Gracias por tu ayuda.",
    "Traduce al quechua: Me gusta la música.",
    "Traduce al quechua: ¿Dónde está la escuela?",
    "Traduce al quechua: Amarillo",
    "Traduce al quechua: Yo cosecho maiz para la chala",
    "Traduce al quechua: ¿En qué circunstancias le pegó su esposo?",
    "Traduce al quechua: Cuando comenzamos a discutir.",
    "Traduce al quechua: Señor, ¿Qué días abre esta oficina?",
    "Traduce al quechua: La oficina atiende lunes, miércoles y viernes.",
    "Traduce al quechua: La oficina atiende martes y jueves.",
    "Traduce al quechua: Regrese mañana en la mañana.",
    "Traduce al quechua: Regrese mañana en la tarde.",
    "Traduce al quechua: Regrese pasado mañana.",
    "Traduce al quechua: Regrese el siguiente lunes.",
    "Traduce al quechua: Hoy no hay atención, es feriado.",
    "Traduce al quechua: Hoy no hay atención, los trabajadores están de huelga.",
    # Traducción quechua -> español
    "Traduce al español: Allin p´unchaw wiraqucha",
    "Traduce al español: Allin p´unchaw doctor",
    "Traduce al español: Allin p´unchay taytay",
    "Traduce al español: Allin p’unchaw kachun runaymasiykuna",
    "Traduce al español: Wiraqucha ¿Imay horastan llamk’ayta qallarinku?",
    "Traduce al español: Kay oficina pusaq horastakichakun",
    "Traduce al español: Chunca iskay pacha kimsachunkayuqta kutimunki",
    "Traduce al español: Kay wasiqa pichqa pachatan wisq’akunqa",
    "Traduce al español: Allinllachu?",
    "Traduce al español: Añay kusay.",
    # Preguntas sobre quechua
    "¿Cómo se dice 'agua' en quechua?",
    "¿Cuáles son los números del 1 al 5 en quechua?",
    # Conversación en quechua
    "Responde en quechua: ¿Castellano simita yachankichu, icha rimay t’ikraqtachu munawaq?",
]

# ==============================================================================
# RUN TESTS
# ==============================================================================
if __name__ == "__main__":
    print("🚀 Iniciando pruebas del modelo...")
    print("=" * 60)

    # Mostrar información del dispositivo
    if torch.cuda.is_available():
        print(f"🔥 GPU disponible: {torch.cuda.get_device_name()}")
        print(
            f"💾 VRAM libre: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
        )
    else:
        print("💻 Usando CPU")

    print("=" * 60)

    # Ejecutar todas las pruebas
    for i, example in enumerate(test_examples, 1):
        print(f"\n📝 Prueba {i}/{len(test_examples)}:")
        try:
            test_model(example, max_tokens=100, show_stream=True)
        except Exception as e:
            print(f"❌ Error en prueba {i}: {e}")
            continue

    print("\n✅ Pruebas completadas!")

    # Prueba interactiva opcional
    print("\n" + "=" * 60)
    print("🎯 ¿Quieres hacer una prueba interactiva? (s/n)")
    if input().lower().startswith("s"):
        while True:
            user_input = input("\n📝 Tu pregunta (o 'salir' para terminar): ")
            if user_input.lower() in ["salir", "exit", "quit"]:
                break
            try:
                test_model(user_input, max_tokens=150, show_stream=True)
            except Exception as e:
                print(f"❌ Error: {e}")

    print("👋 ¡Se ha terminado la prueba!")
