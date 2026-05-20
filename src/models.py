import os
from faster_whisper import WhisperModel
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from llama_cpp import Llama

# Глобальные объекты моделей (загружаются один раз при старте воркера)
whisper_model = None
emotion_model = None
emotion_feature_extractor = None
llm = None

def load_whisper():
    global whisper_model
    if whisper_model is None:
        whisper_model = WhisperModel("Systran/faster-whisper-large-v3-turbo",
                                     device="cpu",
                                     compute_type="int8")
    return whisper_model

def load_emotion_model():
    global emotion_model, emotion_feature_extractor
    if emotion_model is None:
        model_name = "SberDevices/GigaAMEmo"
        emotion_model = AutoModelForAudioClassification.from_pretrained(model_name, trust_remote_code=True)
        emotion_feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
        emotion_model.eval()
    return emotion_model, emotion_feature_extractor

def load_llm():
    global llm
    if llm is None:
        # Путь к скачанной GGUF модели Saiga/Llama3 8B
        model_path = os.getenv("LLM_MODEL_PATH", "models/saiga_llama3_8b_q4_K_M.gguf")
        llm = Llama(model_path=model_path, n_ctx=4096, n_threads=8, verbose=False)
    return llm