import os
import threading
from faster_whisper import WhisperModel
from gigaam import GigaAMEmo
from llama_cpp import Llama

# Глобальные переменные
_whisper_model = None
_whisper_lock = threading.Lock()

_emotion_model = None
_emotion_lock = threading.Lock()

_llm = None
_llm_lock = threading.Lock()

def get_whisper():
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            _whisper_model = WhisperModel(
                "h2oai/faster-whisper-large-v3-turbo",
                device="cpu",
                compute_type="int8"
            )
    return _whisper_model

def get_emotion_model():
    global _emotion_model
    with _emotion_lock:
        if _emotion_model is None:
            _emotion_model = GigaAMEmo.from_pretrained()
            _emotion_model.eval()
    return _emotion_model

def get_llm():
    global _llm
    with _llm_lock:
        if _llm is None:
            model_path = os.getenv("LLM_MODEL_PATH", "models/model-q4_K.gguf")
            _llm = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_threads=int(os.getenv("OMP_NUM_THREADS", 8)),
                verbose=False
            )
    return _llm
