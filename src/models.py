import os
import threading
from faster_whisper import WhisperModel
import gigaam
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
                compute_type="int8",
                cpu_threads=int(os.getenv("OMP_NUM_THREADS", 8))
            )
    return _whisper_model

def get_emotion_model():
    global _emotion_model
    with _emotion_lock:
        if _emotion_model is None:
            _emotion_model = gigaam.load_model('emo')
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

class LockedLlama:
    def __init__(self, llm_instance):
        self.llm = llm_instance
        self.lock = threading.Lock()

    def create_completion(self, *args, **kwargs):
        with self.lock:
            return self.llm.create_completion(*args, **kwargs)

_locked_llm = None
_locked_llm_lock = threading.Lock()

def get_locked_llm():
    global _locked_llm
    with _locked_llm_lock:
        if _locked_llm is None:
            _locked_llm = LockedLlama(get_llm())
    return _locked_llm
