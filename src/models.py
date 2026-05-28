import os
import threading
import logging
import warnings
import gigaam
import torch
from llama_cpp import Llama

# Подавляем FutureWarning от torch/gigaam
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

# Глобальные переменные для моделей
_asr_model = None
_asr_lock = threading.Lock()

_emotion_model = None
_emotion_lock = threading.Lock()

_llm_model = None
_llm_lock = threading.Lock()

def get_asr_model():
    global _asr_model
    with _asr_lock:
        if _asr_model is None:
            model_name = "v3_e2e_ctc"
            logger.info(f"Initializing GigaAM ASR model ({model_name})...")
            try:
                _asr_model = gigaam.load_model(model_name)
                logger.info("GigaAM ASR model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load GigaAM ASR model: {e}")
                raise
    return _asr_model

def get_emotion_model():
    global _emotion_model
    with _emotion_lock:
        if _emotion_model is None:
            logger.info("Initializing GigaAMEmo model...")
            try:
                _emotion_model = gigaam.load_model('emo')
                _emotion_model.eval()
                logger.info("GigaAMEmo model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load GigaAMEmo model: {e}")
                raise
    return _emotion_model

def get_llm():
    """
    Возвращает экземпляр LLM.
    В многопроцессорной среде каждый воркер получит свой экземпляр.
    В многопоточной среде внутри одного процесса используется блокировка.
    """
    global _llm_model
    with _llm_lock:
        if _llm_model is None:
            model_path = os.getenv("LLM_MODEL_PATH", "models/model-q4_K.gguf")
            if os.path.isdir(model_path):
                for f in os.listdir(model_path):
                    if f.endswith(".gguf"):
                        model_path = os.path.join(model_path, f)
                        break

            n_threads = int(os.getenv("OMP_NUM_THREADS", 8))
            logger.info(f"Initializing Llama model (n_threads={n_threads}) from {model_path}...")
            try:
                _llm_model = Llama(
                    model_path=model_path,
                    n_ctx=4096,
                    n_threads=n_threads,
                    chat_format="llama-3",
                    verbose=False
                )
                logger.info("Llama model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Llama model: {e}")
                raise
    return _llm_model

class LockedLlama:
    """Обертка для потокобезопасного использования LLM внутри одного процесса."""
    def __init__(self):
        self._lock = threading.Lock()

    def create_chat_completion(self, *args, **kwargs):
        model = get_llm()
        with self._lock:
            return model.create_chat_completion(*args, **kwargs)

    def create_completion(self, *args, **kwargs):
        model = get_llm()
        with self._lock:
            return model.create_completion(*args, **kwargs)

_locked_llm_instance = None

def get_locked_llm():
    global _locked_llm_instance
    if _locked_llm_instance is None:
        _locked_llm_instance = LockedLlama()
    return _locked_llm_instance

# Для обратной совместимости
def get_whisper():
    return get_asr_model()
