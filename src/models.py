import os
import threading
import logging
import warnings
import gigaam
from faster_whisper import WhisperModel
from llama_cpp import Llama

# Подавляем FutureWarning от torch/gigaam
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

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
            logger.info("Initializing Faster-Whisper model (large-v3-turbo)...")
            try:
                _whisper_model = WhisperModel(
                    "h2oai/faster-whisper-large-v3-turbo",
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=int(os.getenv("OMP_NUM_THREADS", 8))
                )
                logger.info("Faster-Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Faster-Whisper model: {e}")
                raise
    return _whisper_model

def get_emotion_model():
    global _emotion_model
    with _emotion_lock:
        if _emotion_model is None:
            logger.info("Initializing GigaAMEmo model...")
            try:
                # Предупреждение о fp16 обычно летит из gigaam.load_model при работе на CPU
                _emotion_model = gigaam.load_model('emo')
                _emotion_model.eval()
                logger.info("GigaAMEmo model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load GigaAMEmo model: {e}")
                raise
    return _emotion_model

def get_llm():
    global _llm
    with _llm_lock:
        if _llm is None:
            model_path = os.getenv("LLM_MODEL_PATH", "models/model-q4_K.gguf")
            logger.info(f"Initializing Llama model from {model_path}...")
            try:
                _llm = Llama(
                    model_path=model_path,
                    n_ctx=4096,
                    n_threads=int(os.getenv("OMP_NUM_THREADS", 8)),
                    # Указываем формат чата, если необходимо. Для Llama 3 обычно используется "llama-3"
                    chat_format="llama-3",
                    verbose=False
                )
                logger.info("Llama model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Llama model: {e}")
                raise
    return _llm

class LockedLlama:
    def __init__(self, llm_instance):
        self.llm = llm_instance
        self.lock = threading.Lock()

    def create_completion(self, *args, **kwargs):
        with self.lock:
            return self.llm.create_completion(*args, **kwargs)

    def create_chat_completion(self, *args, **kwargs):
        with self.lock:
            return self.llm.create_chat_completion(*args, **kwargs)

_locked_llm = None
_locked_llm_lock = threading.Lock()

def get_locked_llm():
    global _locked_llm
    with _locked_llm_lock:
        if _locked_llm is None:
            _locked_llm = LockedLlama(get_llm())
    return _locked_llm
