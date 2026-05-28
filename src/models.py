import os
import threading
import logging
import warnings
import gigaam
from llama_cpp import Llama

# Подавляем FutureWarning от torch/gigaam
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

# Глобальные переменные
_asr_model = None
_asr_lock = threading.Lock()

_llm = None
_llm_lock = threading.Lock()

def get_asr_model():
    global _asr_model
    with _asr_lock:
        if _asr_model is None:
            logger.info("Initializing GigaAM ASR model (v3_e2e_ctc)...")
            try:
                _asr_model = gigaam.load_model('v3_e2e_ctc')
                _asr_model.eval()
                logger.info("GigaAM ASR model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load GigaAM ASR model: {e}")
                raise
    return _asr_model

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
