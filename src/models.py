import os
import threading
import logging
import warnings
import gigaam
import torch
from llama_cpp import Llama
from queue import Queue, Empty

# Подавляем FutureWarning от torch/gigaam
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

# Глобальные переменные
_asr_model = None
_asr_lock = threading.Lock()

_emotion_model = None
_emotion_lock = threading.Lock()

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

class LlamaPool:
    def __init__(self, model_path, n_instances=10, n_ctx=4096, n_threads=8):
        self.model_path = model_path
        self.n_instances = n_instances
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.pool = Queue()
        self._lock = threading.Lock()
        self._initialized = False

    def _initialize(self):
        with self._lock:
            if self._initialized:
                return
            logger.info(f"Initializing LlamaPool with {self.n_instances} instances...")
            for i in range(self.n_instances):
                logger.info(f"Loading Llama instance {i+1}/{self.n_instances}...")
                model = Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    chat_format="llama-3",
                    verbose=False
                )
                self.pool.put(model)
            self._initialized = True
            logger.info("LlamaPool initialized successfully")

    def get_model(self):
        if not self._initialized:
            self._initialize()
        return self.pool.get()

    def release_model(self, model):
        self.pool.put(model)

_llama_pool = None
_llama_pool_lock = threading.Lock()

def get_llama_pool():
    global _llama_pool
    with _llama_pool_lock:
        if _llama_pool is None:
            model_path = os.getenv("LLM_MODEL_PATH", "models/model-q4_K.gguf")
            # If path is a directory, look for the gguf file inside
            if os.path.isdir(model_path):
                for f in os.listdir(model_path):
                    if f.endswith(".gguf"):
                        model_path = os.path.join(model_path, f)
                        break

            n_instances = int(os.getenv("LLM_POOL_SIZE", 10))
            n_threads = int(os.getenv("OMP_NUM_THREADS", 8))

            _llama_pool = LlamaPool(model_path, n_instances=n_instances, n_threads=n_threads)
    return _llama_pool

# Для совместимости с текущим llm_analysis.py, если он использует get_locked_llm
class PoolLlamaWrapper:
    def __init__(self, pool):
        self.pool = pool

    def create_chat_completion(self, *args, **kwargs):
        model = self.pool.get_model()
        try:
            return model.create_chat_completion(*args, **kwargs)
        finally:
            self.pool.release_model(model)

    def create_completion(self, *args, **kwargs):
        model = self.pool.get_model()
        try:
            return model.create_completion(*args, **kwargs)
        finally:
            self.pool.release_model(model)

_locked_llm = None
def get_locked_llm():
    global _locked_llm
    if _locked_llm is None:
        _locked_llm = PoolLlamaWrapper(get_llama_pool())
    return _locked_llm

# Alias for backward compatibility if needed
def get_llm():
    return get_locked_llm()

def get_whisper():
    # Deprecated but kept for compatibility during migration
    return get_asr_model()
