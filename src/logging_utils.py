import logging
import os

def setup_logging(level=None):
    if level is None:
        level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_str, logging.INFO)

    # Сбрасываем существующие хендлеры, если они есть (важно для spawn процессов)
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers[:]:
            root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s [%(name)s] [%(process)d] %(message)s',
        force=True
    )

    # Отключаем слишком шумные логи от сторонних библиотек
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
