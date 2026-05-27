import subprocess
import time
import os
import signal
import sys
import threading
import queue
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

def enqueue_output(out, queue, name):
    for line in iter(out.readline, ''):
        queue.put((name, line.strip()))
    out.close()

def run_process(command, name, output_queue):
    print(f"Starting {name}...")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:{PROJECT_ROOT}:{PROJECT_ROOT}/src"

    p = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    t = threading.Thread(target=enqueue_output, args=(p.stdout, output_queue, name))
    t.daemon = True
    t.start()

    return p

def main():
    output_queue = queue.Queue()
    processes = []

    # 1. Запуск диспетчера (анализ)
    dispatcher_cmd = ["python3", "src/dispatcher.py"]
    dispatcher_proc = run_process(dispatcher_cmd, "Dispatcher", output_queue)
    processes.append((dispatcher_proc, "Dispatcher"))

    # 2. Запуск Streamlit (веб-интерфейс)
    dashboard_cmd = ["python3", "-m", "streamlit", "run", "dashboard/dashboard.py", "--server.port", "80", "--server.address", "0.0.0.0"]
    dashboard_proc = run_process(dashboard_cmd, "Dashboard", output_queue)
    processes.append((dashboard_proc, "Dashboard"))

    def signal_handler(sig, frame):
        print("\nShutting down...")
        for p, name in processes:
            print(f"Killing {name}...")
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("System is running. Press Ctrl+C to stop.")

    try:
        while True:
            # Читаем все накопленные сообщения из очереди
            try:
                while True:
                    name, line = output_queue.get_nowait()
                    print(f"[{name}] {line}")
            except queue.Empty:
                pass

            # Проверяем живость процессов
            for p, name in processes:
                if p.poll() is not None:
                    print(f"ERROR: {name} process exited with code {p.returncode}")
                    # В реальной системе тут можно добавить перезапуск

            time.sleep(0.1)
    except Exception as e:
        print(f"Main loop error: {e}")
        signal_handler(None, None)

if __name__ == "__main__":
    main()
