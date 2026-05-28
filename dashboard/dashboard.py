import streamlit as st
import hmac
import sys
import os

# Добавляем путь к src, чтобы найти config.py и db_utils.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from config import WEB_USER, WEB_PASSWORD

st.set_page_config(page_title="Speech Analytics", layout="wide")

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["username"], WEB_USER) and \
           hmac.compare_digest(st.session_state["password"], WEB_PASSWORD):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the passowrd is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for username & password.
    st.text_input("Username", key="username")
    st.text_input("Password", type="password", key="password")
    st.button("Log in", on_click=password_entered)

    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False

if not check_password():
    st.stop()

# Обработка запроса на прослушивание файла (дублируем логику для работы ссылок из аналитики)
if "linkedid" in st.query_params:
    linkedid = st.query_params["linkedid"]
    from sqlalchemy import create_engine, text
    db_url = f"postgresql://{PG_CONFIG['user']}:{PG_CONFIG['password']}@{PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['dbname']}"
    engine = create_engine(db_url)
    with engine.connect() as conn:
        res = conn.execute(text("SELECT file_path FROM calls WHERE linkedid = :lid"), {"lid": linkedid}).fetchone()
        if res and res[0] and os.path.exists(res[0]):
            st.info(f"Прослушивание записи для звонка: {linkedid}")
            st.audio(res[0])
            if st.button("Закрыть плеер", key="close_player_main"):
                st.query_params.clear()
                st.rerun()
        else:
            st.error("Файл записи не найден.")
    engine.dispose()

st.title("Система анализа речи")
st.write("Добро пожаловать в панель управления и мониторинга.")

st.sidebar.success("Выберите раздел выше.")

st.info("""
Используйте боковое меню для перехода между разделами:
- **Аналитика**: Просмотр результатов и графиков (бывший Dashboard)
- **Настройки и Управление**: Управление промптами, запуск/остановка системы и ручная обработка.
""")
