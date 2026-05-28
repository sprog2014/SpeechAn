import streamlit as st
import hmac
import sys
import os

# Добавляем путь к src, чтобы найти config.py и db_utils.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from config import WEB_USER, WEB_PASSWORD, PG_CONFIG

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


st.title("Система анализа речи")
st.write("Добро пожаловать в панель управления и мониторинга.")

st.sidebar.success("Выберите раздел выше.")

st.info("""
Используйте боковое меню для перехода между разделами:
- **Аналитика**: Просмотр результатов и графиков (бывший Dashboard)
- **Настройки и Управление**: Управление промптами, запуск/остановка системы и ручная обработка.
""")
