import streamlit as st
import requests
import json

# --- 1. Константы ---
# URL вашего FastAPI эндпоинта (убедитесь, что он запущен)
API_URL = "http://127.0.0.1:8000/query/"

# Доступные разделы 1С для фильтрации (должны совпадать с теми, что используются в API и индексации)
# "все" означает отсутствие фильтрации по конкретному разделу
AVAILABLE_SECTIONS = ["все", "УНФ", "БП", "ERP", "Розница", "ЗУП", "Unknown"]

# --- 2. Конфигурация страницы Streamlit ---
st.set_page_config(
    page_title="RAG Система для Базы Знаний 1С",
    layout="wide",  # 'centered' или 'wide'
    initial_sidebar_state="expanded" # 'auto', 'expanded', 'collapsed'
)

# --- Заголовок и описание ---
st.title("Система Вопросов и Ответов по Базе Знаний 1С 🤖")
st.caption("Задайте вопрос на естественном языке и получите ответ, основанный на проиндексированных документах.")

# --- 3. Боковая панель для ввода параметров ---
with st.sidebar:
    st.header("Параметры запроса")

    selected_section = st.selectbox(
        "Выберите раздел 1С для поиска:",
        options=AVAILABLE_SECTIONS,
        index=0,  # "все" по умолчанию
        help="Фильтрация поиска по конкретному разделу 1С. 'все' - поиск по всем разделам."
    )

    st.markdown("---")
    st.subheader("О приложении")
    st.info(
        "Это приложение использует модель RAG (Retrieval Augmented Generation) "
        "для ответов на вопросы. Ответы генерируются на основе информации, "
        "найденной в загруженной базе знаний."
    )
    st.caption("Разработано с использованием LangChain, FastAPI и Streamlit.")

# --- 4. Основная область для ввода вопроса ---
question = st.text_area(
    "Введите ваш вопрос здесь:",
    height=150,
    placeholder="Например: Как создать новый отчет в УНФ?"
)

# Кнопка для отправки запроса
submit_button = st.button("Отправить запрос 🚀")

# --- 5. Логика обработки запроса и отображения результатов ---
if submit_button:
    if question.strip():  # Проверяем, что вопрос не пустой (после удаления пробелов)
        with st.spinner("Обработка вашего запроса... Пожалуйста, подождите. ⏳"):
            payload = {
                "query": question,
                "section": selected_section
            }

            try:
                print(f"Отправка запроса на API: {API_URL} с данными: {payload}")
                response = requests.post(API_URL, json=payload, timeout=180) # Увеличен таймаут
                response.raise_for_status()  # Проверка на HTTP ошибки (4xx или 5xx)

                api_response = response.json()
                print(f"Ответ от API получен: {api_response}")

                # --- 6. Отображение результатов ---
                st.subheader("Ответ:")
                st.markdown(api_response.get("answer", "Ответ не был получен от API."))

                st.subheader("Источники:")
                sources = api_response.get("sources", [])
                if sources:
                    for i, source in enumerate(sources):
                        expander_title = f"Источник {i+1}: {source.get('file_name', 'Неизвестный файл')}"
                        if source.get('section_1c') and source.get('section_1c') != 'Unknown':
                             expander_title += f" (Раздел: {source.get('section_1c')})"

                        with st.expander(expander_title):
                            if source.get('file_name'):
                                st.write(f"**Имя файла:** {source.get('file_name')}")
                            if source.get('section_1c'):
                                st.write(f"**Раздел 1С:** {source.get('section_1c')}")
                            if source.get('source_type'):
                                st.write(f"**Тип источника:** {source.get('source_type')}")
                            if source.get('full_path'):
                                st.caption(f"Полный путь: {source.get('full_path')}")
                            if source.get('page_number') is not None: # 0 может быть валидным номером страницы
                                st.write(f"**Номер страницы:** {source.get('page_number')}")
                            if source.get('db_table'):
                                st.write(f"**Таблица БД:** {source.get('db_table')}")
                            if source.get('record_id'):
                                st.write(f"**ID Записи:** {source.get('record_id')}")
                            if source.get('content_snippet'):
                                st.markdown("**Фрагмент содержимого:**")
                                st.markdown(f"> _{source.get('content_snippet')}_")
                else:
                    st.info("Источники для данного ответа не найдены или не были предоставлены API.")

            except requests.exceptions.Timeout:
                st.error(f"Ошибка: Запрос к API превысил время ожидания ({180} секунд). Сервер может быть перегружен или обрабатывает слишком сложный запрос.")
            except requests.exceptions.ConnectionError:
                st.error(f"Ошибка: Не удалось подключиться к API по адресу {API_URL}. Убедитесь, что FastAPI сервер запущен и доступен.")
            except requests.exceptions.HTTPError as e:
                st.error(f"Ошибка HTTP от API: {e.response.status_code} {e.response.reason}")
                try:
                    error_details = e.response.json()
                    st.json(error_details) # Показать детали ошибки от API, если они есть
                except json.JSONDecodeError:
                    st.error("Не удалось декодировать JSON из ответа об ошибке API.")
            except json.JSONDecodeError:
                st.error("Ошибка: Не удалось декодировать JSON из ответа API. Ответ API не является валидным JSON.")
                st.text("Полученный ответ (или его часть):")
                st.text(response.text[:500] if response else "Ответ отсутствует") # Показать часть сырого ответа для отладки
            except Exception as e:
                st.error(f"Произошла непредвиденная ошибка: {e}")
                print(f"Непредвиденная ошибка: {type(e)} - {e}") # Логирование в консоль для отладки

    else:
        # --- 7. Обработка пустого вопроса ---
        st.warning("Пожалуйста, введите ваш вопрос в текстовое поле выше.")

# Инструкция по запуску (можно добавить в конец или в комментарий в начале)
# Чтобы запустить это приложение:
# 1. Убедитесь, что ваш FastAPI сервер (main_api.py) запущен.
# 2. В новом терминале, в активированном виртуальном окружении, выполните:
#    streamlit run app_ui.py
#
# Не забудьте установить streamlit: pip install streamlit requests
if 'requests' not in st.__dict__: # Простая проверка, что requests импортирован (для примера)
    pass # Это условие всегда будет ложным, если импорт успешен. Используется для примера.
