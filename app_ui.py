#app_ui.py

DEBUG_MODE = False  # Установи True для включения отладочного вывода

import streamlit as st
import requests
import json
import os
import urllib.parse

STATIC_SERVER_HOST = "http://localhost:8600"

# --- 1. Константы ---
# URL вашего FastAPI эндпоинта (убедитесь, что он запущен)
API_URL = "http://127.0.0.1:8000/query/"

# Доступные разделы 1С для фильтрации (должны совпадать с теми, что используются в API и индексации)
# "все" означает отсутствие фильтрации по конкретному разделу
AVAILABLE_SECTIONS = ["все", "УНФ", "БП", "ERP", "Розница", "ЗУП", "Unknown"]

def call_rag_api(payload: dict, timeout: int = 180):
    """Синхронный вызов FastAPI и обработка ошибок в одном месте."""
    try:
        response = requests.post(API_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"HTTP {e.response.status_code}: {e.response.reason}") from e
    except requests.exceptions.Timeout:
        raise RuntimeError("Timeout while calling API") from None
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot connect to API") from None

# --- Основная функция приложения Streamlit ---
def main():
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
            index=0,
            key="section_selector", # <--- ОЧЕНЬ ВАЖНО: ЭТОТ KEY НЕОБХОДИМ ДЛЯ ТЕСТОВ
            help="Фильтрация поиска по конкретному разделу 1С. 'все' - поиск по всем разделам."
        )

        include_full_faq = st.checkbox(
            "Включать полный FAQ‑ответ",
            value=True,
            help="Если включено, API будет присылать полный ответ FAQ вместе с результатами."
        )

        st.markdown("---")
        st.subheader("О приложении")
        st.info(
            "Это приложение использует модель RAG (Retrieval Augmented Generation) "
            "для ответов на вопросы. Ответы генерируются на основе информации, "
            "найденной в загруженной базе знаний."
        )
        st.caption("Разработано с использованием LangChain, FastAPI и Streamlit.")

    # --- Добавляем историю чата ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "question_input" not in st.session_state:
        st.session_state.question_input = ""


    # --- Форма для вопроса ---
    with st.form(key="query_form", clear_on_submit=False):
        question = st.text_area(
            "Введите ваш вопрос здесь:",
            height=150,
            placeholder="Например: Как создать новый отчет в УНФ?",
            key="question_input"
        )
        # --- Кнопки отправки и очистки, выровненные по горизонтали ---
        col_submit, col_clear = st.columns(2)
        with col_submit:
            submitted = st.form_submit_button("🚀 Отправить запрос", use_container_width=True)
        with col_clear:
            clear_requested = st.form_submit_button("🧹 Очистить", use_container_width=True)

    # Отрисовка истории запросов
    if st.session_state.chat_history:
        st.markdown("### 💬 История запросов")
        # Отображаем в обратном порядке, чтобы самые новые были сверху
        for entry in reversed(st.session_state.chat_history): 
            with st.expander(f"Вопрос: {entry['question']}"):
                st.markdown(entry['answer'])
                # Отображение источников для каждого ответа в истории
                if "sources" in entry and entry["sources"]:
                    st.markdown("**Источники для этого вопроса:**")
                    # Эта логика дублируется из основного блока, но может быть полезной для истории
                    seen = set()
                    unique_sources_history = []
                    for source_hist in entry["sources"]:
                        key_hist = (source_hist.get("file_name"), source_hist.get("url"))
                        if key_hist in seen and source_hist.get('source_type') == 'pdf':
                            for us_hist in unique_sources_history:
                                if (us_hist.get('file_name'), us_hist.get('url')) == key_hist:
                                    existing_pages_hist = str(us_hist.get('page_number', ''))
                                    new_page_hist = str(source_hist.get('page_number', ''))
                                    if new_page_hist and new_page_hist not in existing_pages_hist:
                                        us_hist['page_number'] = f"{existing_pages_hist}, {new_page_hist}" if existing_pages_hist else new_page_hist
                                    break
                            continue
                        if key_hist not in seen:
                            seen.add(key_hist)
                            unique_sources_history.append(source_hist)

                    for j, source_hist in enumerate(unique_sources_history):
                        question_from_source_hist = source_hist.get("question")
                        expander_title_hist = ""
                        if source_hist.get('source_type') in {'faq', 'SD 1cfresh'}:
                            expander_title_hist = f"Источник {j+1}: FAQ - {question_from_source_hist or 'Без вопроса'}"
                        elif source_hist.get('file_name'):
                            expander_title_hist = f"Источник {j+1}: {source_hist.get('file_name')}"
                            if source_hist.get('section_1c') and source_hist.get('section_1c') != 'Unknown':
                                expander_title_hist += f" (Раздел: {source_hist.get('section_1c')})"
                        else:
                            expander_title_hist = f"Источник {j+1}: Неизвестный тип"

                        # Вместо вложенного expander выводим ключевую информацию напрямую
                        st.markdown(f"**Вопрос:** {source_hist.get('question', '—')}")
                        full_faq_content_hist = source_hist.get('content_snippet', 'Содержимое FAQ не найдено.')
                        st.markdown("**Ответ:**")
                        st.markdown(f"> {full_faq_content_hist.replace(chr(10), chr(10) + '> ')}")


    # --- 5. Логика обработки запроса и отображения результатов ---

    # --- Обработка очистки, если нажали вторую кнопку формы ---
    if 'clear_requested' in locals() and clear_requested:
        st.session_state.question_input = ""
        st.session_state.chat_history = []
        st.rerun()

    if submitted:
        if question.strip():  # Проверяем, что вопрос не пустой (после удаления пробелов)
            with st.spinner("🔎 Выполняется поиск ответа. Пожалуйста, подождите..."):
                payload = {
                    "query": question,
                    "section": selected_section, # selected_section берется из st.selectbox
                    "include_full_faq": include_full_faq,
                }

                try:
                    print(f"Отправка запроса на API: {API_URL} с данными: {payload}")
                    api_response = call_rag_api(payload)
                    print(f"Ответ от API получен: {api_response}")

                    # --- 6. Отображение результатов ---
                    print("--- Ответ GPT ---")
                    print(api_response.get("answer", "Ответ не был получен от API."))
                    print("------------------")
                    st.markdown(api_response.get("answer", "Ответ не был получен от API."))

                    # Добавляем в историю чата
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": api_response.get("answer", "Ответ не был получен от API."),
                        "sources": api_response.get("sources", []) # Сохраняем источники в истории
                    })
                    
                    sources = api_response.get("sources", [])
                    # Удаляем дубликаты источников по устойчивому ключу
                    seen = set()
                    unique_sources = []
                    for source in sources:
                        key = (
                            source.get("file_name"),
                            source.get("url") # Используем URL для уникальности, даже если это UUID
                        )
                        if DEBUG_MODE:
                            st.text(f"[DEBUG] Ключ источника: {key}")
                        
                        # Дополнительная логика дедупликации для PDF, если есть разные страницы одного файла
                        # В промпте LLM мы просили включать номер страницы.
                        # Здесь можно объединить информацию о страницах, если filename и url совпадают
                        # Если source_type == 'pdf' и key уже виден, то обновить page_number
                        if key in seen and source.get('source_type') == 'pdf':
                            for us in unique_sources:
                                if (us.get('file_name'), us.get('url')) == key:
                                    existing_pages = str(us.get('page_number', ''))
                                    new_page = str(source.get('page_number', ''))
                                    if new_page and new_page not in existing_pages:
                                        us['page_number'] = f"{existing_pages}, {new_page}" if existing_pages else new_page
                                    break
                            continue # Пропускаем добавление, так как обновили существующий
                        
                        if key not in seen:
                            seen.add(key)
                            unique_sources.append(source)

                    sources = unique_sources
                    st.subheader(f"Источники ({len(sources)})")
                    if DEBUG_MODE:
                        st.text(f"[DEBUG] Уникальных источников после фильтрации: {len(sources)} из {len(api_response.get('sources', []))}")
                    
                    if sources:
                        for i, source in enumerate(sources):
                            question_from_source = source.get("question") # Вопрос из FAQ
                            
                            expander_title = ""
                            if source.get('source_type') in {'faq', 'SD 1cfresh'}:
                                expander_title = f"Источник {i+1}: FAQ - {question_from_source or 'Без вопроса'}"
                            elif source.get('file_name'):
                                expander_title = f"Источник {i+1}: {source.get('file_name')}"
                                if source.get('section_1c') and source.get('section_1c') != 'Unknown':
                                    expander_title += f" (Раздел: {source.get('section_1c')})"
                            else:
                                expander_title = f"Источник {i+1}: Неизвестный тип"


                            with st.expander(expander_title):
                                # --- Специальная логика для FAQ-источников ---
                                if source.get('source_type') in {'faq', 'SD 1cfresh'}:
                                    st.markdown("**Полный ответ из Базы Знаний (FAQ):**")
                                    try:
                                        api_url = "http://127.0.0.1:8000/faq_full_answer/"
                                        section_val = source.get("1c_section") or source.get("section_1c") or "Unknown"
                                        params = {
                                            "section": section_val,
                                            "file": source.get("file_name", ""),
                                            "url": source.get("url", "")
                                        }
                                        r = requests.get(api_url, params=params, timeout=5)
                                        r.raise_for_status()
                                        faq_data = r.json()
                                        question_text = faq_data.get("question", "").strip()
                                        answer_text = faq_data.get("answer", "").strip()

                                        st.markdown(f"**Вопрос:** {question_text}")
                                        st.markdown(f"> {answer_text.replace(chr(10), chr(10) + '> ')}")
                                    except Exception as e:
                                        st.warning(f"Не удалось загрузить полный ответ из базы: {e}")
                                    # Для FAQ-источников с внутренним UUID, не выводим кликабельную ссылку здесь.
                                    # Если URL вдруг оказался внешней ссылкой, можно вывести ее.
                                    if source.get('url') and not (source.get('url').replace('-', '').isalnum() and len(source.get('url')) == 36): # Простая проверка на UUID
                                        link_text = question_from_source or "Перейти к источнику (FAQ)"
                                        st.markdown(f"🔗 [{link_text}]({source['url']})", unsafe_allow_html=True)
                                elif source.get('source_type') in {'pdf', 'excel', 'markdown'} and source.get('file_name'):
                                    # --- Логика для локальных файлов (PDF, Excel, Markdown) ---
                                    try:
                                        section = source.get("1c_section") or source.get("section_1c") or "Unknown"
                                        file_name = source.get("file_name", "")
                                        encoded_section = urllib.parse.quote(section)
                                        encoded_file_name = urllib.parse.quote(file_name)
                                        ext = os.path.splitext(file_name)[1].lstrip(".").lower()
                                        web_path = f"{STATIC_SERVER_HOST}/{ext}s/{encoded_section}/{encoded_file_name}"
                                        st.markdown(f"📄 [Открыть файл]({web_path})", unsafe_allow_html=True)
                                    except Exception as e:
                                        if DEBUG_MODE:
                                            st.warning(f"Не удалось сформировать ссылку на файл: {e}")
                                    
                                    # Для этих типов также показываем фрагмент
                                    if source.get('content_snippet'):
                                        st.markdown("**Фрагмент содержимого:**")
                                        full_snippet = source.get("content_snippet") or ""
                                        if len(full_snippet) > 500:
                                            full_snippet = full_snippet[:500] + "..."
                                        st.markdown(f"> _{full_snippet}_")

                                elif source.get("url"): # --- Логика для прочих внешних URL ---
                                    link_text = source.get("question") or source.get("content_snippet", "").strip().split("\n")[0] or "Перейти к источнику"
                                    if len(link_text) > 100: link_text = link_text[:100] + "..."
                                    st.markdown(f"🔗 [{link_text}]({source['url']})", unsafe_allow_html=True)
                                
                                # --- Общие метаданные (отображаются для всех типов) ---
                                if source.get('file_name'): # Показываем всегда, если есть
                                    st.write(f"**Имя файла:** {source.get('file_name')}")
                                if source.get('1c_section') or source.get('section_1c'):
                                    st.write(f"**Раздел 1С:** {source.get('1c_section') or source.get('section_1c')}")
                                if source.get('source_type'):
                                    st.write(f"**Тип источника:** {source.get('source_type')}")
                                if source.get('full_path'): # Отображаем полный путь, если доступен
                                    st.caption(f"Полный путь: {source.get('full_path')}")
                                if source.get('page_number') is not None: # 0 может быть валидным номером страницы
                                    st.write(f"**Номер страницы:** {source.get('page_number')}")
                                if source.get('similarity_percent') is not None:
                                    percent = source['similarity_percent']
                                    st.write(f"**Сходство (cosine similarity):** {percent:.1f}%")
                                    # визуальный индикатор
                                    if percent >= 85:
                                        st.success(f"🔵 {percent:.1f}% — очень релевантно")
                                    elif percent >= 65:
                                        st.info(f"🟡 {percent:.1f}% — возможно полезно")
                                    else:
                                        st.warning(f"🔴 {percent:.1f}% — низкая релевантность")
                                    if DEBUG_MODE and source.get("raw_score") is not None:
                                        st.caption(f"raw_score={source['raw_score']:.4f}")
                                if source.get('db_table'):
                                    st.write(f"**Таблица БД:** {source.get('db_table')}")
                                if source.get('record_id'):
                                    st.write(f"**ID Записи:** {source.get('record_id')}")
                                if source.get('code'): # Для FAQ
                                    st.write(f"**Код:** {source.get('code')}")
                                if source.get('subsection'): # Для FAQ
                                    st.write(f"**Подраздел:** {source.get('subsection')}")
                                if source.get('date'): # Для FAQ
                                    st.write(f"📅 Дата публикации: {source['date']}")
                                
                                if DEBUG_MODE:
                                    st.json(source) # Для полной отладки структуры источника
                    else:
                        st.info("Источники для данного ответа не найдены или не были предоставлены API.")

                except RuntimeError as e:
                    st.error(str(e))
                except json.JSONDecodeError:
                    st.error("Ошибка: Не удалось декодировать JSON из ответа API. Ответ API не является валидным JSON.")
                except Exception as e:
                    st.error(f"Произошла непредвиденная ошибка: {e}")
                    print(f"Непредвиденная ошибка: {type(e)} - {e}") # Логирование в консоль для отладки
            # st.rerun() # Эту строку нужно удалить!
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

# --- Инструкция по запуску (вне функции main) ---
if __name__ == "__main__":
    main() # Запускаем основную функцию приложения
