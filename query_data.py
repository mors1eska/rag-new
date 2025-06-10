import os
import argparse
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain.llms import HuggingFacePipeline # Для использования локальных моделей HF
from langchain.chains import RetrievalQA # Также можно использовать RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate # Импортируем PromptTemplate

# --- Конфигурация ---
load_dotenv()  # Загрузка переменных окружения из файла .env

# Конфигурация ChromaDB (должна совпадать с index_data.py)
CHROMA_PERSIST_DIR = "db_chroma"
CHROMA_COLLECTION_NAME = "rag_collection"

# Конфигурация модели для эмбеддингов (должна совпадать с index_data.py)
EMBEDDING_MODEL_NAME_HF = "sentence-transformers/all-MiniLM-L6-v2" # Пример
USE_OPENAI_EMBEDDINGS = True  # Установите False для использования HuggingFaceEmbeddings

# Конфигурация LLM
# Вариант 1: OpenAI LLM (требует OPENAI_API_KEY)
LLM_PROVIDER = "openai"  # или "huggingface"
# Вариант 2: HuggingFace LLM (пример, замените на конкретную модель)
# HF_MODEL_ID = "google/flan-t5-large" # Пример модели, обученной на инструкциях

# Известные разделы 1С (для текста справки, здесь не строгая валидация)
KNOWN_1C_SECTIONS = ["УНФ", "БП", "ERP", "Розница", "ЗУП", "Unknown", "все"]

# --- ОПРЕДЕЛЕНИЕ ПОЛЬЗОВАТЕЛЬСКОГО ПРОМПТА ---
# Этот промпт инструктирует LLM, как форматировать ответ и включать ссылки.
custom_prompt_template = """Используйте предоставленные фрагменты контекста, чтобы ответить на вопрос пользователя.
Ответ должен быть полным, лаконичным, четким и написан на русском языке.
Форматируйте ответ, используя синтаксис Markdown для улучшения читаемости:
- Разделяйте текст на абзацы (используйте двойной перенос строки `\n\n`).
- Используйте заголовки (например, `## Подзаголовок`) для структурирования информации.
- Выделяйте ключевые термины **жирным шрифтом**.
- Используйте списки (`- ` или `1. `), если это уместно для перечисления пунктов.

КРАЙНЕ ВАЖНО: Включите ссылки на источники непосредственно в текст ответа, в конце предложений или абзацев, где информация была взята из контекста. Для каждого источника используйте формат `[Источник: <Имя_файла>/<Номер_страницы>]` или `[Источник: <Имя_файла>]`, если страницы нет. Если источник FAQ, используйте `[Источник: FAQ - <Вопрос_из_FAQ>]`. Старайтесь ссылаться на максимально точный источник.

Если вы не можете найти ответ в предоставленных фрагментах контекста, просто ответьте: "Извините, я не могу найти информацию по вашему вопросу в доступных документах." Не пытайтесь выдумывать ответ или использовать свои общие знания.

Контекст:
{context}

Вопрос: {question}

Ответ:
"""
CUSTOM_RAG_PROMPT = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])


# --- Вспомогательные функции ---

def get_embedding_model():
    """Инициализирует и возвращает выбранную модель для эмбеддингов."""
    if USE_OPENAI_EMBEDDINGS:
        if os.getenv("OPENAI_API_KEY"):
            print("Используются OpenAI Embeddings для ретривала.")
            return OpenAIEmbeddings()
        else:
            print("OPENAI_API_KEY не найден для эмбеддингов, используется HuggingFace.")
            print(f"Используются HuggingFace Embeddings: {EMBEDDING_MODEL_NAME_HF}")
            return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME_HF,
                                       model_kwargs={'device': 'cpu'})
    else:
        print(f"Используются HuggingFace Embeddings для ретривала: {EMBEDDING_MODEL_NAME_HF}")
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME_HF,
                                   model_kwargs={'device': 'cpu'})

def get_llm():
    """Инициализирует и возвращает выбранную LLM."""
    if LLM_PROVIDER == "openai":
        if os.getenv("OPENAI_API_KEY"):
            print("Используется OpenAI LLM (ChatOpenAI gpt-3.5-turbo).")
            return ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
        else:
            print("OPENAI_API_KEY не найден. Невозможно инициализировать OpenAI LLM.")
            raise ValueError("OPENAI_API_KEY требуется для OpenAI LLM.")
    # elif LLM_PROVIDER == "huggingface":
    #     print(f"Используется HuggingFace LLM: {HF_MODEL_ID}")
    #     # Это требует `pip install transformers` и, возможно, дополнительной настройки
    #     # return HuggingFacePipeline.from_model_id(
    #     #     model_id=HF_MODEL_ID,
    #     #     task="text2text-generation", # или "text-generation"
    #     #     model_kwargs={"temperature": 0.7, "max_length": 500}
    #     # )
    #     print("Провайдер HuggingFace LLM - заглушка, реализуйте по мере необходимости.")
    #     raise NotImplementedError("Провайдер HuggingFace LLM требует полной реализации.")
    else:
        raise ValueError(f"Неподдерживаемый провайдер LLM: {LLM_PROVIDER}")

# --- Основная логика запросов ---

def main_query(user_query: str, section_filter: str = None):
    """
    Загружает векторное хранилище, извлекает релевантные документы и генерирует ответ.
    """
    print("\nИнициализация компонентов для выполнения запроса...")

    # 1. Инициализация модели для эмбеддингов
    embedding_model = get_embedding_model()

    # 2. Загрузка векторного хранилища
    if not os.path.exists(CHROMA_PERSIST_DIR):
        print(f"Ошибка: Каталог для ChromaDB не найден по пути '{CHROMA_PERSIST_DIR}'.")
        print("Пожалуйста, сначала запустите скрипт `index_data.py` для создания и наполнения базы данных.")
        return {'answer': "Ошибка: База данных знаний не найдена. Пожалуйста, запустите индексацию данных.", 'sources': []}


    print(f"Загрузка векторного хранилища Chroma из: {CHROMA_PERSIST_DIR}")
    try:
        vector_store = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embedding_model,
            persist_directory=CHROMA_PERSIST_DIR
        )
        # Тест: есть ли документы в коллекции (необязательно)
        # print(f"Количество документов в коллекции: {vector_store._collection.count()}")
    except Exception as e:
        print(f"Ошибка загрузки векторного хранилища Chroma: {e}")
        print("Убедитесь, что имя коллекции и функция для эмбеддингов совпадают с использованными при индексации.")
        return {'answer': f"Ошибка загрузки базы данных: {e}", 'sources': []}


    # 3. Создание ретривера с опциональной фильтрацией
    retriever_search_kwargs = {}
    if section_filter and section_filter.lower() != 'все':
        retriever_search_kwargs = {'filter': {'1c_section': section_filter}}
        print(f"Применен фильтр: Раздел 1С = '{section_filter}'")
    else:
        print("Фильтр по разделу 1С не применен (поиск по всем разделам).")

    try:
        retriever = vector_store.as_retriever(
            search_type="similarity",  # или "mmr"
            search_kwargs={"k": 5, **retriever_search_kwargs} # Извлечь топ-5, применить фильтр
        )
    except Exception as e:
        print(f"Ошибка создания ретривера: {e}")
        return {'answer': f"Ошибка создания ретривера: {e}", 'sources': []}


    # 4. Инициализация LLM
    try:
        llm = get_llm()
    except ValueError as e:
        print(e)
        return {'answer': f"Ошибка инициализации LLM: {e}", 'sources': []}
    except NotImplementedError as e:
        print(e)
        return {'answer': f"Ошибка инициализации LLM: {e}", 'sources': []}


    # 5. Создание QA цепочки
    # Использование RetrievalQA и установка return_source_documents=True
    # Передаем наш пользовательский промпт
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": CUSTOM_RAG_PROMPT} 
    )

    print(f"\nВыполнение запроса: \"{user_query}\"")

    # 6. Выполнение запроса
    try:
        result = qa_chain.invoke({"query": user_query})
    except Exception as e:
        print(f"Ошибка во время выполнения запроса: {e}")
        return {'answer': f"Ошибка во время выполнения запроса: {e}", 'sources': []}

    # 7. Подготовка результатов для возврата (например, для API)
    answer = result.get("result", "Ответ не найден.")
    sources = []
    if "source_documents" in result and result["source_documents"]:
        for i, doc in enumerate(result["source_documents"]):
            metadata = doc.metadata
            source_info = {
                "content_snippet": doc.page_content, # Отправляем весь контент, UI обрежет
                "file_name": metadata.get('file_name'),
                "full_path": metadata.get('full_path'),
                "source_type": metadata.get('source_type'),
                "section_1c": metadata.get('1c_section'),
                "page_number": metadata.get('page_number'), # Для PDF
                "db_table": metadata.get('db_table'), # Для SQLite
                "record_id": metadata.get('record_id'), # Для SQLite
                "url": metadata.get('url'), # Для FAQ/web
                "question": metadata.get('question'), # Для FAQ
                "date": metadata.get('date'), # Для FAQ
                "code": metadata.get('code'), # Для FAQ
                "subsection": metadata.get('subsection') # Для FAQ
                # Добавьте другие релевантные поля метаданных
            }
            sources.append(source_info)
    
    # Также выведем в консоль для отладки (как было)
    print("\n--- Ответ ---")
    print(answer)
    print("\n--- Источники (для консоли) ---")
    if sources:
        for i, source in enumerate(sources):
            print(f"\nИсточник {i+1}:")
            print(f"  Фрагмент содержимого: {source['content_snippet'][:200]}...")
            for key, value in source.items():
                if key != "content_snippet" and value is not None:
                    print(f"  {key.replace('_', ' ').capitalize()}: {value}")
            print("-" * 20)
    else:
        print("Источники для этого ответа не найдены или не возвращены.")
    print("\nПроцесс выполнения запроса завершен.")

    return {'answer': answer, 'sources': sources}


# --- Интерфейс командной строки ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Запрос к проиндексированным данным с использованием RAG и LangChain.",
        formatter_class=argparse.RawTextHelpFormatter # Для корректного отображения многострочных help-сообщений
    )
    parser.add_argument(
        "query",
        type=str,
        help="Вопрос, который вы хотите задать."
    )
    parser.add_argument(
        "--section",
        type=str,
        default="все", # По умолчанию все разделы
        choices=KNOWN_1C_SECTIONS, # Предоставить варианты для лучшего UX
        help=(
            "Необязательно: Фильтр по конкретному разделу 1С.\n"
            "Доступные разделы (чувствительны к регистру, определены в скрипте): " + ", ".join(KNOWN_1C_SECTIONS[:-1]) + "\n"
            "Используйте 'все' для поиска по всем разделам (по умолчанию)."
        )
    )

    args = parser.parse_args()

    print("Запуск процесса RAG для выполнения запроса к данным...")
    print(f"Запрос пользователя: {args.query}")
    if args.section:
        print(f"Фильтр по разделу 1С: {args.section}")

    # Изменено: теперь main_query возвращает словарь, который можно использовать
    response_data = main_query(args.query, args.section)

    print("\n--- Пример использования ---")
    print("python query_data.py \"Как настроить резервное копирование в УНФ?\" --section УНФ")
    print("python query_data.py \"Общие вопросы по бухгалтерии\" --section БП")
    print("python query_data.py \"Что такое ERP системы?\"") # Без фильтра по разделу, поиск по всем
    print("python query_data.py \"Как создать отчет?\" --section все") # Явный поиск по всем разделам
    print("---------------------\n")
