import os
import argparse
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain.llms import HuggingFacePipeline # Для использования локальных моделей HF
from langchain.chains import RetrievalQA # Также можно использовать RetrievalQAWithSourcesChain

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
        return

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
        return

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
        return

    # 4. Инициализация LLM
    try:
        llm = get_llm()
    except ValueError as e:
        print(e)
        return
    except NotImplementedError as e:
        print(e)
        return


    # 5. Создание QA цепочки
    # Использование RetrievalQA и установка return_source_documents=True
    # Для RetrievalQAWithSourcesChain формат вывода немного отличается.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Другие типы: "map_reduce", "refine", "map_rerank"
        retriever=retriever,
        return_source_documents=True, # Важно для получения информации об источниках
        # chain_type_kwargs={"prompt": YOUR_CUSTOM_PROMPT} # Необязательно
    )

    print(f"\nВыполнение запроса: \"{user_query}\"")

    # 6. Выполнение запроса
    try:
        result = qa_chain.invoke({"query": user_query}) # Стандарт Langchain LCEL
    except Exception as e:
        print(f"Ошибка во время выполнения запроса: {e}")
        return

    # 7. Вывод результатов
    print("\n--- Ответ ---")
    print(result.get("result", "Ответ не найден."))

    print("\n--- Источники ---")
    if "source_documents" in result and result["source_documents"]:
        for i, doc in enumerate(result["source_documents"]):
            print(f"\nИсточник {i+1}:")
            metadata = doc.metadata
            print(f"  Фрагмент содержимого: {doc.page_content[:200]}...") # Показать фрагмент
            if "file_name" in metadata:
                print(f"  Имя файла: {metadata.get('file_name')}")
            if "full_path" in metadata:
                print(f"  Полный путь: {metadata.get('full_path')}")
            if "source_type" in metadata:
                print(f"  Тип источника: {metadata.get('source_type')}")
            if "1c_section" in metadata:
                print(f"  Раздел 1С: {metadata.get('1c_section')}")
            if "page_number" in metadata: # Для PDF
                 print(f"  Номер страницы: {metadata.get('page_number')}")
            if "db_table" in metadata: # Для SQLite
                print(f"  Таблица БД: {metadata.get('db_table')}")
                print(f"  ID записи: {metadata.get('record_id')}")
            # Добавьте другие релевантные поля метаданных, которые вы проиндексировали
            print("-" * 20)
    else:
        print("Источники для этого ответа не найдены или не возвращены.")

    print("\nПроцесс выполнения запроса завершен.")


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

    main_query(args.query, args.section)

    print("\n--- Пример использования ---")
    print("python query_data.py \"Как настроить резервное копирование в УНФ?\" --section УНФ")
    print("python query_data.py \"Общие вопросы по бухгалтерии\" --section БП")
    print("python query_data.py \"Что такое ERP системы?\"") # Без фильтра по разделу, поиск по всем
    print("python query_data.py \"Как создать отчет?\" --section все") # Явный поиск по всем разделам
    print("---------------------\n")
