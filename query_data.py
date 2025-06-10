import os
import argparse
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain.llms import HuggingFacePipeline # Для использования локальных моделей HF
from langchain.chains import RetrievalQA # Также можно использовать RetrievalQAWithSourcesChain
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np # Для операций с массивами, например, сортировки

import sqlite3

# --- Конфигурация ---
load_dotenv()  # Загрузка переменных окружения из файла .env

# Базовый путь к данным (предполагаем, что структура data/database/ существует)
BASE_DATA_DIR = "data"
SQLITE_DB_PATH = os.path.join(BASE_DATA_DIR, "database", "1c_knowledge_base.db")
SQLITE_FTS_TABLE = "documents_fts"

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

from typing import List, Tuple, Dict, Any # Added Dict, Any, List, Tuple
import json # For parsing weights from CLI
from ensemble_logic import ensemble_merge # For ensemble search

# --- TF-IDF Search Function ---

def tfidf_search(query: str, top_n: int = 5) -> List[Tuple[str, float, Dict[str, Any], str]]:
    """
    Выполняет поиск по TF-IDF векторам.
    Загружает сохраненную TF-IDF модель и векторы, преобразует запрос,
    вычисляет косинусное сходство и возвращает top_n результатов
    включая базовые метаданные и плейсхолдер для сниппета.
    """
    print(f"\n--- Запуск TF-IDF поиска для запроса: \"{query}\" (top_n={top_n}) ---")
    tfidf_model_path = os.path.join(CHROMA_PERSIST_DIR, "tfidf_model.joblib")
    tfidf_vectors_path = os.path.join(CHROMA_PERSIST_DIR, "tfidf_vectors_and_ids.joblib")

    if not os.path.exists(tfidf_model_path) or not os.path.exists(tfidf_vectors_path):
        print(f"  Ошибка: Файлы TF-IDF не найдены в '{CHROMA_PERSIST_DIR}'.")
        print("  Пожалуйста, запустите `index_data.py` для их создания.")
        return []

    try:
        print(f"  Загрузка TF-IDF модели из: {tfidf_model_path}")
        vectorizer: TfidfVectorizer = joblib.load(tfidf_model_path)

        print(f"  Загрузка TF-IDF векторов и ID из: {tfidf_vectors_path}")
        vectors_data = joblib.load(tfidf_vectors_path)
        doc_ids = vectors_data['ids']
        doc_vectors = vectors_data['vectors']

        print(f"  Модель и векторы TF-IDF успешно загружены.")
        print(f"  Количество загруженных документов для TF-IDF: {len(doc_ids)}")
        print(f"  Размерность матрицы векторов документов: {doc_vectors.shape}")

    except Exception as e:
        print(f"  Ошибка при загрузке файлов TF-IDF: {e}")
        return []

    try:
        # Преобразование запроса в TF-IDF вектор
        query_vector = vectorizer.transform([query])
        print(f"  Размерность вектора запроса: {query_vector.shape}")

        # Вычисление косинусного сходства
        # cosine_similarity возвращает матрицу, нам нужна первая строка
        similarities = cosine_similarity(query_vector, doc_vectors).flatten()

        # Получение индексов top_n наиболее похожих документов
        # argsort возвращает индексы в порядке возрастания, поэтому берем с конца
        # или используем отрицание для сортировки по убыванию
        top_indices = np.argsort(similarities)[-top_n:][::-1]

        results_rich: List[Tuple[str, float, Dict[str, Any], str]] = []
        print("\n  Результаты TF-IDF поиска (документ, сходство, метаданные, сниппет):")
        for i in top_indices:
            if similarities[i] > 0:  # Отсеиваем документы с нулевым сходством
                doc_id = doc_ids[i]
                score = float(similarities[i]) # Ensure score is float
                # Для TF-IDF, если полные метаданные и сниппеты не хранятся вместе с векторами,
                # возвращаем то, что есть (doc_id) и плейсхолдеры.
                metadata = {'doc_id': doc_id, 'source': 'TF-IDF'}
                snippet = "Snippet not available for TF-IDF result."
                # Если doc_id содержит путь, можно попытаться извлечь имя файла
                if isinstance(doc_id, str):
                    metadata['file_name'] = os.path.basename(doc_id.split('_page_')[0])


                results_rich.append((doc_id, score, metadata, snippet))
                print(f"    - \"{doc_id}\", Сходство: {score:.4f}, Мета: {metadata}")

        if not results_rich:
            print("  Похожих документов по TF-IDF не найдено.")

        return results_rich

    except Exception as e:
        print(f"  Ошибка во время TF-IDF поиска: {e}")
        return []

# --- SQLite FTS Search Function ---

def fts_search(query: str, top_n: int = 5) -> List[Tuple[str, float, Dict[str, Any], str]]:
    """
    Выполняет поиск по SQLite FTS5 индексу.
    Возвращает top_n результатов с их BM25 рангами (преобразованными в положительные числа),
    включая doc_id, контент (как сниппет) и базовые метаданные.
    """
    print(f"\n--- Запуск SQLite FTS поиска для запроса: \"{query}\" (top_n={top_n}) ---")

    if not os.path.exists(SQLITE_DB_PATH):
        print(f"  Ошибка: Файл SQLite DB не найден по пути '{SQLITE_DB_PATH}'.")
        print("  Пожалуйста, запустите `index_data.py` для создания и наполнения БД.")
        return []

    conn = None
    results = []
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()

        # FTS5 MATCH query. 'rank' будет содержать BM25 скор. Извлекаем также контент.
        sql_query = f"""
        SELECT doc_id, rank, content
        FROM {SQLITE_FTS_TABLE}
        WHERE {SQLITE_FTS_TABLE} MATCH ?
        ORDER BY rank
        LIMIT ?;
        """

        print(f"  Выполнение FTS запроса: SELECT doc_id, rank, content FROM {SQLITE_FTS_TABLE} WHERE {SQLITE_FTS_TABLE} MATCH '{query}' ORDER BY rank LIMIT {top_n}")
        cursor.execute(sql_query, (query, top_n))
        raw_results = cursor.fetchall()

        print("\n  Результаты FTS поиска (doc_id, преобразованный ранг, метаданные, сниппет):")
        for doc_id, rank_score, content_str in raw_results:
            processed_score = -float(rank_score)
            metadata = {'doc_id': doc_id, 'source': 'FTS'}
            # Если doc_id содержит путь, можно попытаться извлечь имя файла
            if isinstance(doc_id, str):
                metadata['file_name'] = os.path.basename(doc_id.split('_page_')[0].split('_chunk_')[0])

            snippet = content_str[:200] # Используем начало контента как сниппет
            results.append((doc_id, processed_score, metadata, snippet))
            print(f"    - \"{doc_id}\", Сходство (BM25-based): {processed_score:.4f} (raw rank: {rank_score:.4f}), Мета: {metadata}")

        if not results:
            print("  Похожих документов по FTS не найдено.")

    except sqlite3.Error as e:
        print(f"  Ошибка SQLite FTS поиска: {e}")
        results = [] # Ensure results is empty on error
    except Exception as e_global:
        print(f"  Непредвиденная ошибка во время FTS поиска: {e_global}")
        results = [] # Ensure results is empty on error
    finally:
        if conn:
            conn.close()
            print(f"  Соединение с SQLite ({SQLITE_DB_PATH}) для FTS поиска закрыто.")

    return results

# --- Semantic Search Function (ChromaDB) ---

def semantic_search(query: str, top_n: int = 5, section_filter: str = None) -> List[Tuple[str, float, Dict[str, Any], str]]:
    """
    Выполняет семантический поиск с использованием ChromaDB.
    Возвращает top_n результатов с их преобразованными скорами сходства,
    включая метаданные документа и сниппет контента.
    """
    print(f"\n--- Запуск семантического поиска (ChromaDB) для запроса: \"{query}\" (top_n={top_n}, filter='{section_filter}') ---")

    # 1. Инициализация модели для эмбеддингов
    try:
        embedding_model = get_embedding_model()
    except Exception as e:
        print(f"  Ошибка при инициализации модели эмбеддингов: {e}")
        return []

    # 2. Загрузка векторного хранилища
    if not os.path.exists(CHROMA_PERSIST_DIR):
        print(f"  Ошибка: Каталог для ChromaDB не найден по пути '{CHROMA_PERSIST_DIR}'.")
        print("  Пожалуйста, сначала запустите скрипт `index_data.py`.")
        return []

    print(f"  Загрузка векторного хранилища Chroma из: {CHROMA_PERSIST_DIR}")
    try:
        vector_store = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embedding_model,
            persist_directory=CHROMA_PERSIST_DIR
        )
    except Exception as e:
        print(f"  Ошибка загрузки векторного хранилища Chroma: {e}")
        return []

    # 3. Подготовка фильтра для поиска
    search_filters = {}
    if section_filter and section_filter.lower() != 'все':
        search_filters = {'1c_section': section_filter}
        print(f"  Применен фильтр к семантическому поиску: Раздел 1С = '{section_filter}'")
    else:
        print("  Фильтр по разделу 1С не применен к семантическому поиску.")

    # 4. Выполнение поиска с получением скоров
    try:
        # similarity_search_with_score возвращает список (Document, score)
        # Score здесь - это обычно расстояние (L2, косинусное расстояние и т.д.), где МЕНЬШЕ - лучше.
        print(f"  Выполнение similarity_search_with_score (k={top_n}, filter={search_filters if search_filters else None})")
        found_documents_with_scores = vector_store.similarity_search_with_score(
            query,
            k=top_n,
            filter=search_filters if search_filters else None
        )
    except Exception as e:
        print(f"  Ошибка во время семантического поиска в ChromaDB: {e}")
        return []

    # 5. Обработка результатов
    results_rich: List[Tuple[str, float, Dict[str, Any], str]] = []
    print("\n  Результаты семантического поиска (doc_id, преобразованное сходство, метаданные, сниппет):")
    if not found_documents_with_scores:
        print("  Похожих документов семантически не найдено.")
        return []

    for i, (doc, distance_score) in enumerate(found_documents_with_scores):
        similarity_score = 1 / (1 + distance_score) if distance_score >= 0 else 0

        # Используем существующие метаданные из документа ChromaDB
        metadata = doc.metadata
        snippet = doc.page_content[:200] # Берем начало контента как сниппет

        # doc_id для семантического поиска должен быть консистентен, если это возможно.
        # Chroma документы уже имеют метаданные, включая 'full_path', 'file_name', 'page_number'.
        # Создадим doc_id, который будет максимально похож на используемый в FTS для возможности сопоставления,
        # но добавим маркер '_ssidx_' для указания источника и уникальности в рамках этого вызова.
        base_path = metadata.get('full_path', metadata.get('file_name', f"unknown_semantic_doc_{i}"))
        page_num = metadata.get('page_number', -1)

        # Формируем doc_id похожий на FTS, но с отличительным суффиксом
        # Это не гарантирует глобальную уникальность с FTS ID, если чанки разные,
        # но обеспечивает уникальность в рамках данного набора результатов семантического поиска.
        # Имя файла уже есть в metadata['file_name']
        doc_id = f"{base_path}_page_{page_num}_ssidx_{i}" if page_num != -1 else f"{base_path}_ssidx_{i}"
        metadata['constructed_doc_id'] = doc_id # Сохраняем созданный ID в метаданных для информации

        results_rich.append((doc_id, similarity_score, metadata, snippet))
        print(f"    - \"{doc_id}\", Сходство: {similarity_score:.4f} (raw distance: {distance_score:.4f}), Мета: {metadata}")

    results_rich.sort(key=lambda x: x[1], reverse=True)

    return results_rich

# --- Основная логика запросов (RAG) ---

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
    parser.add_argument(
        "--ensemble",
        action='store_true',
        help="Выполнить ансамблированный поиск (semantic, TF-IDF, FTS)."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default='{"semantic": 0.5, "tfidf": 0.3, "fts": 0.2}',
        help='JSON строка с весами для ансамблирования, например: \'{"semantic": 0.6, "tfidf": 0.2, "fts": 0.2}\'.'
    )
    parser.add_argument(
        "--top_n_individual",
        type=int,
        default=10,
        help="Количество результатов для извлечения каждым индивидуальным методом поиска перед ансамблированием."
    )
    parser.add_argument(
        "--top_k_ensemble",
        type=int,
        default=10,
        help="Количество финальных результатов после ансамблирования."
    )
    parser.add_argument(
        "--filter_after_merge_json",
        type=str,
        default=None,
        help='JSON строка для фильтрации результатов после ансамблирования, например: \'{"1c_section": "УНФ", "source_type": "faq"}\'.'
    )

    args = parser.parse_args()

    if args.ensemble:
        print("--- Выполнение ансамблированного поиска ---")
        print(f"Запрос пользователя: {args.query}")
        print(f"Раздел для семантического поиска: {args.section if args.section else 'все'}")
        print(f"Top N для индивидуальных поисковиков: {args.top_n_individual}")
        print(f"Top K для финального результата: {args.top_k_ensemble}")

        actual_weights = {}
        try:
            actual_weights = json.loads(args.weights)
            print(f"Используемые веса: {actual_weights}")
        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга JSON весов: {e}. Используются веса по умолчанию.")
            actual_weights = {"semantic": 0.5, "tfidf": 0.3, "fts": 0.2}

        # Вызов индивидуальных поисковых функций
        section_query = args.section if args.section and args.section.lower() != 'все' else None

        rich_semantic_results = semantic_search(args.query, top_n=args.top_n_individual, section_filter=section_query)
        rich_tfidf_results = tfidf_search(args.query, top_n=args.top_n_individual)
        rich_fts_results = fts_search(args.query, top_n=args.top_n_individual)

        # Подготовка данных для ensemble_merge
        simple_semantic = [(r[0], r[1]) for r in rich_semantic_results]
        simple_tfidf = [(r[0], r[1]) for r in rich_tfidf_results]
        simple_fts = [(r[0], r[1]) for r in rich_fts_results]

        # Кеширование деталей для вывода
        details_map_cli: Dict[str, Dict[str, Any]] = {}
        all_rich_results_cli = rich_semantic_results + rich_tfidf_results + rich_fts_results
        for doc_id, _, metadata, snippet in all_rich_results_cli:
            if doc_id not in details_map_cli: # Сохраняем первый встреченный
                details_map_cli[doc_id] = {"metadata": metadata, "snippet": snippet}

        # Вызов функции ансамблирования
        merged_results_cli = ensemble_merge( # Renamed to avoid confusion if we filter
            simple_semantic, simple_tfidf, simple_fts,
            weights=actual_weights,
            top_k=args.top_k_ensemble
        )

        final_results_to_print_cli = merged_results_cli
        filtering_applied_cli_msg = ""

        if args.filter_after_merge_json:
            try:
                post_filters_dict = json.loads(args.filter_after_merge_json)
                if isinstance(post_filters_dict, dict) and post_filters_dict:
                    print(f"\nПрименение фильтрации после ансамблирования (CLI): {post_filters_dict}")
                    temp_filtered_results = []
                    for doc_id, aggregated_score in merged_results_cli:
                        details = details_map_cli.get(doc_id)
                        if details:
                            retrieved_metadata = details.get("metadata", {})
                            match = True
                            for filter_key, filter_value in post_filters_dict.items():
                                if str(retrieved_metadata.get(filter_key, '')).strip() != str(filter_value).strip():
                                    match = False
                                    break
                            if match:
                                temp_filtered_results.append((doc_id, aggregated_score))
                    final_results_to_print_cli = temp_filtered_results
                    filtering_applied_cli_msg = " (после фильтрации)"
                    print(f"Количество результатов после CLI фильтрации: {len(final_results_to_print_cli)}")
                else:
                    print("Предупреждение: JSON для фильтрации не является словарем или пуст. Фильтрация не применяется.")
            except json.JSONDecodeError as e:
                print(f"Ошибка парсинга JSON для --filter_after_merge_json: {e}. Фильтрация не применяется.")

        print(f"\n--- Результаты ансамблированного поиска{filtering_applied_cli_msg} ---")
        if not final_results_to_print_cli:
            print(f"Ансамблированный поиск{filtering_applied_cli_msg} не дал результатов.")
        else:
            for i, (doc_id, aggregated_score) in enumerate(final_results_to_print_cli):
                details = details_map_cli.get(doc_id)
                print(f"\n{i+1}. Doc ID: {doc_id}, Aggregated Score: {aggregated_score:.4f}")
                if details:
                    print(f"   Snippet: {details['snippet'][:250]}...")
                    print(f"   Metadata: {details['metadata']}")
                else:
                    print("   (Детали не найдены в кеше)")
        print(f"--- Конец ансамблированного поиска{filtering_applied_cli_msg} ---")

    else:
        print("Запуск процесса RAG для выполнения запроса к данным...")
        print(f"Запрос пользователя: {args.query}")
        if args.section:
            print(f"Фильтр по разделу 1С: {args.section}")
        main_query(args.query, args.section)
        # Старые индивидуальные тесты удалены для чистоты вывода.
        # Если они нужны, их можно раскомментировать или запускать с 특정 параметрами.

    print("\n--- Пример использования RAG ---")
    print("python query_data.py \"Как настроить резервное копирование в УНФ?\" --section УНФ")
    print("\n--- Пример использования Ансамбля ---")
    print("python query_data.py \"Резервное копирование УНФ\" --ensemble --weights '{\"semantic\":0.6,\"tfidf\":0.2,\"fts\":0.2}' --top_n_individual 5 --top_k_ensemble 5")
    print("python query_data.py \"Отчеты в УНФ\" --ensemble --filter_after_merge_json '{\"1c_section\": \"УНФ\", \"source_type\": \"faq\"}'")
    print("python query_data.py \"Как настроить резервное копирование в УНФ?\" --section УНФ") # This line was duplicated, removing one
    print("python query_data.py \"Общие вопросы по бухгалтерии\" --section БП")
    print("python query_data.py \"Что такое ERP системы?\"") # Без фильтра по разделу, поиск по всем
    print("python query_data.py \"Как создать отчет?\" --section все") # Явный поиск по всем разделам
    print("---------------------\n")
