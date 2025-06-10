import os
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from fastapi.staticfiles import StaticFiles

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# --- 1. Загрузка переменных окружения ---
load_dotenv()
print("Переменные окружения загружены.")

# --- 2. Константы (должны совпадать с index_data.py и query_data.py) ---
CHROMA_PERSIST_DIR = "db_chroma"
CHROMA_COLLECTION_NAME = "rag_collection"

# Выбор модели для эмбеддингов
USE_OPENAI_EMBEDDINGS = True  # По умолчанию True, если OPENAI_API_KEY доступен
EMBEDDING_MODEL_NAME_HF = "sentence-transformers/all-MiniLM-L6-v2" # Пример

# Выбор LLM
# В данном API используется только OpenAI LLM для простоты.
# Расширение для поддержки других LLM (например, HuggingFace локально) потребует доп. логики.
LLM_MODEL_NAME_OPENAI = "gpt-3.5-turbo"

print(f"CHROMA_PERSIST_DIR: {CHROMA_PERSIST_DIR}")
print(f"CHROMA_COLLECTION_NAME: {CHROMA_COLLECTION_NAME}")
print(f"USE_OPENAI_EMBEDDINGS: {USE_OPENAI_EMBEDDINGS}")

# --- 3. Глобальная инициализация моделей ---
EMBEDDING_MODEL = None
LLM = None
VECTORSTORE = None
# QA_CHAIN = None # Глобальная переменная QA_CHAIN не используется, т.к. создается в эндпоинте

# Инициализация модели для эмбеддингов
try:
    if USE_OPENAI_EMBEDDINGS:
        if os.getenv("OPENAI_API_KEY"):
            print("Инициализация OpenAI Embeddings...")
            EMBEDDING_MODEL = OpenAIEmbeddings()
        else:
            print("OPENAI_API_KEY не найден. Попытка использовать HuggingFace Embeddings.")
            EMBEDDING_MODEL = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME_HF,
                model_kwargs={'device': 'cpu'}
            )
            USE_OPENAI_EMBEDDINGS = False # Обновляем флаг, если переключились
    else:
        print(f"Инициализация HuggingFace Embeddings: {EMBEDDING_MODEL_NAME_HF}")
        EMBEDDING_MODEL = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME_HF,
            model_kwargs={'device': 'cpu'}
        )
    print("Модель для эмбеддингов успешно инициализирована.")
except Exception as e:
    print(f"Критическая ошибка при инициализации модели для эмбеддингов: {e}")
    EMBEDDING_MODEL = None # Убедимся, что модель не используется, если инициализация не удалась

# Инициализация LLM (только OpenAI в этой версии)
try:
    if os.getenv("OPENAI_API_KEY"):
        print(f"Инициализация OpenAI LLM: {LLM_MODEL_NAME_OPENAI}...")
        LLM = ChatOpenAI(temperature=0.7, model_name=LLM_MODEL_NAME_OPENAI)
        print("LLM успешно инициализирована.")
    else:
        print("OPENAI_API_KEY не найден. LLM не может быть инициализирована.")
        LLM = None
except Exception as e:
    print(f"Критическая ошибка при инициализации LLM: {e}")
    LLM = None

# Загрузка ChromaDB Vectorstore
if EMBEDDING_MODEL: # Продолжаем, только если модель эмбеддингов загружена
    if not os.path.exists(CHROMA_PERSIST_DIR):
        print(f"ПРЕДУПРЕЖДЕНИЕ: Директория ChromaDB '{CHROMA_PERSIST_DIR}' не найдена.")
        print("API может не работать корректно до тех пор, пока данные не будут проиндексированы.")
        VECTORSTORE = None
    else:
        try:
            print(f"Загрузка Chroma vector store из: {CHROMA_PERSIST_DIR}...")
            VECTORSTORE = Chroma(
                collection_name=CHROMA_COLLECTION_NAME,
                embedding_function=EMBEDDING_MODEL,
                persist_directory=CHROMA_PERSIST_DIR
            )
            # collection_count = VECTORSTORE._collection.count() # Example check, can be removed
            # print(f"Collection '{CHROMA_COLLECTION_NAME}' document count: {collection_count}")
            print("Chroma vector store successfully loaded.")
        except Exception as e:
            print(f"Ошибка при загрузке Chroma vector store: {e}")
            VECTORSTORE = None
else:
    print("Модель для эмбеддингов не была инициализирована. Загрузка Vectorstore пропущена.")
    VECTORSTORE = None


from query_data import semantic_search, tfidf_search, fts_search # Added
from ensemble_logic import ensemble_merge # Added
# Ensure Tuple is imported if not already (it should be part of List, Optional, Dict, Any)
from typing import List, Optional, Dict, Any, Tuple # Added Tuple explicitly for clarity

# --- 4. Pydantic Модели ---
class QueryRequest(BaseModel):
    query: str
    section: Optional[str] = "все" # По умолчанию "все", если не указано
    # top_k: Optional[int] = 5 # Можно добавить для настройки количества извлекаемых документов

class EnsembleQueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10 # Number of final results to return from the ensemble
    top_n_individual: Optional[int] = 10 # Number of results to fetch from each individual search method
    weights: Optional[Dict[str, float]] = None # Weights for 'semantic', 'tfidf', 'fts'
    section: Optional[str] = "все" # Section filter for semantic search (applied before merge)
    filter_after_merge: Optional[Dict[str, str]] = None # Filters to apply on metadata after merging

class SourceDocument(BaseModel):
    # Existing fields
    file_name: Optional[str] = None
    section_1c: Optional[str] = None
    source_type: Optional[str] = None
    full_path: Optional[str] = None
    page_number: Optional[int] = None
    db_table: Optional[str] = None
    record_id: Optional[str] = None
    content_snippet: Optional[str] = None
    url: Optional[str] = None
    date: Optional[str] = None
    question: Optional[str] = None

    # New fields for ensemble results
    ensemble_score: Optional[float] = None
    retrieval_source: Optional[str] = None # To indicate original source type like 'semantic', 'TF-IDF', 'FTS' if needed

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]


# --- 5. Экземпляр FastAPI ---
app = FastAPI(
    title="API для RAG системы по базе знаний 1С",
    description="Этот API позволяет выполнять запросы к проиндексированной базе знаний 1С.",
    version="1.0.0"
)

# Делает папку data/ доступной как /data/*
if os.path.isdir("data"):
    app.mount("/data", StaticFiles(directory="data"), name="data")

# --- 6. CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешить все источники (для разработки)
    allow_credentials=True,
    allow_methods=["*"],  # Разрешить все методы
    allow_headers=["*"],  # Разрешить все заголовки
)
print("CORS middleware добавлен.")

# --- 7. Endpoint /query/ (POST) ---
@app.post("/query/", response_model=QueryResponse)
async def execute_query(request: QueryRequest):
    print(f"\nПолучен запрос на /query/: {request.dict()}")

    if not EMBEDDING_MODEL:
        print("Ошибка API: Модель для эмбеддингов не инициализирована.")
        raise HTTPException(status_code=500, detail="Ошибка сервера: модель для эмбеддингов не сконфигурирована.")
    if not LLM:
        print("Ошибка API: LLM не инициализирована.")
        raise HTTPException(status_code=500, detail="Ошибка сервера: LLM не сконфигурирована.")
    if not VECTORSTORE:
        print("Ошибка API: Vectorstore не доступен.")
        raise HTTPException(status_code=500, detail="Ошибка сервера: база данных векторов не доступна или не проиндексирована.")

    try:
        # Настройка ретривера на основе фильтра по секции
        retriever_search_kwargs = {}
        if request.section and request.section.lower() != 'все':
            retriever_search_kwargs = {'filter': {'1c_section': request.section}}
            print(f"Применяется фильтр для ретривера: {retriever_search_kwargs}")
        else:
            print("Фильтр по секции не применяется.")

        retriever = VECTORSTORE.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5, **retriever_search_kwargs} # Извлекаем топ-5 документов
        )
        print("Ретривер успешно создан.")

        # Создание QA цепочки
        # Примечание: QA_CHAIN теперь создается для каждого запроса, чтобы учесть динамический retriever
        current_qa_chain = RetrievalQA.from_chain_type(
            llm=LLM,
            chain_type="stuff", # или "map_reduce", "refine"
            retriever=retriever,
            return_source_documents=True
        )
        print("QA цепочка успешно создана.")

        # Выполнение запроса
        print(f"Выполнение RAG запроса: \"{request.query}\"")
        rag_result = await current_qa_chain.ainvoke({"query": request.query}) # Используем асинхронный вызов
        print("RAG запрос успешно выполнен.")

        answer = rag_result.get("result", "Ответ не найден.")

        sources_output: List[SourceDocument] = []
        if "source_documents" in rag_result and rag_result["source_documents"]:
            print(f"Найдено {len(rag_result['source_documents'])} источников.")
            for doc in rag_result["source_documents"]:
                metadata = doc.metadata
                question = metadata.get("question")
                # Если question пустой, пытаемся извлечь из page_content
                if not question and doc.page_content.startswith("Вопрос:"):
                    question = doc.page_content.split("\n")[0].replace("Вопрос:", "").strip()

                source_doc = SourceDocument(
                    file_name=metadata.get("file_name"),
                    section_1c=metadata.get("1c_section"),
                    source_type=metadata.get("source_type"),
                    full_path=metadata.get("full_path"),
                    page_number=metadata.get("page_number"),
                    db_table=metadata.get("db_table"),
                    record_id=metadata.get("record_id"),
                    content_snippet=(doc.page_content[:500] + "…") if len(doc.page_content) > 500 else doc.page_content,
                    question=question,
                    url=metadata.get("url"),
                    date=metadata.get("date"),
                )
                sources_output.append(source_doc)
        else:
            print("Источники не найдены.")

        response_data = QueryResponse(answer=answer, sources=sources_output)
        print(f"Ответ API: {response_data.dict(exclude_none=True)}") # Логируем без None полей для краткости
        return response_data

    except Exception as e:
        print(f"Ошибка при обработке запроса: {e}")
        # import traceback # For detailed debugging if needed
        # print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера при обработке запроса: {str(e)}")


# --- 8. Endpoint /ensemble_query/ (POST) ---
@app.post("/ensemble_query/", response_model=QueryResponse)
async def execute_ensemble_query(request: EnsembleQueryRequest):
    print(f"\nПолучен запрос на /ensemble_query/: {request.dict()}")

    # Проверка наличия основных компонентов (аналогично /query/)
    if not EMBEDDING_MODEL: # Needed for semantic search part of ensemble
        print("Ошибка API: Модель для эмбеддингов не инициализирована.")
        raise HTTPException(status_code=500, detail="Ошибка сервера: модель для эмбеддингов не сконфигурирована.")
    # LLM не используется напрямую в этом эндпоинте, только ретриверы

    actual_weights = request.weights
    if actual_weights is None:
        actual_weights = {"semantic": 0.5, "tfidf": 0.3, "fts": 0.2} # Default weights
    print(f"Используемые веса для ансамблирования: {actual_weights}")

    try:
        # 1. Вызов индивидуальных функций поиска
        # Запускаем синхронные функции в ThreadPoolExecutor, чтобы не блокировать event loop FastAPI
        print(f"Запрос к semantic_search (top_n={request.top_n_individual}, section='{request.section}')...")
        rich_semantic_results: List[Tuple[str, float, Dict, str]] = await app.loop.run_in_executor(
            None, semantic_search, request.query, request.top_n_individual, request.section
        )
        print(f"Получено {len(rich_semantic_results)} результатов от semantic_search.")

        print(f"Запрос к tfidf_search (top_n={request.top_n_individual})...")
        rich_tfidf_results: List[Tuple[str, float, Dict, str]] = await app.loop.run_in_executor(
            None, tfidf_search, request.query, request.top_n_individual
        )
        print(f"Получено {len(rich_tfidf_results)} результатов от tfidf_search.")

        print(f"Запрос к fts_search (top_n={request.top_n_individual})...")
        rich_fts_results: List[Tuple[str, float, Dict, str]] = await app.loop.run_in_executor(
            None, fts_search, request.query, request.top_n_individual
        )
        print(f"Получено {len(rich_fts_results)} результатов от fts_search.")

        # 2. Кеширование "богатых" результатов для последующего извлечения деталей
        details_map: Dict[str, Dict[str, Any]] = {}
        all_rich_results = rich_semantic_results + rich_tfidf_results + rich_fts_results

        for res_tuple in all_rich_results:
            if len(res_tuple) == 4: # Ожидаем (doc_id, score, metadata, snippet)
                doc_id, original_score, metadata, snippet = res_tuple
                if doc_id not in details_map: # Сохраняем первый встреченный (можно улучшить стратегию)
                    details_map[doc_id] = {
                        "metadata": metadata,
                        "snippet": snippet,
                        "original_score": original_score, # Может быть полезно для отладки
                        # Определяем источник для отладки, если он есть в метаданных
                        "source_type_from_meta": metadata.get("source", "unknown")
                    }
            else:
                print(f"Предупреждение: Некорректный формат кортежа в all_rich_results: {res_tuple}")


        print(f"Создана карта деталей (details_map) с {len(details_map)} уникальными doc_id.")

        # 3. Подготовка упрощенных результатов для ensemble_merge
        simple_semantic = [(r[0], r[1]) for r in rich_semantic_results if len(r) == 4]
        simple_tfidf = [(r[0], r[1]) for r in rich_tfidf_results if len(r) == 4]
        simple_fts = [(r[0], r[1]) for r in rich_fts_results if len(r) == 4]

        # 4. Вызов ensemble_merge
        print("Вызов ensemble_merge...")
        # ensemble_merge может быть CPU-bound, если списки большие, тоже можно в executor
        merged_results: List[Tuple[str, float]] = await app.loop.run_in_executor(
            None, ensemble_merge, simple_semantic, simple_tfidf, simple_fts, actual_weights, request.top_k
        )
        print(f"Получено {len(merged_results)} результатов после ансамблирования.")

        # 4.5 Фильтрация после ансамблирования (если заданы критерии)
        final_results_to_process: List[Tuple[str, float]] = []
        filtering_applied_message = ""

        if request.filter_after_merge and request.filter_after_merge.items():
            print(f"Применение фильтрации после ансамблирования: {request.filter_after_merge}")
            for doc_id, aggregated_score in merged_results:
                details = details_map.get(doc_id)
                if details:
                    retrieved_metadata = details.get("metadata", {})
                    match = True
                    for filter_key, filter_value in request.filter_after_merge.items():
                        if str(retrieved_metadata.get(filter_key, '')).strip() != str(filter_value).strip():
                            match = False
                            break
                    if match:
                        final_results_to_process.append((doc_id, aggregated_score))
            print(f"Количество результатов после фильтрации: {len(final_results_to_process)}")
            filtering_applied_message = " и последующей фильтрации"
        else:
            final_results_to_process = merged_results
            print("Фильтрация после ансамблирования не применялась.")

        # 5. Формирование ответа
        output_sources: List[SourceDocument] = []
        for doc_id, aggregated_score in final_results_to_process: # Используем отфильтрованные результаты
            details = details_map.get(doc_id)
            if details:
                retrieved_metadata = details.get("metadata", {})
                # Добавляем агрегированный балл и исходный балл (если нужно) в метаданные
                retrieved_metadata['ensemble_score'] = aggregated_score
                # retrieved_metadata['original_retrieval_score'] = details.get("original_score")
                # retrieved_metadata['retrieval_source_type'] = details.get("source_type_from_meta")


                # Создаем SourceDocument, как в /query/ эндпоинте
                question = retrieved_metadata.get("question")
                content_snippet = details.get("snippet", "")
                if not question and content_snippet.startswith("Вопрос:"):
                     question = content_snippet.split("\n")[0].replace("Вопрос:", "").strip()


                source_doc = SourceDocument(
                    file_name=retrieved_metadata.get("file_name"),
                    section_1c=retrieved_metadata.get("1c_section"),
                    source_type=retrieved_metadata.get("source_type", details.get("source_type_from_meta")), # Приоритет metadata, потом из details_map
                    full_path=retrieved_metadata.get("full_path"),
                    page_number=retrieved_metadata.get("page_number"),
                    db_table=retrieved_metadata.get("db_table"), # Если есть
                    record_id=retrieved_metadata.get("record_id"), # Если есть
                    content_snippet=(content_snippet[:500] + "…") if len(content_snippet) > 500 else content_snippet,
                    question=question,
                    url=retrieved_metadata.get("url"),
                    date=retrieved_metadata.get("date"),
                    # Добавляем кастомные поля в метаданные, если SourceDocument их не поддерживает напрямую
                    # ensemble_score=aggregated_score # Это поле нужно добавить в SourceDocument или передавать иначе
                )
                # Пока что ensemble_score не является частью SourceDocument, его можно добавить в content_snippet или metadata, если нужно его видеть в ответе.
                # Для структурированного ответа лучше расширить SourceDocument или создать новую модель ответа.
                # В данном случае, если 'ensemble_score' добавлено в retrieved_metadata, оно будет частью "сырых" метаданных.
                # Если нужно его явно в ответе, то SourceDocument должен иметь поле ensemble_score: Optional[float] = None
                # и тогда source_doc.ensemble_score = aggregated_score

                # Чтобы оценка была видна, временно добавим ее в начало сниппета.
                # Это не идеальное решение, лучше модифицировать SourceDocument.
                # source_doc.content_snippet = f"[Score: {aggregated_score:.4f}] {source_doc.content_snippet}"
                # Вместо этого добавляем в метаданные:
                # retrieved_metadata['ensemble_score'] = aggregated_score # No longer needed if fields exist
                # retrieved_metadata['original_retrieval_score'] = details.get("original_score")
                # retrieved_metadata['retrieval_source_type'] = details.get("source_type_from_meta")

                source_doc = SourceDocument(
                    file_name=retrieved_metadata.get("file_name"),
                    section_1c=retrieved_metadata.get("1c_section"),
                    source_type=retrieved_metadata.get("source_type", details.get("source_type_from_meta")),
                    full_path=retrieved_metadata.get("full_path"),
                    page_number=retrieved_metadata.get("page_number"),
                    db_table=retrieved_metadata.get("db_table"),
                    record_id=retrieved_metadata.get("record_id"),
                    content_snippet=(content_snippet[:500] + "…") if len(content_snippet) > 500 else content_snippet,
                    question=question,
                    url=retrieved_metadata.get("url"),
                    date=retrieved_metadata.get("date"),
                    ensemble_score=aggregated_score, # Assign to the new field
                    retrieval_source=details.get("source_type_from_meta") # Assign to the new field
                )
                output_sources.append(source_doc)
            else:
                print(f"Предупреждение: Детали для doc_id '{doc_id}' не найдены в details_map.")

        answer = f"Ensemble search completed. Found {len(output_sources)} relevant documents after merging{filtering_applied_message}."
        response_data = QueryResponse(answer=answer, sources=output_sources)
        print(f"Ответ API (ансамбль): {response_data.dict(exclude_none=True)}")
        return response_data

    except HTTPException as http_exc:
        # Перебрасываем HTTPException дальше
        raise http_exc
    except Exception as e:
        print(f"Критическая ошибка при обработке запроса ансамбля: {e}")
        # import traceback # For detailed debugging if needed
        # print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера при обработке запроса ансамбля: {str(e)}")


@app.get("/")
async def root():
    return {"message": "API для RAG системы по базе знаний 1С. Используйте POST /query/ или POST /ensemble_query/ для отправки запросов."}

# --- 9. Комментарий для запуска Uvicorn ---
# Для запуска этого API используйте команду в терминале (в активированном виртуальном окружении):
# uvicorn main_api:app --reload --port 8000
#
# Убедитесь, что:
# 1. Файл .env существует и содержит OPENAI_API_KEY (если используется OpenAI).
# 2. Директория db_chroma существует и содержит проиндексированные данные.
# 3. Файл базы данных SQLite (1c_knowledge_base.db) существует и содержит FTS-таблицу.
# 4. Файлы TF-IDF модели и векторов существуют в db_chroma.

if __name__ == "__main__":
    print("Для запуска API, пожалуйста, используйте Uvicorn:")
    print("uvicorn main_api:app --reload --port 8000")
    # Можно добавить базовую проверку доступности моделей при запуске файла напрямую,
    # но основной запуск должен быть через Uvicorn.
    if not EMBEDDING_MODEL:
         print("ПРЕДУПРЕЖДЕНИЕ: Модель для эмбеддингов не была успешно инициализирована при старте.")
    if not LLM:
         print("ПРЕДУПРЕЖДЕНИЕ: LLM не была успешно инициализирована при старте.")
    if not VECTORSTORE:
         print("ПРЕДУПРЕЖДЕНИЕ: Vectorstore не был успешно загружен или не найден.")
