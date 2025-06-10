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
QA_CHAIN = None # Будет инициализироваться для каждого запроса с нужным ретривером

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
            # Проверка, есть ли документы в коллекции
            # collection_count = VECTORSTORE._collection.count() # Это может быть внутренним API Chroma
            # print(f"Количество документов в коллекции '{CHROMA_COLLECTION_NAME}': {collection_count}")
            # if collection_count == 0:
            #     print("ПРЕДУПРЕЖДЕНИЕ: Коллекция ChromaDB пуста. API будет возвращать пустые ответы.")
            print("Chroma vector store успешно загружен.")
        except Exception as e:
            print(f"Ошибка при загрузке Chroma vector store: {e}")
            VECTORSTORE = None
else:
    print("Модель для эмбеддингов не была инициализирована. Загрузка Vectorstore пропущена.")
    VECTORSTORE = None


# --- 4. Pydantic Модели ---
class QueryRequest(BaseModel):
    query: str
    section: Optional[str] = "все" # По умолчанию "все", если не указано
    # top_k: Optional[int] = 5 # Можно добавить для настройки количества извлекаемых документов

class SourceDocument(BaseModel):
    file_name: Optional[str] = None
    section_1c: Optional[str] = None # Используем snake_case для Pydantic, FastAPI преобразует в camelCase в JSON
    source_type: Optional[str] = None
    full_path: Optional[str] = None
    page_number: Optional[int] = None
    db_table: Optional[str] = None
    record_id: Optional[str] = None
    content_snippet: Optional[str] = None # Добавим фрагмент контента
    url: Optional[str] = None
    date: Optional[str] = None
    question: Optional[str] = None

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
        # Можно добавить более детальное логирование ошибки здесь, если нужно
        # import traceback
        # print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера при обработке запроса: {str(e)}")

@app.get("/")
async def root():
    return {"message": "API для RAG системы по базе знаний 1С. Используйте POST /query/ для отправки запросов."}

# --- 8. Комментарий для запуска Uvicorn ---
# Для запуска этого API используйте команду в терминале (в активированном виртуальном окружении):
# uvicorn main_api:app --reload --port 8000
#
# Убедитесь, что:
# 1. Файл .env существует и содержит OPENAI_API_KEY (если используется OpenAI).
# 2. Директория db_chroma существует и содержит проиндексированные данные (создается скриптом index_data.py).

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
