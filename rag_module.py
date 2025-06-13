"""rag_module.py

Общий модуль с функциональностью Retrieval‑Augmented Generation (RAG),
который переиспользуется в main_api.py и query_data.py.

Основные публичные объекты:
    * CUSTOM_RAG_PROMPT – шаблон промпта.
    * main_query(question: str, section_filter: Optional[str] = None) – функция
     ‑обёртка для быстрого вызова RAG и получения ответа + источников.
    * RAGService – класс‑обёртка, если нужен stateful объект.

Все тяжёлые зависимости (эмбеддинги, Chroma, LLM) лениво инициализируются
и кэшируются, поэтому создаются ровно один раз за жизнь процесса.

Переменные окружения (см. .env):
    CHROMA_PERSIST_DIR      – директория с Чрома‑базой (по умолчанию db_chroma)
    CHROMA_COLLECTION_NAME  – имя коллекции                       (rag_collection)
    EMBEDDING_MODEL_NAME_HF – HF‑модель эмбеддингов               (all‑MiniLM‑L6-v2)
    USE_OPENAI_EMBEDDINGS   – 'true' | 'false'  (по умолчанию true)
    LLM_PROVIDER            – 'openai'         (позже можно добавить 'huggingface')
    OPENAI_API_KEY          – ключ OpenAI, если используется.
    RAG_DEBUG               – 'true' | 'false'  (включает подробный отладочный вывод)

Автор: ChatGPT‑refactor, 2025‑06‑11.
"""

from __future__ import annotations

from pathlib import Path
import json

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.globals import set_debug

# --------------------------------------------------------------------------- #
# Конфигурация
# --------------------------------------------------------------------------- #

load_dotenv()  # Понимаем .env при первом импорте

CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "db_chroma")
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "rag_collection")
EMBEDDING_MODEL_NAME_HF: str = os.getenv(
    "EMBEDDING_MODEL_NAME_HF", "sentence-transformers/all-MiniLM-L6-v2"
)
USE_OPENAI_EMBEDDINGS: bool = os.getenv("USE_OPENAI_EMBEDDINGS", "true").lower() == "true"
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")

DEBUG_RAG: bool = os.getenv("RAG_DEBUG", "false").lower() == "true"

# Глобальная отладка LangChain (включается переменной окружения LC_DEBUG=true)
if os.getenv("LC_DEBUG", "false").lower() == "true":
    set_debug(True)

# --------------------------------------------------------------------------- #
# Промпт
# --------------------------------------------------------------------------- #

CUSTOM_RAG_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "Ты — бот‑консультант по продуктам 1С. Отвечай **строго** на основе "
            "предоставленного контекста. Если информации нет — честно скажи: "
            "\"Извините, я не могу найти информацию по вашему вопросу в доступных "
            "документах.\" Форматируй ответ в Markdown (заголовки, списки, "
            "**жирный текст**, абзацы) и добавь немного эмпатии."
        ),
        # RetrievalQA подставляет {context} и {question}
        HumanMessagePromptTemplate.from_template("{context}"),
    ]
)

# --------------------------------------------------------------------------- #
# Внутренние ленивые фабрики
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _get_embedding_model():
    """Создаёт (или возвращает из cache) модель эмбеддингов."""
    if USE_OPENAI_EMBEDDINGS and os.getenv("OPENAI_API_KEY"):
        print("🟢  Инициализация OpenAIEmbeddings …")
        return OpenAIEmbeddings()
    print(
        f"🟡  Используются HuggingFaceEmbeddings: {EMBEDDING_MODEL_NAME_HF} "
        "(OpenAI отключён или не задан ключ)."
    )
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME_HF,
        model_kwargs={"device": "cpu"},
    )


@lru_cache(maxsize=1)
def _get_vector_store():
    """Загружает ChromaDB коллекцию."""
    if not os.path.exists(CHROMA_PERSIST_DIR):
        raise FileNotFoundError(
            f"Директория ChromaDB '{CHROMA_PERSIST_DIR}' не найдена. "
            "Сначала проиндексируйте данные (index_data.py)."
        )

    print(
        f"🔵  Загрузка Chroma (collection='{CHROMA_COLLECTION_NAME}') "
        f"из '{CHROMA_PERSIST_DIR}' …"
    )
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=_get_embedding_model(),
        persist_directory=CHROMA_PERSIST_DIR,
    )


@lru_cache(maxsize=1)
def _get_llm():
    """Инициализирует LLM."""
    if LLM_PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError(
                "OPENAI_API_KEY не найден, но LLM_PROVIDER='openai'."
            )
        print("🟢  Инициализация ChatOpenAI (gpt-3.5-turbo)")
        return ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")
    # Заглушка: можно добавить HuggingFace или другие провайдеры
    raise ValueError(f"Неизвестный LLM_PROVIDER '{LLM_PROVIDER}'.")


# --------------------------------------------------------------------------- #
# FAQ helpers
# --------------------------------------------------------------------------- #

def _load_full_faq_answer(section: str, file_name: str, url: str) -> tuple[str | None, str | None]:
    """Return (answer, question) from the local FAQ JSON, if found.

    * `section` – 1C section sub‑folder (e.g., 'УНФ')
    * `file_name` – JSON file inside `data/{section}/faq`
    * `url` – unique url/id of the FAQ record
    """
    faq_path = Path("data") / section / "faq" / file_name
    if not faq_path.exists():
        return None, None
    try:
        rows = json.loads(faq_path.read_text(encoding="utf-8"))
        for row in rows:
            if str(row.get("url", "")).strip() == url.strip():
                return (row.get("answer", "").strip() or None,
                        row.get("question", "").strip() or None)
    except Exception:
        # silently ignore malformed json etc.
        pass
    return None, None

# --------------------------------------------------------------------------- #
# Служебный класс (по желанию)
# --------------------------------------------------------------------------- #


class RAGService:
    """Упрощённый stateful‑сервис. Можно создавать один экземпляр и шарить."""

    def __init__(self) -> None:
        # Ленивая инициализация тяжёлых объектов
        self._embedding_model = _get_embedding_model()
        self._vector_store = _get_vector_store()
        self._llm = _get_llm()

    # -------------------- Публичное API -------------------- #

    def answer(
        self,
        question: str,
        section_filter: Optional[str] = None,
        *,
        include_full_faq: bool = False,
        k: int = 5,
        score_threshold: float = 0.3,
    ) -> Dict[str, Any]:
        """Основная точка входа для получения ответа."""
        # --- 1. Достаём документы сразу со score ---------------------------------
        #   Chroma возвращает (Document, distance). distance ∈ [0,1].
        docs_with_scores = self._vector_store.similarity_search_with_score(
            question,
            k=k,
            filter=None
            if section_filter in (None, "все")
            else {"1c_section": section_filter},
        )

        # 2. Сортируем по distance и фильтруем по порогу
        docs_with_scores.sort(key=lambda t: t[1])  # по возрастанию distance
        filtered_dws = [(d, s) for d, s in docs_with_scores if s <= score_threshold]

        if DEBUG_RAG:
            print(f"\n⚙️  [RAG_DEBUG] Запрос: {question!r}")
            print(f"⚙️  [RAG_DEBUG] Получено документов: {len(docs_with_scores)}")
            for idx, (d, s) in enumerate(filtered_dws, 1):
                print(f"    {idx}. score={s:.4f}  file={d.metadata.get('file_name')}")

        # 3. Если после порога ничего не осталось – берём 1‑й документ для контекста
        if not filtered_dws and docs_with_scores:
            filtered_dws = docs_with_scores[:1]

        # --- 3‑bis. Обогащаем FAQ‑документы полным Q/A ---------------------
        enriched_docs: List[Any] = []
        for d, dist in filtered_dws:
            md = d.metadata
            src_type = md.get("source_type")
            q = md.get("question", "").strip()
            a = md.get("answer", "").strip()

            # --- Подтягиваем полный ответ FAQ при необходимости -----------------
            if (
                include_full_faq
                and src_type in {"faq", "SD 1cfresh"}
                and not a
                and md.get("file_name")
                and md.get("url")
            ):
                a_local, q_local = _load_full_faq_answer(
                    md.get("1c_section") or section_filter or "",
                    md["file_name"],
                    md["url"],
                )
                if a_local:
                    a = a_local
                    md["answer"] = a_local
                    if q_local:
                        q = q_local
            # -------------------------------------------------------------------

            # Формируем page_content
            if src_type in {"faq", "SD 1cfresh"} and a:
                d.page_content = f"Вопрос: {q}\n\nОтвет: {a}"

            enriched_docs.append(d)
        # 4. Готовим документы для LLM
        input_docs = enriched_docs

        # ---------- Формируем промпт без дополнительного retriever ----------
        joined_ctx = "\n\n".join(doc.page_content for doc in input_docs) or "Контекст отсутствует."
        prompt = CUSTOM_RAG_PROMPT.format(context=joined_ctx)

        # ChatOpenAI возвращает ChatMessage; страхуемся на случай строкового вывода
        answer_msg = self._llm.invoke(prompt)
        answer_text: str = answer_msg.content if hasattr(answer_msg, "content") else str(answer_msg)
        # --------------------------------------------------------------------

        sources = []
        for doc, dist in filtered_dws:
            sim_pct = round((1 - dist) * 100, 2)
            md = doc.metadata
            md["raw_score"] = dist
            md["similarity_percent"] = sim_pct
            sources.append(
                {
                    **{k: md.get(k) for k in (
                        "file_name","full_path","source_type","1c_section","page_number",
                        "db_table","record_id","url","question","date","code","subsection"
                    )},
                    "content_snippet": doc.page_content,
                    "raw_score": dist,
                    "similarity_percent": sim_pct,
                    "answer": md.get("answer"),
                }
            )

        if DEBUG_RAG:
            print(f"⚙️  [RAG_DEBUG] После фильтра (<= {score_threshold}): {len(sources)} источников\n")
        return {"answer": answer_text, "sources": sources}


# --------------------------------------------------------------------------- #
# Функция‑обёртка «как было раньше»
# --------------------------------------------------------------------------- #


def main_query(
    question: str,
    section_filter: Optional[str] = None,
    *,
    include_full_faq: bool = False
) -> Dict[str, Any]:
    """Backwards‑совместимый thin‑wrapper вокруг :pyclass:`RAGService`.

    Пример:
        >>> from rag_module import main_query
        >>> response = main_query("Как сформировать счёт‑фактуру?", section_filter="УНФ")
        >>> print(response["answer"])
    """
    _service = RAGService()  # Создаётся ровно один раз благодаря @lru_cache
    return _service.answer(
        question=question,
        section_filter=section_filter,
        include_full_faq=include_full_faq,
    )
