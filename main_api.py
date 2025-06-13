#main_api

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
# Global debug flag (off by default; enable with RAG_DEBUG=true)
DEBUG_RAG = os.getenv("RAG_DEBUG", "false").lower() in ("1", "true", "yes", "y")
import traceback

from functools import lru_cache
from pathlib import Path
import json

from rag_module import main_query

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    section: str | None = None
    include_full_faq: bool = True

BASE_DATA_DIR = Path("data")   # корневая папка с данными

@lru_cache(maxsize=128)
def _load_faq_file(section: str, file_name: str):
    """
    Читаем FAQ‑файл и кешируем содержимое.
    """
    faq_path = BASE_DATA_DIR / section / "faq" / file_name
    if not faq_path.exists():
        raise FileNotFoundError(f"FAQ file not found: {faq_path}")
    with faq_path.open("r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/query/")
async def execute_query(req: QueryRequest):
    try:
        if DEBUG_RAG:
            print(f"[FAQ‑ENRICH DEBUG] query='{req.query}' "
                  f"section='{req.section}' full={req.include_full_faq}")
            print(f"[FAQ‑ENRICH DEBUG] include_full_faq={req.include_full_faq}")

        result = main_query(
            req.query,
            req.section,
            include_full_faq=req.include_full_faq
        )
        # --- sanitize fallback text ---
        fallback_starts = (
            "Извините, я не могу найти",
            "Извините, я не нашёл",
        )
        has_sources = bool(result.get("sources"))
        has_real_answer = bool(result.get("answer") and
                               not any(result["answer"].startswith(p) for p in fallback_starts))

        # если есть источники или «нормальный» ответ, удаляем лишний fallback‑текст
        if has_sources or has_real_answer:
            if result.get("answer") and any(result["answer"].startswith(p) for p in fallback_starts):
                # оставляем только вторую часть ответа (после первого \n\n) или очищаем
                cleaned = result["answer"].split("\n\n", 1)
                result["answer"] = cleaned[1].strip() if len(cleaned) > 1 else ""

        return result
    except Exception as e:
        if DEBUG_RAG:
            traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Новый endpoint для получения полного ответа из FAQ
@app.get("/faq_full_answer/")
async def faq_full_answer(section: str, file: str, url: str):
    """
    Возвращает полный question+answer из FAQ‑json по URL (или UUID).
    """
    try:
        entries = _load_faq_file(section, file)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="FAQ file not found")

    for item in entries:
        # сравним по url без лишних пробелов
        if str(item.get("url", "")).strip() == url.strip():
            return {
                "question": item.get("question", ""),
                "answer": item.get("answer", "")
            }
    raise HTTPException(status_code=404,
                        detail="FAQ entry with specified url not found.")