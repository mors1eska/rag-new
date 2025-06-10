def convert_excel_to_faq_format(xlsx_path: str, mapping_path: str = None):
    import pandas as pd
    from pathlib import Path
    import os
    import json
    from collections import defaultdict
    df = pd.read_excel(xlsx_path)
    import json
    mapping_data = {}
    if mapping_path and os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)

    mapping_entry = mapping_data.get(os.path.basename(xlsx_path))
    if not mapping_entry:
        print(f"❌ Нет маппинга для {xlsx_path} в {mapping_path}")
        return

    source_type_label = mapping_entry.get("source_type", "unknown")
    program_map = mapping_entry.get("program_map", {})
    required_columns = ["Наименование", "Номер", "Ссылка", "Описание", "Компоненты", "Разделы"]
    for col in required_columns:
        if col not in df.columns:
            print(f"❌ Отсутствует колонка: {col}")
            return

    grouped_data = defaultdict(list)
    for _, row in df.iterrows():
        name = str(row["Наименование"]).strip()
        answer = str(row["Описание"]).strip()
        if not name or not answer:
            continue
        raw_components = str(row["Компоненты"]).split("/")
        for comp in raw_components:
            comp = comp.strip()
            folder = program_map.get(comp)
            if not folder:
                continue
            target_dir = os.path.join(BASE_DATA_DIR, folder, "faq")
            os.makedirs(target_dir, exist_ok=True)
            output_path = os.path.join(target_dir, f"from_excel_SD_1cfresh.json")
            grouped_data[output_path].append({
                "question": name,
                "answer": answer,
                "url": str(row["Ссылка"]).strip(),
                "source_type": source_type_label,
                "code": str(row["Номер"]).strip(),
                "subsection": str(row["Разделы"]).strip()
            })

    for path, records in grouped_data.items():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"✅ Сохранено {len(records)} записей в {path}")
import os
import glob
import json
from collections import defaultdict
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
# from langchain_community.document_loaders.sql_database import SQLDatabaseLoader # Пример, может потребовать кастомной обработки
# from langchain_community.utilities.sql_database import SQLDatabase # Пример
from langchain.schema import Document # Для создания документов из SQLite данных
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import sqlite3

# Маппинг названий программ 1С из JSON -> папки конфигураций
PROGRAM_TO_FOLDER_MAP = {
    "1С:БП": "БП",
    "1С:УНФ": "УНФ",
    "1С:КА/ERP": "ERP",
    "1С:Управление торговлей": "УТ",
    "1С:КА": "КА",
    "1С:ЗУП": "ЗУП",
    # Добавляй по мере необходимости
}

# --- Конфигурация ---
load_dotenv()  # Загрузка переменных окружения из файла .env

# Базовый путь к данным
BASE_DATA_DIR = "data" # Родительская директория для папок конфигураций (УНФ, БП и т.д.)
SQLITE_DB_PATH = os.path.join(BASE_DATA_DIR, "database", "1c_knowledge_base.db") # Путь к БД SQLite остается специфичным
SQLITE_FTS_TABLE = "documents_fts"

# Конфигурация ChromaDB
CHROMA_PERSIST_DIR = "db_chroma"
CHROMA_COLLECTION_NAME = "rag_collection"

# Конфигурация модели для эмбеддингов
EMBEDDING_MODEL_NAME_HF = "sentence-transformers/all-MiniLM-L6-v2"
USE_OPENAI_EMBEDDINGS = True

# Конфигурация разделителя текста
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Разделы 1С теперь представляют собой имена папок конфигураций верхнего уровня внутри BASE_DATA_DIR
# Например, data/УНФ/, data/БП/
# Скрипт будет автоматически определять эти папки.
# KNOWN_1C_SECTIONS больше не используется для поиска в пути, а для итерации по папкам конфигураций.
# Список этих папок будет получен динамически.


# --- Вспомогательные функции ---

def get_embedding_model():
    """Инициализирует и возвращает выбранную модель для эмбеддингов."""
    if USE_OPENAI_EMBEDDINGS:
        if os.getenv("OPENAI_API_KEY"):
            print("Используются OpenAI Embeddings.")
            return OpenAIEmbeddings()
        else:
            print("OPENAI_API_KEY не найден, используется HuggingFace Embeddings.")
            print(f"Используются HuggingFace Embeddings: {EMBEDDING_MODEL_NAME_HF}")
            return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME_HF,
                                       model_kwargs={'device': 'cpu'})
    else:
        print(f"Используются HuggingFace Embeddings: {EMBEDDING_MODEL_NAME_HF}")
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME_HF,
                                   model_kwargs={'device': 'cpu'})


def get_1c_section_from_path(file_path: str, base_data_dir: str) -> str:
    """
    Извлекает имя раздела 1С из пути к файлу.
    Раздел 1С - это имя директории первого уровня внутри base_data_dir.
    Пример: base_data_dir="data", file_path="data/УНФ/pdfs/doc1.pdf" -> "УНФ"
    """
    try:
        # Нормализуем base_data_dir, чтобы он заканчивался разделителем, если его нет
        normalized_base_path = os.path.join(os.path.normpath(base_data_dir), "")
        normalized_file_path = os.path.normpath(file_path)

        if not normalized_file_path.startswith(normalized_base_path):
            return "Unknown" # Файл не находится внутри базовой директории данных

        # Получаем относительный путь файла от базовой директории
        relative_path = os.path.relpath(normalized_file_path, normalized_base_path)

        # Раздел 1С - это первая часть относительного пути
        path_parts = relative_path.split(os.sep)
        if path_parts:
            return path_parts[0]
    except Exception as e:
        print(f"Ошибка при извлечении раздела из пути {file_path}: {e}")
        pass
    return "Unknown"


# --- Функции загрузки и обработки данных ---

def load_process_pdfs(config_specific_dir: str, section_name: str, text_splitter: RecursiveCharacterTextSplitter) -> list:
    """
    Загружает PDF-файлы из поддиректории 'pdfs' внутри директории конкретной конфигурации 1С,
    извлекает метаданные, разделяет их и возвращает чанки документов.
    config_specific_dir: Путь к папке конфигурации (например, "data/УНФ")
    section_name: Имя раздела 1С (например, "УНФ")
    """
    all_chunks = []
    pdf_data_subdir = os.path.join(config_specific_dir, "pdfs") # Ищем подпапку 'pdfs'

    print(f"  Загрузка PDF из: {pdf_data_subdir} для раздела '{section_name}'")
    if not os.path.isdir(pdf_data_subdir):
        print(f"    Каталог PDF не найден: {pdf_data_subdir}")
        return all_chunks

    for pdf_file_path in glob.glob(os.path.join(pdf_data_subdir, "*.pdf"), recursive=False): # Ищем PDF только в этой папке
        try:
            print(f"    Обработка PDF: {pdf_file_path}")
            loader = PyPDFLoader(pdf_file_path)
            documents = loader.load()

            processed_docs_for_file = []
            for doc in documents:
                doc.metadata["source_type"] = "pdf"
                doc.metadata["file_name"] = os.path.basename(pdf_file_path)
                doc.metadata["full_path"] = pdf_file_path
                doc.metadata["1c_section"] = section_name # Используем переданное имя раздела
                if 'page' not in doc.metadata:
                    doc.metadata['page_number'] = doc.metadata.get('page_number', -1) +1
                processed_docs_for_file.append(doc)

            chunks = text_splitter.split_documents(processed_docs_for_file)
            all_chunks.extend(chunks)
            print(f"      Загружено {len(documents)} страниц, разделено на {len(chunks)} чанков.")
        except Exception as e:
            print(f"      Ошибка при обработке PDF {pdf_file_path}: {e}")
    return all_chunks


def load_process_markdown(config_specific_dir: str, section_name: str, text_splitter: RecursiveCharacterTextSplitter) -> list:
    """
    Заглушка для загрузки Markdown файлов из поддиректории 'markdown' внутри директории конфигурации.
    (Будет реализовано позже)
    """
    all_chunks = []
    md_data_subdir = os.path.join(config_specific_dir, "markdown")
    print(f"  Загрузка Markdown из: {md_data_subdir} для раздела '{section_name}' (Заглушка - Не реализовано)")
    if not os.path.isdir(md_data_subdir):
        # print(f"    Каталог Markdown не найден: {md_data_subdir}") # Менее шумный вывод для заглушек
        return all_chunks
    # Пример структуры:
    # for md_file_path in glob.glob(os.path.join(md_data_subdir, "*.md"), recursive=False):
    #     try:
    #         print(f"    Обработка MD: {md_file_path}")
    #         loader = UnstructuredMarkdownLoader(md_file_path)
    #         documents = loader.load()
    #         for doc in documents:
    #             doc.metadata["source_type"] = "markdown"
    #             doc.metadata["file_name"] = os.path.basename(md_file_path)
    #             doc.metadata["full_path"] = md_file_path
    #             doc.metadata["1c_section"] = section_name
    #         chunks = text_splitter.split_documents(documents)
    #         all_chunks.extend(chunks)
    #         print(f"      Загружено и разделено на {len(chunks)} чанков.")
    #     except Exception as e:
    #         print(f"      Ошибка при обработке MD {md_file_path}: {e}")
    return all_chunks


def load_process_excel(config_specific_dir: str, section_name: str, text_splitter: RecursiveCharacterTextSplitter) -> list:
    all_chunks = []
    excel_data_subdir = os.path.join(config_specific_dir, "excel")
    print(f"  Загрузка Excel из: {excel_data_subdir} для раздела '{section_name}'")
    if not os.path.isdir(excel_data_subdir):
        return all_chunks

    for excel_file_path in glob.glob(os.path.join(excel_data_subdir, "*.xlsx"), recursive=False):
        try:
            print(f"    Обработка Excel: {excel_file_path}")
            loader = UnstructuredExcelLoader(excel_file_path)
            documents = loader.load()
            for doc in documents:
                doc.metadata["source_type"] = "excel"
                doc.metadata["file_name"] = os.path.basename(excel_file_path)
                doc.metadata["full_path"] = excel_file_path
                doc.metadata["1c_section"] = section_name
            chunks = text_splitter.split_documents(documents)
            all_chunks.extend(chunks)
            print(f"      Загружено и разделено на {len(chunks)} чанков.")
        except Exception as e:
            print(f"      Ошибка при обработке Excel {excel_file_path}: {e}")
    return all_chunks

def load_process_faq(config_specific_dir: str, section_name: str, text_splitter: RecursiveCharacterTextSplitter) -> list:
    all_chunks = []
    faq_data_subdir = os.path.join(config_specific_dir, "faq")
    print(f"  Загрузка FAQ из: {faq_data_subdir} для раздела '{section_name}'")
    if not os.path.isdir(faq_data_subdir):
        return all_chunks

    for faq_file_path in glob.glob(os.path.join(faq_data_subdir, "*.json"), recursive=False):
        try:
            print(f"    Обработка FAQ: {faq_file_path}")
            with open(faq_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            documents = []
            for item in data:
                q = item.get("question", "").strip()
                a = item.get("answer", "").strip()
                if q and a:
                    content = f"Вопрос: {q}\n\nОтвет: {a}"
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source_type": "faq",
                            "file_name": os.path.basename(faq_file_path),
                            "full_path": faq_file_path,
                            "1c_section": section_name,
                            "url": item.get("url", ""),
                            "date": item.get("date", ""),
                            "question": item.get("question", "")
                        }
                    ))
            # Используем стандартный, надежный метод для разделения документов на чанки
            chunks = text_splitter.split_documents(documents)
            all_chunks.extend(chunks)
            print(f"      Загружено {len(documents)} FAQ, разделено на {len(chunks)} чанков.")
        except Exception as e:
            print(f"      Ошибка при обработке FAQ {faq_file_path}: {e}")
    return all_chunks


def load_process_sqlite(db_path: str, text_splitter: RecursiveCharacterTextSplitter) -> list:
    """
    Заглушка для загрузки данных из SQLite.
    Для SQLite метаданные '1c_section' должны извлекаться из специального столбца в таблицах,
    а не из пути к файлу самой БД.
    (Будет реализовано позже)
    """
    all_chunks = []
    print(f"\nЗагрузка данных SQLite из: {db_path} (Заглушка - Не реализовано)")
    print("  Примечание: Для SQLite, метаданные '1c_section' должны быть в таблице.")
    if not os.path.isfile(db_path):
        print(f"Файл БД SQLite не найден: {db_path}")
        return all_chunks
    # Пример структуры:
    # from sqlalchemy import create_engine, text
    # engine = create_engine(f"sqlite:///{db_path}")
    # with engine.connect() as connection:
    #     result = connection.execute(text("SELECT id, content_column, 1c_section_column, source_info FROM your_table_name")) # Предполагается наличие 1c_section_column
    #     for row in result:
    #         page_content = row.content_column
    #         metadata = {
    #             "source_type": "sqlite",
    #             "db_table": "your_table_name",
    #             "record_id": str(row.id),
    #             "1c_section": row['1c_section_column'], # Извлечение из столбца
    #             "original_source": row.source_info
    #         }
    #         doc = Document(page_content=page_content, metadata=metadata)
    #         current_chunks = text_splitter.split_documents([doc])
    #         all_chunks.extend(current_chunks)
    # print(f"    Обработано записей и разделено на {len(all_chunks)} чанков.")
    return all_chunks


# --- Основная логика индексации ---

def split_faq_json_by_program(source_dir: str, base_data_dir: str) -> None:
    """
    Ищет JSON-файлы в директории source_dir и разбивает их по программам,
    записывая отдельные файлы в соответствующие папки внутри base_data_dir.
    """
    json_mapping_path = os.path.join(source_dir, "mappings.json")
    json_program_map = {}
    if os.path.exists(json_mapping_path):
        with open(json_mapping_path, "r", encoding="utf-8") as f:
            json_program_map = json.load(f).get("json_program_map", {})
    for json_file in glob.glob(os.path.join(source_dir, "*.json")):
        if os.path.basename(json_file) == "mappings.json":
            continue  # пропускаем mappings.json
        try:
            print(f"\n📦 Обнаружен общий FAQ-файл: {json_file}")
            with open(json_file, "r", encoding="utf-8") as f:
                entries = json.load(f)

            program_map = defaultdict(list)

            for block in entries:
                program = block.get("program", "Unknown").strip()
                questions = block.get("questions", [])
                program_map[program].extend(questions)

            for program_name, qlist in program_map.items():
                folder_name = json_program_map.get(program_name, program_name.replace("1С:", "").strip())
                target_dir = os.path.join(base_data_dir, folder_name, "faq")
                os.makedirs(target_dir, exist_ok=True)
                output_path = os.path.join(target_dir, f"from_import_{os.path.basename(json_file)}")
                with open(output_path, "w", encoding="utf-8") as out_f:
                    json.dump(qlist, out_f, ensure_ascii=False, indent=2)
                print(f"  ✅ Сохранено {len(qlist)} QA в {output_path}")
            
            # Удалим оригинальный файл после успешной обработки
            os.remove(json_file)
            print(f"  🗑️ Удалён исходный файл: {json_file}")
        except Exception as e:
            print(f"  ⚠️ Ошибка при разборе {json_file}: {e}")

def main():
    """
    Основная функция для управления загрузкой, обработкой и индексацией данных.
    """
    print("Запуск процесса индексации данных RAG...")

    embedding_model = get_embedding_model()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    all_document_chunks = []

    if not os.path.isdir(BASE_DATA_DIR):
        print(f"Ошибка: Базовая директория данных '{BASE_DATA_DIR}' не найдена.")
        return

    # Предобработка FAQ-файлов общего назначения
    shared_import_dir = os.path.join(BASE_DATA_DIR, "import")
    if os.path.isdir(shared_import_dir):
        split_faq_json_by_program(shared_import_dir, BASE_DATA_DIR)

    # Импорт Excel-файла в формат FAQ, если он существует
    excel_import_file = os.path.join(BASE_DATA_DIR, "import", "kb_import.xlsx")
    mappings_path = os.path.join(BASE_DATA_DIR, "import", "mappings.json")
    if os.path.exists(excel_import_file):
        convert_excel_to_faq_format(excel_import_file, mapping_path=mappings_path)
        try:
            os.remove(excel_import_file)
            print(f"🗑️ Удалён импортированный Excel-файл: {excel_import_file}")
        except Exception as e:
            print(f"⚠️ Не удалось удалить Excel-файл {excel_import_file}: {e}")

    # Итерация по папкам конфигураций (УНФ, БП и т.д.) внутри BASE_DATA_DIR
    for section_folder_name in os.listdir(BASE_DATA_DIR):
        if section_folder_name.startswith("."):
            continue  # Пропустить скрытые папки (например, .DS_Store, .streamlit)
        config_specific_dir = os.path.join(BASE_DATA_DIR, section_folder_name)
        if os.path.isdir(config_specific_dir):
            # Проверяем, является ли имя папки одним из "известных" разделов,
            # хотя get_1c_section_from_path теперь извлекает его из пути.
            # Эта проверка может быть полезна для логирования или пропуска нерелевантных папок.
            print(f"\nОбработка конфигурации/раздела: {section_folder_name}")

            # Важно: section_name для метаданных теперь это имя папки config_specific_dir
            # Это гарантирует, что get_1c_section_from_path будет работать правильно,
            # если бы он вызывался для каждого файла отдельно.
            # Но мы уже знаем section_name на этом этапе.

            all_document_chunks.extend(load_process_pdfs(config_specific_dir, section_folder_name, text_splitter))
            all_document_chunks.extend(load_process_markdown(config_specific_dir, section_folder_name, text_splitter))
            all_document_chunks.extend(load_process_excel(config_specific_dir, section_folder_name, text_splitter))
            all_document_chunks.extend(load_process_faq(config_specific_dir, section_folder_name, text_splitter))
            # SQLite обрабатывается отдельно, так как его структура не привязана к папкам конфигураций таким же образом

    # Загрузка данных из SQLite (обрабатывается отдельно)
    all_document_chunks.extend(load_process_sqlite(SQLITE_DB_PATH, text_splitter))


    if not all_document_chunks:
        print("\nДокументы не найдены или не обработаны. Выход.")
        return

    print(f"\nВсего чанков документов для индексации: {len(all_document_chunks)}")

    # --- Начало индексации в SQLite FTS ---
    print(f"\nНачало индексации в SQLite FTS DB: {SQLITE_DB_PATH}")
    conn = None
    try:
        # Убедимся, что директория для БД существует
        db_dir = os.path.dirname(SQLITE_DB_PATH)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
            print(f"  Создана директория для SQLite БД: {db_dir}")

        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()

        # Создание FTS таблицы, если она не существует
        # doc_id должен быть уникальным для корректной работы INSERT OR REPLACE
        # Используем full_path + номер чанка для уникальности, если документ разделен на чанки
        create_table_sql = f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {SQLITE_FTS_TABLE} (
            doc_id TEXT UNIQUE,
            content TEXT,
            tokenize = 'porter unicode61'
        );
        """
        cursor.execute(create_table_sql)
        print(f"  FTS таблица '{SQLITE_FTS_TABLE}' готова/создана.")

        # Удаление старых записей перед вставкой новых (если это необходимо)
        # Например, если идентификаторы могут меняться или документы удаляются
        # cursor.execute(f"DELETE FROM {SQLITE_FTS_TABLE};")
        # print(f"  Старые записи из '{SQLITE_FTS_TABLE}' удалены (если были).")


        inserted_count = 0
        if all_document_chunks:
            for i, doc_chunk in enumerate(all_document_chunks):
                content = doc_chunk.page_content
                # Создание уникального doc_id для каждого чанка
                base_path = doc_chunk.metadata.get('full_path', doc_chunk.metadata.get('file_name', f"unknown_doc_{i}"))
                page_num = doc_chunk.metadata.get('page_number', -1) # -1 если нет номера страницы

                # Для уникальности ID чанка, можно добавить номер страницы и индекс чанка (если один документ дает много чанков)
                # Однако, text_splitter обычно не добавляет индекс чанка в метаданные.
                # Если full_path + page_number достаточно уникальны для чанка, используем их.
                # Если документ не PDF и не имеет страниц, page_num будет -1.
                # Простой ID на основе индекса чанка в all_document_chunks для гарантии уникальности.
                chunk_specific_id = f"_chunk_{i}" # Добавляем индекс чанка ко всем ID для уникальности

                doc_id = f"{base_path}_page_{page_num}{chunk_specific_id}" if page_num != -1 else f"{base_path}{chunk_specific_id}"

                if not content.strip(): # Пропускаем пустые чанки
                    print(f"    Пропущен пустой чанк для doc_id: {doc_id}")
                    continue

                try:
                    cursor.execute(
                        f"INSERT OR REPLACE INTO {SQLITE_FTS_TABLE} (doc_id, content) VALUES (?, ?)",
                        (doc_id, content)
                    )
                    inserted_count += 1
                except sqlite3.IntegrityError as ie: # Должно быть обработано INSERT OR REPLACE, но на всякий случай
                    print(f"    Ошибка целостности при вставке doc_id {doc_id}: {ie}. Возможно, проблема с UNIQUE constraint, если doc_id не уникален.")
                except Exception as e_insert:
                    print(f"    Ошибка при вставке чанка для doc_id {doc_id}: {e_insert}")


            conn.commit()
            print(f"  Успешно вставлено/заменено {inserted_count} чанков в FTS таблицу.")
        else:
            print("  Нет чанков для индексации в SQLite FTS.")

    except sqlite3.Error as e:
        print(f"  Ошибка SQLite: {e}")
    except Exception as e_global:
        print(f"  Непредвиденная ошибка при работе с SQLite: {e_global}")
    finally:
        if conn:
            conn.close()
            print(f"  Соединение с SQLite ({SQLITE_DB_PATH}) закрыто.")
    print("--- Индексация в SQLite FTS завершена ---")


    import shutil
    # Очистка старой базы ChromaDB и/или TF-IDF артефактов перед созданием новой
    # Это важно, чтобы избежать конфликтов или использования устаревших данных.
    # Если CHROMA_PERSIST_DIR используется и для Chroma, и для TF-IDF, rmtree очистит всё.
    if os.path.exists(CHROMA_PERSIST_DIR):
        print(f"Удаление старой директории для хранения Chroma и TF-IDF: {CHROMA_PERSIST_DIR}")
        # В реальном сценарии здесь может быть более гранулярная очистка,
        # например, удаление только поддиректории Chroma или определенных файлов.
        # Для данного задания, если директория существует, предполагаем, что ее можно пересоздать.
        shutil.rmtree(CHROMA_PERSIST_DIR)

    # Гарантируем, что директория существует перед любыми операциями записи
    if not os.path.exists(CHROMA_PERSIST_DIR):
        os.makedirs(CHROMA_PERSIST_DIR)
        print(f"Создан каталог для Chroma и TF-IDF: {CHROMA_PERSIST_DIR}")

    # --- Генерация и сохранение TF-IDF ---
    print("\nГенерация TF-IDF векторов...")
    try:
        if all_document_chunks:
            texts = [doc.page_content for doc in all_document_chunks]
            # Обеспечиваем уникальный идентификатор для каждого документа
            doc_ids = []
            for i, doc in enumerate(all_document_chunks):
                # Пытаемся получить 'full_path', если нет, то 'file_name', если нет, то генерируем уникальный ID
                path = doc.metadata.get('full_path', doc.metadata.get('file_name'))
                if path:
                    doc_ids.append(path)
                else:
                    # Если путь не доступен, создаем идентификатор на основе индекса и типа источника
                    source_type = doc.metadata.get('source_type', 'unknown_source')
                    doc_ids.append(f"{source_type}_{i}")

            if texts:
                tfidf_vectorizer = TfidfVectorizer(
                    max_df=0.95,
                    min_df=2,
                    ngram_range=(1, 2),
                    stop_words=None # Для русского языка нужны специфичные стоп-слова или их отсутствие
                                    # Можно рассмотреть 'russian' если scikit-learn поддерживает или использовать внешнюю библиотеку
                )
                tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

                tfidf_model_path = os.path.join(CHROMA_PERSIST_DIR, "tfidf_model.joblib")
                tfidf_vectors_path = os.path.join(CHROMA_PERSIST_DIR, "tfidf_vectors_and_ids.joblib")

                joblib.dump(tfidf_vectorizer, tfidf_model_path)
                print(f"  TF-IDF модель сохранена в: {tfidf_model_path}")

                joblib.dump({'ids': doc_ids, 'vectors': tfidf_matrix}, tfidf_vectors_path)
                print(f"  TF-IDF векторы и идентификаторы сохранены в: {tfidf_vectors_path}")
                print(f"  Размерность TF-IDF матрицы: {tfidf_matrix.shape}")
            else:
                print("  Нет текстовых данных для генерации TF-IDF (список текстов пуст).")
        else:
            print("  Нет чанков документов для генерации TF-IDF.")

    except Exception as e:
        print(f"  Ошибка при генерации или сохранении TF-IDF: {e}")
    print("--- Генерация TF-IDF завершена ---")
    # --- Конец генерации TF-IDF ---

    print(f"\nИнициализация векторного хранилища Chroma в: {CHROMA_PERSIST_DIR}")
    print(f"Используется имя коллекции: {CHROMA_COLLECTION_NAME}")

    # На этом этапе CHROMA_PERSIST_DIR уже должен существовать.
    # Если он был удален и не создан снова до этого момента, Chroma.from_documents может выдать ошибку.
    # Но мы уже создали его выше.

    try:
        vector_store = Chroma.from_documents(
            documents=all_document_chunks,
            embedding=embedding_model,
            collection_name=CHROMA_COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR
        )
        print(f"\nУспешно проиндексировано {len(all_document_chunks)} чанков в Chroma.")
        print(f"Векторное хранилище сохранено в: {CHROMA_PERSIST_DIR}")
    except Exception as e:
        print(f"Ошибка во время индексации в ChromaDB: {e}")

    print("\nПроцесс индексации завершен.")


if __name__ == "__main__":
    # Пример создания структуры папок для теста, если они не существуют
    # Пользователь должен будет создать эти папки и разместить в них файлы.
    # os.makedirs(os.path.join(BASE_DATA_DIR, "УНФ", "pdfs"), exist_ok=True)
    # os.makedirs(os.path.join(BASE_DATA_DIR, "БП", "pdfs"), exist_ok=True)
    # os.makedirs(os.path.join(BASE_DATA_DIR, "УНФ", "markdown"), exist_ok=True)
    # os.makedirs(os.path.join(BASE_DATA_DIR, "database"), exist_ok=True)
    # print(f"Убедитесь, что директория '{BASE_DATA_DIR}' существует и содержит подпапки для конфигураций (например, УНФ, БП),")
    # print(f"а внутри них - подпапки для типов данных (pdfs, markdown, excel).")
    # print(f"Например: {os.path.join(BASE_DATA_DIR, 'УНФ', 'pdfs', 'your_document.pdf')}")

    main()
