import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader, PandasExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
# from langchain_community.document_loaders.sql_database import SQLDatabaseLoader # Пример, может потребовать кастомной обработки
# from langchain_community.utilities.sql_database import SQLDatabase # Пример

# --- Конфигурация ---
load_dotenv()  # Загрузка переменных окружения из файла .env

# Пути к данным
DATA_DIR = "data"
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
MD_DIR = os.path.join(DATA_DIR, "markdown")
EXCEL_DIR = os.path.join(DATA_DIR, "excel")
SQLITE_DB_PATH = os.path.join(DATA_DIR, "database", "1c_knowledge_base.db")

# Конфигурация ChromaDB
CHROMA_PERSIST_DIR = "db_chroma"
CHROMA_COLLECTION_NAME = "rag_collection"

# Конфигурация модели для эмбеддингов
# Вариант 1: HuggingFace Embeddings (например, многоязычная или русская модель)
# Убедитесь, что выбрали модель, подходящую для русского языка, если он основной.
# Используется меньшая, универсальная модель в качестве примера. Замените при необходимости.
EMBEDDING_MODEL_NAME_HF = "sentence-transformers/all-MiniLM-L6-v2"
# Вариант 2: OpenAI Embeddings (требует OPENAI_API_KEY в .env)
USE_OPENAI_EMBEDDINGS = True  # Установите False для использования HuggingFaceEmbeddings

# Конфигурация разделителя текста
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Разделы 1С для поиска в путях (добавьте больше при необходимости)
KNOWN_1C_SECTIONS = ["УНФ", "БП", "ERP", "Розница", "ЗУП"]


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
                                       model_kwargs={'device': 'cpu'}) # Или 'cuda', если доступно
    else:
        print(f"Используются HuggingFace Embeddings: {EMBEDDING_MODEL_NAME_HF}")
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME_HF,
                                   model_kwargs={'device': 'cpu'})


def get_1c_section_from_path(file_path: str) -> str:
    """
    Пытается извлечь раздел 1С из пути к файлу.
    Предполагается, что разделы являются частью структуры каталогов.
    Пример: data/pdfs/УНФ/doc1.pdf -> "УНФ"
    """
    try:
        path_parts = os.path.normpath(file_path).split(os.sep)
        for part in path_parts:
            if part in KNOWN_1C_SECTIONS:
                return part
    except Exception:
        pass # Ошибка при обработке пути
    return "Unknown" # Возвращаем "Unknown", если раздел не найден


# --- Функции загрузки и обработки данных ---

def load_process_pdfs(pdf_dir: str, text_splitter: RecursiveCharacterTextSplitter) -> list:
    """
    Загружает PDF-файлы из каталога, извлекает метаданные, разделяет их и возвращает чанки документов.
    """
    all_chunks = []
    print(f"\nЗагрузка PDF из: {pdf_dir}")
    if not os.path.isdir(pdf_dir):
        print(f"Каталог PDF не найден: {pdf_dir}")
        return all_chunks

    for pdf_file_path in glob.glob(os.path.join(pdf_dir, "**", "*.pdf"), recursive=True):
        try:
            print(f"  Обработка PDF: {pdf_file_path}")
            loader = PyPDFLoader(pdf_file_path)
            documents = loader.load()  # Каждая страница - это Document

            processed_docs_for_file = []
            for doc in documents:
                # Добавление пользовательских метаданных
                doc.metadata["source_type"] = "pdf"
                doc.metadata["file_name"] = os.path.basename(pdf_file_path)
                doc.metadata["full_path"] = pdf_file_path
                doc.metadata["1c_section"] = get_1c_section_from_path(pdf_file_path)
                # номер страницы часто включается PyPDFLoader как 'page'
                if 'page' not in doc.metadata: # на всякий случай
                    doc.metadata['page_number'] = doc.metadata.get('page_number', -1) +1


                processed_docs_for_file.append(doc)

            chunks = text_splitter.split_documents(processed_docs_for_file)
            all_chunks.extend(chunks)
            print(f"    Загружено {len(documents)} страниц, разделено на {len(chunks)} чанков.")
        except Exception as e:
            print(f"    Ошибка при обработке PDF {pdf_file_path}: {e}")
    return all_chunks


def load_process_markdown(md_dir: str, text_splitter: RecursiveCharacterTextSplitter) -> list:
    """
    Заглушка для загрузки Markdown файлов.
    (Будет реализовано позже)
    """
    all_chunks = []
    print(f"\nЗагрузка Markdown из: {md_dir} (Заглушка - Не реализовано)")
    if not os.path.isdir(md_dir):
        print(f"Каталог Markdown не найден: {md_dir}")
        return all_chunks
    # Пример структуры:
    # for md_file_path in glob.glob(os.path.join(md_dir, "**", "*.md"), recursive=True):
    #     try:
    #         print(f"  Обработка MD: {md_file_path}")
    #         loader = UnstructuredMarkdownLoader(md_file_path)
    #         documents = loader.load()
    #         for doc in documents:
    #             doc.metadata["source_type"] = "markdown"
    #             doc.metadata["file_name"] = os.path.basename(md_file_path)
    #             doc.metadata["full_path"] = md_file_path
    #             doc.metadata["1c_section"] = get_1c_section_from_path(md_file_path)
    #         chunks = text_splitter.split_documents(documents)
    #         all_chunks.extend(chunks)
    #         print(f"    Загружено и разделено на {len(chunks)} чанков.")
    #     except Exception as e:
    #         print(f"    Ошибка при обработке MD {md_file_path}: {e}")
    return all_chunks


def load_process_excel(excel_dir: str, text_splitter: RecursiveCharacterTextSplitter) -> list:
    """
    Заглушка для загрузки Excel файлов.
    (Будет реализовано позже)
    """
    all_chunks = []
    print(f"\nЗагрузка Excel из: {excel_dir} (Заглушка - Не реализовано)")
    if not os.path.isdir(excel_dir):
        print(f"Каталог Excel не найден: {excel_dir}")
        return all_chunks
    # Пример структуры:
    # for excel_file_path in glob.glob(os.path.join(excel_dir, "**", "*.xlsx"), recursive=True):
    #     try:
    #         print(f"  Обработка Excel: {excel_file_path}")
    #         # PandasExcelLoader загружает определенные листы и столбцы.
    #         # UnstructuredExcelLoader может быть проще для извлечения всего текста.
    #         loader = PandasExcelLoader(excel_file_path, sheet_name=None) # Или UnstructuredExcelLoader
    #         documents = loader.load() # Это требует осторожной обработки в зависимости от структуры Excel
    #         # Скорее всего, потребуется итерировать по строкам/листам и создавать документы вручную
    #         # или обрабатывать вывод загрузчика для соответствия структуре Document.
    #         # Например, каждая строка может быть документом, или содержимое определенных ячеек.
    #         # Убедитесь, что добавлены метаданные, такие как имя листа, номер строки.
    #         # for doc in documents: # Это концептуально
    #         #     doc.metadata["source_type"] = "excel"
    #         #     doc.metadata["file_name"] = os.path.basename(excel_file_path)
    #         #     doc.metadata["full_path"] = excel_file_path
    #         #     doc.metadata["1c_section"] = get_1c_section_from_path(excel_file_path) # Если применимо из пути
    #         #     # Добавьте более специфичные метаданные, такие как sheet_name, row_number
    #         # chunks = text_splitter.split_documents(documents) # Может потребоваться разделить содержимое ячейки
    #         # all_chunks.extend(chunks)
    #         # print(f"    Загружено и разделено на {len(chunks)} чанков.")
    #     except Exception as e:
    #         print(f"    Ошибка при обработке Excel {excel_file_path}: {e}")
    return all_chunks


def load_process_sqlite(db_path: str, text_splitter: RecursiveCharacterTextSplitter) -> list:
    """
    Заглушка для загрузки данных из SQLite.
    (Будет реализовано позже)
    """
    all_chunks = []
    print(f"\nЗагрузка данных SQLite из: {db_path} (Заглушка - Не реализовано)")
    if not os.path.isfile(db_path):
        print(f"Файл БД SQLite не найден: {db_path}")
        return all_chunks
    # Пример структуры:
    # try:
    #     # db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    #     # Пример запроса: query = "SELECT id, content_column, metadata_column, 1c_section_column FROM your_table"
    #     # results = db.run(query) # Это нужно парсить из строки или использовать SQLAlchemy напрямую
    #
    #     # Использование SQLAlchemy напрямую часто более гибко:
    #     # from sqlalchemy import create_engine, text
    #     # engine = create_engine(f"sqlite:///{db_path}")
    #     # with engine.connect() as connection:
    #     #     result = connection.execute(text("SELECT id, content_column, 1c_section_column, source_info FROM your_table_name"))
    #     #     for row in result:
    #     #         # Предполагая, что row имеет атрибуты, такие как id, content_column, и т.д.
    #     #         page_content = row.content_column
    #     #         metadata = {
    #     #             "source_type": "sqlite",
    #     #             "db_table": "your_table_name",
    #     #             "record_id": str(row.id),
    #     #             "1c_section": row.1c_section_column, # Убедитесь, что этот столбец существует
    #     #             "original_source": row.source_info # например, forum_url
    #     #         }
    #     #         doc = Document(page_content=page_content, metadata=metadata)
    #     #         # Разделить, если содержимое большое
    #     #         # current_chunks = text_splitter.split_documents([doc])
    #     #         # all_chunks.extend(current_chunks)
    #     # print(f"    Обработано записей и разделено на {len(all_chunks)} чанков.")
    # except Exception as e:
    #     print(f"    Ошибка при обработке БД SQLite {db_path}: {e}")
    return all_chunks


# --- Основная логика индексации ---

def main():
    """
    Основная функция для управления загрузкой, обработкой и индексацией данных.
    """
    print("Запуск процесса индексации данных RAG...")

    # Инициализация компонентов
    embedding_model = get_embedding_model()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    # Загрузка и обработка всех источников данных
    all_document_chunks = []
    all_document_chunks.extend(load_process_pdfs(PDF_DIR, text_splitter))
    all_document_chunks.extend(load_process_markdown(MD_DIR, text_splitter))
    all_document_chunks.extend(load_process_excel(EXCEL_DIR, text_splitter))
    all_document_chunks.extend(load_process_sqlite(SQLITE_DB_PATH, text_splitter))

    if not all_document_chunks:
        print("\nДокументы не найдены или не обработаны. Выход.")
        return

    print(f"\nВсего чанков документов для индексации: {len(all_document_chunks)}")

    # Инициализация векторного хранилища Chroma и добавление документов
    # Использование Chroma.from_documents создаст коллекцию, если она не существует,
    # и добавит документы. Если каталог и коллекция существуют, и вы запускаете это снова,
    # это может привести к дубликатам в зависимости от поведения Chroma или если вы стремитесь
    # воссоздать ее. Для простоты, эта версия предполагает свежую сборку или простые добавления,
    # если базовый клиент Chroma настроен для этого.
    # Более надежная обработка включала бы проверку существования коллекции, удаление при необходимости,
    # или использование `vector_store.add_documents()` с существующим хранилищем.
    print(f"\nИнициализация векторного хранилища Chroma в: {CHROMA_PERSIST_DIR}")
    print(f"Используется имя коллекции: {CHROMA_COLLECTION_NAME}")

    # Создание каталога, если он не существует
    if not os.path.exists(CHROMA_PERSIST_DIR):
        os.makedirs(CHROMA_PERSIST_DIR)
        print(f"Создан каталог для Chroma: {CHROMA_PERSIST_DIR}")

    try:
        vector_store = Chroma.from_documents(
            documents=all_document_chunks,
            embedding=embedding_model,
            collection_name=CHROMA_COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR
        )
        # Явное сохранение, если from_documents не делает это автоматически (зависит от версии)
        # vector_store.persist() # Клиент Chroma обычно обрабатывает это с persist_directory

        print(f"\nУспешно проиндексировано {len(all_document_chunks)} чанков в Chroma.")
        print(f"Векторное хранилище сохранено в: {CHROMA_PERSIST_DIR}")

        # Вы можете протестировать хранилище поиском по сходству (необязательно)
        # if len(all_document_chunks) > 0:
        #     print("\nВыполнение быстрого тестового поиска по сходству...")
        #     results = vector_store.similarity_search("тест", k=1)
        #     if results:
        #         print(f"  Тестовый поиск нашел: {results[0].page_content[:100]}...")
        #     else:
        #         print("  Тестовый поиск не вернул результатов (коллекция может быть пуста или проблема с запросом).")

    except Exception as e:
        print(f"Ошибка во время индексации в ChromaDB: {e}")

    print("\nПроцесс индексации завершен.")


if __name__ == "__main__":
    # Создание тестовых каталогов для данных, если они не существуют
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(MD_DIR, exist_ok=True)
    os.makedirs(EXCEL_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "database"), exist_ok=True)
    # Пример: Создание пустого PDF для тестирования (PyPDFLoader требует реального PDF)
    # Для реального теста, поместите настоящий PDF файл в data/pdfs/УНФ/dummy_UNF_doc.pdf
    # with open(os.path.join(PDF_DIR, "УНФ", "dummy_UNF_doc.pdf"), "wb") as f:
    #     # PyPDFLoader не сможет загрузить пустой файл как PDF.
    #     # Для теста нужен минимальный валидный PDF или используйте обработку исключений в load_process_pdfs.
    #     pass


    main()
