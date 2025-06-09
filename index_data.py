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
from langchain.schema import Document # Для создания документов из SQLite данных

# --- Конфигурация ---
load_dotenv()  # Загрузка переменных окружения из файла .env

# Базовый путь к данным
BASE_DATA_DIR = "data" # Родительская директория для папок конфигураций (УНФ, БП и т.д.)
SQLITE_DB_PATH = os.path.join(BASE_DATA_DIR, "database", "1c_knowledge_base.db") # Путь к БД SQLite остается специфичным

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
    """
    Заглушка для загрузки Excel файлов из поддиректории 'excel' внутри директории конфигурации.
    (Будет реализовано позже)
    """
    all_chunks = []
    excel_data_subdir = os.path.join(config_specific_dir, "excel")
    print(f"  Загрузка Excel из: {excel_data_subdir} для раздела '{section_name}' (Заглушка - Не реализовано)")
    if not os.path.isdir(excel_data_subdir):
        # print(f"    Каталог Excel не найден: {excel_data_subdir}")
        return all_chunks
    # Пример структуры:
    # for excel_file_path in glob.glob(os.path.join(excel_data_subdir, "*.xlsx"), recursive=False):
    # ... (логика аналогична предыдущим, с учетом специфики Excel)
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

    # Итерация по папкам конфигураций (УНФ, БП и т.д.) внутри BASE_DATA_DIR
    for section_folder_name in os.listdir(BASE_DATA_DIR):
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
            # SQLite обрабатывается отдельно, так как его структура не привязана к папкам конфигураций таким же образом

    # Загрузка данных из SQLite (обрабатывается отдельно)
    all_document_chunks.extend(load_process_sqlite(SQLITE_DB_PATH, text_splitter))


    if not all_document_chunks:
        print("\nДокументы не найдены или не обработаны. Выход.")
        return

    print(f"\nВсего чанков документов для индексации: {len(all_document_chunks)}")

    print(f"\nИнициализация векторного хранилища Chroma в: {CHROMA_PERSIST_DIR}")
    print(f"Используется имя коллекции: {CHROMA_COLLECTION_NAME}")

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
