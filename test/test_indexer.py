# test_indexer.py
import pytest
import pandas as pd 
import os
import json
from unittest.mock import patch, MagicMock
from langchain.schema import Document

# Импортируем функции из основного скрипта
from index_data import (
    convert_excel_to_faq_format,
    load_process_pdfs,
    load_process_faq,
    load_process_excel,
    split_faq_json_by_program,
    main,
    BASE_DATA_DIR, # Должны быть замоканы через conftest
    CHROMA_PERSIST_DIR,
    SQLITE_DB_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

# Используем фикстуры из conftest.py
# mock_global_paths, mock_embedding_model, mock_os_environ, mock_pypdf_loader, mock_unstructured_excel_loader
# и другие фикстуры для создания тестовых файлов.

def test_convert_excel_to_faq_format_success(temp_data_dir, create_dummy_excel, create_excel_mapping):
    """
    Проверяет, что convert_excel_to_faq_format корректно преобразует Excel в FAQ JSONs.
    """
    excel_path = create_dummy_excel
    mapping_path = create_excel_mapping

    convert_excel_to_faq_format(excel_path, mapping_path)

    # Проверяем, что файлы были созданы в правильных директориях
    bp_faq_path = os.path.join(temp_data_dir, "БП", "faq", "from_excel_SD_1cfresh.json")
    unf_faq_path = os.path.join(temp_data_dir, "УНФ", "faq", "from_excel_SD_1cfresh.json")

    assert os.path.exists(bp_faq_path)
    assert os.path.exists(unf_faq_path)

    with open(bp_faq_path, "r", encoding="utf-8") as f:
        bp_data = json.load(f)
    assert len(bp_data) == 2 # Вопрос 1 и Вопрос 2 (т.к. Вопрос 2 имеет также компонент БП)
    assert any(item["question"] == "Тестовый вопрос 1" for item in bp_data)
    assert any(item["question"] == "Тестовый вопрос 2" for item in bp_data)
    assert bp_data[0]["source_type"] == "excel_kb"

    with open(unf_faq_path, "r", encoding="utf-8") as f:
        unf_data = json.load(f)
    assert len(unf_data) == 1
    assert unf_data[0]["question"] == "Тестовый вопрос 2"
    assert unf_data[0]["source_type"] == "excel_kb"

def test_convert_excel_to_faq_format_no_mapping(temp_data_dir, create_dummy_excel, capsys):
    """
    Проверяет поведение convert_excel_to_faq_format, когда нет маппинга.
    """
    excel_path = create_dummy_excel
    non_existent_mapping_path = os.path.join(temp_data_dir, "no_such_mappings.json")

    convert_excel_to_faq_format(excel_path, non_existent_mapping_path)
    captured = capsys.readouterr()
    assert "Нет маппинга" in captured.out

def test_convert_excel_to_faq_format_missing_columns(temp_data_dir, capsys):
    """
    Проверяет поведение convert_excel_to_faq_format при отсутствии необходимых колонок.
    """
    excel_import_dir = os.path.join(temp_data_dir, "import")
    os.makedirs(excel_import_dir, exist_ok=True)
    excel_path = os.path.join(excel_import_dir, "kb_import_missing_col.xlsx")
    
    df_data = {
        "Наименование": ["Тестовый вопрос 1"],
        "Номер": ["123"],
        "Ссылка": ["http://link1.com"],
        "Описание": ["Тестовый ответ 1"],
        # "Компоненты": ["1С:БП"], # Отсутствует
        "Разделы": ["Базовый"]
    }
    df = pd.DataFrame(df_data)
    df.to_excel(excel_path, index=False)

    mapping_path = os.path.join(excel_import_dir, "mappings.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({"kb_import_missing_col.xlsx": {"source_type": "excel_kb", "program_map": {"1С:БП": "БП"}}}, f)

    convert_excel_to_faq_format(excel_path, mapping_path)
    captured = capsys.readouterr()
    assert "Отсутствует колонка: Компоненты" in captured.out

### Тесты для функций загрузки и обработки данных

# text_splitter для тестов
from langchain.text_splitter import RecursiveCharacterTextSplitter
test_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    is_separator_regex=False,
)

def test_load_process_pdfs(temp_data_dir, mock_pypdf_loader, capsys):
    """
    Проверяет, что load_process_pdfs корректно загружает, обрабатывает и чанкует PDF.
    """
    config_dir = os.path.join(temp_data_dir, "БП")
    pdf_dir = os.path.join(config_dir, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    dummy_pdf_path = os.path.join(pdf_dir, "test.pdf")
    # Создаем фиктивный файл, который будет "прочитан" моком
    with open(dummy_pdf_path, "w") as f:
        f.write("dummy content")

    section_name = "БП"
    
    chunks = load_process_pdfs(config_dir, section_name, test_text_splitter)

    assert mock_pypdf_loader.called # Проверяем, что PyPDFLoader был вызван
    assert len(chunks) > 0
    for chunk in chunks:
        assert "Test PDF content" in chunk.page_content
        assert chunk.metadata["source_type"] == "pdf"
        assert chunk.metadata["1c_section"] == section_name
        assert chunk.metadata["file_name"] == "test.pdf"
        assert chunk.metadata["full_path"] == dummy_pdf_path
        assert 'page_number' in chunk.metadata
        assert chunk.metadata['page_number'] in [1, 2] # PyPDFLoader возвращает 0-индексированные страницы, мы добавляем +1

    captured = capsys.readouterr()
    assert f"Обработка PDF: {dummy_pdf_path}" in captured.out
    assert f"Загружено 2 страниц, разделено на" in captured.out


def test_load_process_excel(temp_data_dir, mock_unstructured_excel_loader, capsys):
    """
    Проверяет, что load_process_excel корректно загружает, обрабатывает и чанкует Excel.
    """
    config_dir = os.path.join(temp_data_dir, "УТ")
    excel_dir = os.path.join(config_dir, "excel")
    os.makedirs(excel_dir, exist_ok=True)
    dummy_excel_path = os.path.join(excel_dir, "test.xlsx")
    with open(dummy_excel_path, "w") as f:
        f.write("dummy content")

    section_name = "УТ"
    
    chunks = load_process_excel(config_dir, section_name, test_text_splitter)

    assert mock_unstructured_excel_loader.called # Проверяем, что UnstructuredExcelLoader был вызван
    assert len(chunks) > 0
    for chunk in chunks:
        assert "Excel table data" in chunk.page_content
        assert chunk.metadata["source_type"] == "excel"
        assert chunk.metadata["1c_section"] == section_name
        assert chunk.metadata["file_name"] == "test.xlsx"
        assert chunk.metadata["full_path"] == dummy_excel_path

    captured = capsys.readouterr()
    assert f"Обработка Excel: {dummy_excel_path}" in captured.out
    assert f"Загружено и разделено на" in captured.out

def test_load_process_faq(temp_data_dir, create_dummy_faq_file_for_loading, capsys):
    """
    Проверяет, что load_process_faq корректно загружает, обрабатывает и чанкует FAQ JSON.
    """
    config_dir = os.path.join(temp_data_dir, "УНФ")
    faq_file_path = create_dummy_faq_file_for_loading
    section_name = "УНФ"
    
    chunks = load_process_faq(config_dir, section_name, test_text_splitter)

    assert len(chunks) == 2 # Два вопроса в фикстуре, каждый станет чанком
    for chunk in chunks:
        assert chunk.metadata["source_type"] == "faq"
        assert chunk.metadata["1c_section"] == section_name
        assert chunk.metadata["file_name"] == "test_faq.json"
        assert chunk.metadata["full_path"] == faq_file_path
        assert "Вопрос: FAQ Вопрос" in chunk.page_content
        assert "Ответ: FAQ Ответ" in chunk.page_content
        assert "url" in chunk.metadata
        assert "date" in chunk.metadata
        assert "question" in chunk.metadata # Проверяем, что оригинальный вопрос сохранен

    captured = capsys.readouterr()
    assert f"Обработка FAQ: {faq_file_path}" in captured.out
    assert f"Загружено 2 FAQ, разделено на 2 чанков." in captured.out


def test_split_faq_json_by_program(temp_data_dir, create_dummy_json_faq, create_excel_mapping, capsys):
    """
    Проверяет, что split_faq_json_by_program корректно разбивает общий JSON на файлы по программам.
    """
    json_path = create_dummy_json_faq
    # create_excel_mapping также создает mappings.json с json_program_map
    
    # Предполагаем, что BASE_DATA_DIR мокнут на temp_data_dir
    split_faq_json_by_program(os.path.dirname(json_path), temp_data_dir)

    # Проверяем, что исходный файл удален
    assert not os.path.exists(json_path)

    # Проверяем, что целевые файлы созданы
    bp_faq_path = os.path.join(temp_data_dir, "БП", "faq", "from_import_general_faq.json")
    unf_faq_path = os.path.join(temp_data_dir, "УНФ", "faq", "from_import_general_faq.json") # Из mappings.json ПрограммаБ -> УНФ

    assert os.path.exists(bp_faq_path)
    assert os.path.exists(unf_faq_path)

    with open(bp_faq_path, "r", encoding="utf-8") as f:
        bp_data = json.load(f)
    assert len(bp_data) == 2
    assert bp_data[0]["question"] == "Вопрос А1"

    with open(unf_faq_path, "r", encoding="utf-8") as f:
        unf_data = json.load(f)
    assert len(unf_data) == 1
    assert unf_data[0]["question"] == "Вопрос Б1"

    captured = capsys.readouterr()
    assert f"Обнаружен общий FAQ-файл: {json_path}" in captured.out
    assert f"Сохранено 2 QA в {bp_faq_path}" in captured.out
    assert f"Сохранено 1 QA в {unf_faq_path}" in captured.out
    assert f"Удалён исходный файл: {json_path}" in captured.out


def test_main_function_integrates_all_steps(
    mock_embedding_model,
    mock_os_environ,
    temp_data_dir,
    temp_chroma_dir,
    create_dummy_excel,
    create_excel_mapping,
    create_dummy_json_faq,
    create_dummy_pdf, # Только для создания файла, мок лоадера сам его прочитает
    create_dummy_faq_file_for_loading,
    create_dummy_excel_for_loading,
    mock_pypdf_loader, # Мок, чтобы PyPDFLoader не пытался читать реальный PDF
    mock_unstructured_excel_loader, # Мок для Excel
    mock_shutil_rmtree # Проверяем, что shutil.rmtree вызывается
):
    """
    Проверяет сквозной процесс функции main: от загрузки до индексации.
    """
    # Дополнительно создаем пустую папку для Excel (для триггера load_process_excel)
    os.makedirs(os.path.join(temp_data_dir, "УТ", "excel"), exist_ok=True)
    # Создаем dummy markdown dir, хотя лоадер не реализован
    os.makedirs(os.path.join(temp_data_dir, "БП", "markdown"), exist_ok=True)
    # Создаем dummy database dir
    os.makedirs(os.path.join(temp_data_dir, "database"), exist_ok=True)
    # Создаем пустой файл SQLite для проверки, что load_process_sqlite не падает
    open(SQLITE_DB_PATH, 'a').close()

    # Запускаем основную функцию
    main()

    # Проверяем, что ChromaDB была создана
    assert os.path.exists(temp_chroma_dir)
    assert os.path.isdir(temp_chroma_dir)
    
    # Проверяем, что shutil.rmtree был вызван для очистки старой ChromaDB
    mock_shutil_rmtree.assert_called_with(temp_chroma_dir)

    # Проверяем, что Excel-файл был удален после обработки
    assert not os.path.exists(create_dummy_excel)
    
    # Проверяем, что исходный JSON FAQ был удален
    assert not os.path.exists(create_dummy_json_faq)

    # Проверяем, что функции загрузки данных были вызваны
    # (Это косвенно проверяется через создание файлов и assert'ы выше в отдельных тестах)
    # Здесь можно добавить более детальные проверки, например, через mock_pypdf_loader.call_count

    # Проверить, что в ChromaDB есть данные (например, путем загрузки)
    from langchain_community.vectorstores import Chroma
    from index_data import CHROMA_COLLECTION_NAME # Импортируем из вашего скрипта

    embedding_model = mock_embedding_model.return_value # Получаем мок-объект

    # Загружаем ChromaDB, чтобы проверить количество документов
    try:
        # Убедимся, что ChromaDB корректно загружается из персистентной директории
        # Если вы используете новую версию ChromaDB, возможно, потребуется более явное указание client
        # client = chromadb.PersistentClient(path=temp_chroma_dir)
        # vector_store = Chroma(client=client, embedding_function=embedding_model, collection_name=CHROMA_COLLECTION_NAME)
        vector_store_reloaded = Chroma(
            persist_directory=temp_chroma_dir,
            embedding_function=embedding_model,
            collection_name=CHROMA_COLLECTION_NAME
        )
        
        # Получаем все документы из коллекции
        retrieved_docs = vector_store_reloaded.get(include=['documents', 'metadatas'])
        
        # Ожидаем, что будут чанки от PDF (2), Excel (2), FAQ (2), и от Excel-импорта (3, после разбиения)
        # 2 (PDF) + 2 (FAQ from json) + 2 (Excel table) + 3 (Excel import to FAQ) = 9
        # Если есть чанкование по умолчанию для 1000/200, то количество чанков может быть больше.
        # Точное количество зависит от CHUNK_SIZE и CHUNK_OVERLAP и содержания фиктивных документов.
        # Для наших простых моков, каждый документ может стать одним чанком или несколькими, если большой.
        # PDF: 2 страницы (может стать 2 чанками)
        # FAQ (from_import_general_faq.json): 3 вопроса (может стать 3 чанками)
        # Excel (from_excel_SD_1cfresh.json): 3 записи (может стать 3 чанками)
        # FAQ (test_faq.json): 2 вопроса (может стать 2 чанками)
        # Excel (test_table.xlsx): 2 строки (может стать 2 чанками)
        # Итого: 2 + 3 + 3 + 2 + 2 = 12 чанков - это если каждый документ/запись становится 1 чанком.
        # text_splitter может разбить их по-другому, если контент длинный.
        # Для моков, где content короткий, вероятно, будет 1 документ = 1 чанк.
        
        # Проверяем количество документов
        # assert len(retrieved_docs['documents']) >= 12 # Или более точное число, если известны чанки.
        # Так как моки возвращают короткий текст, обычно это 1 чанк на документ/строку.
        expected_chunks = 2 + 3 + 3 + 2 + 2 # От PDF, общего FAQ, Excel-импорта, FAQ из папки, Excel из папки
        assert len(retrieved_docs['documents']) == expected_chunks

        # Проверяем наличие метаданных
        assert any("source_type" in meta for meta in retrieved_docs['metadatas'])
        assert any("1c_section" in meta for meta in retrieved_docs['metadatas'])

        # Проверяем, что есть чанки от разных источников
        source_types = set(meta.get("source_type") for meta in retrieved_docs['metadatas'])
        assert "pdf" in source_types
        assert "faq" in source_types
        assert "excel_kb" in source_types # Из преобразованного Excel
        assert "excel" in source_types # Из Excel файлов в папке /excel
        # assert "sqlite" in source_types # SQLite заглушен, поэтому не будет
    except Exception as e:
        pytest.fail(f"Ошибка при проверке ChromaDB после main: {e}")