# conftest.py
import pytest
import os
import shutil
import json
import pandas as pd
from unittest.mock import patch
from langchain.schema import Document 
from unittest.mock import patch, MagicMock

# Определяем базовые пути, как в вашем скрипте, но для тестовых нужд
TEST_BASE_DATA_DIR = "test_data"
TEST_CHROMA_PERSIST_DIR = "test_db_chroma"
TEST_SQLITE_DB_PATH = os.path.join(TEST_BASE_DATA_DIR, "database", "test_db.db")

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    Создает и очищает тестовую среду до и после всех тестов сессии.
    """
    # Удаляем все тестовые директории перед запуском тестов
    if os.path.exists(TEST_BASE_DATA_DIR):
        shutil.rmtree(TEST_BASE_DATA_DIR)
    if os.path.exists(TEST_CHROMA_PERSIST_DIR):
        shutil.rmtree(TEST_CHROMA_PERSIST_DIR)

    # Создаем необходимые директории
    os.makedirs(TEST_BASE_DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(TEST_BASE_DATA_DIR, "import"), exist_ok=True)
    os.makedirs(os.path.join(TEST_BASE_DATA_DIR, "БП", "pdfs"), exist_ok=True)
    os.makedirs(TEST_CHROMA_PERSIST_DIR, exist_ok=True)
    os.makedirs(os.path.join(TEST_BASE_DATA_DIR, "УНФ", "faq"), exist_ok=True)
    os.makedirs(os.path.join(TEST_BASE_DATA_DIR, "УТ", "excel"), exist_ok=True)
    os.makedirs(os.path.join(TEST_BASE_DATA_DIR, "database"), exist_ok=True)

    yield

    # Очистка после всех тестов
    if os.path.exists(TEST_BASE_DATA_DIR):
        shutil.rmtree(TEST_BASE_DATA_DIR)
    if os.path.exists(TEST_CHROMA_PERSIST_DIR):
        shutil.rmtree(TEST_CHROMA_PERSIST_DIR)

@pytest.fixture
def temp_data_dir():
    """
    Фикстура для временной директории данных для каждого теста.
    """
    # Просто возвращаем путь, очистка будет выполнена setup_test_environment
    return TEST_BASE_DATA_DIR

@pytest.fixture
def temp_chroma_dir():
    """
    Фикстура для временной директории ChromaDB для каждого теста.
    """
    return TEST_CHROMA_PERSIST_DIR

@pytest.fixture
def mock_embedding_model():
    """
    Мок-объект для модели эмбеддингов.
    """
    class MockEmbeddings:
        def embed_documents(self, texts):
            # Возвращаем фиктивные эмбеддинги
            return [[float(i) for i in range(768)] for _ in texts]
        def embed_query(self, text):
            return [float(i) for i in range(768)]

# Захватываем объект-мок, созданный patch, и возвращаем его
    with patch('index_data.get_embedding_model', return_value=MockEmbeddings()) as mock_get_embedding_model_func:
        yield mock_get_embedding_model_func # <-- Теперь явно возвращаем объект-мок


@pytest.fixture
def mock_os_environ():
    """
    Мок-объект для переменных окружения (например, OPENAI_API_KEY).
    """
    with patch.dict(os.environ, {}, clear=True): # Очищаем и устанавливаем пустой dict
        yield

@pytest.fixture
def create_dummy_pdf(temp_data_dir):
    """
    Создает фиктивный PDF-файл.
    """
    pdf_dir = os.path.join(temp_data_dir, "БП", "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, "test_doc.pdf")
    # Создаем очень простой фиктивный PDF, который PyPDFLoader сможет обработать
    # В реальных тестах лучше использовать библиотеку для создания валидных PDF
    # Или мокать PyPDFLoader
    with open(pdf_path, "w") as f:
        f.write("Это тестовый PDF-файл.") # Не совсем PDF, но для мока PyPDFLoader сойдет
    return pdf_path

@pytest.fixture
def create_dummy_excel(temp_data_dir):
    """
    Создает фиктивный Excel-файл для функции convert_excel_to_faq_format.
    """
    excel_import_dir = os.path.join(temp_data_dir, "import")
    os.makedirs(excel_import_dir, exist_ok=True)
    excel_path = os.path.join(excel_import_dir, "kb_import.xlsx")
    
    df_data = {
        "Наименование": ["Тестовый вопрос 1", "Тестовый вопрос 2"],
        "Номер": ["123", "456"],
        "Ссылка": ["http://link1.com", "http://link2.com"],
        "Описание": ["Тестовый ответ 1", "Тестовый ответ 2"],
        "Компоненты": ["1С:БП", "1С:УНФ/1С:БП"],
        "Разделы": ["Базовый", "Прочее"]
    }
    df = pd.DataFrame(df_data)
    df.to_excel(excel_path, index=False)
    return excel_path

@pytest.fixture
def create_excel_mapping(temp_data_dir):
    """
    Создает фиктивный mapping.json для Excel-файла.
    """
    mapping_dir = os.path.join(temp_data_dir, "import")
    os.makedirs(mapping_dir, exist_ok=True)
    mapping_path = os.path.join(mapping_dir, "mappings.json")
    
    mapping_data = {
        "kb_import.xlsx": {
            "source_type": "excel_kb",
            "program_map": {
                "1С:БП": "БП",
                "1С:УНФ": "УНФ"
            }
        },
        "json_program_map": {
            "ПрограммаА": "БП",
            "ПрограммаБ": "УНФ"
        }
    }
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, ensure_ascii=False, indent=2)
    return mapping_path

@pytest.fixture
def create_dummy_json_faq(temp_data_dir):
    """
    Создает фиктивный JSON FAQ файл для split_faq_json_by_program.
    """
    json_import_dir = os.path.join(temp_data_dir, "import")
    os.makedirs(json_import_dir, exist_ok=True)
    json_path = os.path.join(json_import_dir, "general_faq.json")

    faq_data = [
        {
            "program": "ПрограммаА",
            "questions": [
                {"question": "Вопрос А1", "answer": "Ответ А1", "url": "url_a1"},
                {"question": "Вопрос А2", "answer": "Ответ А2", "url": "url_a2"}
            ]
        },
        {
            "program": "ПрограммаБ",
            "questions": [
                {"question": "Вопрос Б1", "answer": "Ответ Б1", "url": "url_b1"}
            ]
        }
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(faq_data, f, ensure_ascii=False, indent=2)
    return json_path

@pytest.fixture
def create_dummy_faq_file_for_loading(temp_data_dir):
    """
    Создает фиктивный FAQ JSON файл в папке конфигурации для load_process_faq.
    """
    faq_dir = os.path.join(temp_data_dir, "УНФ", "faq")
    os.makedirs(faq_dir, exist_ok=True)
    faq_path = os.path.join(faq_dir, "test_faq.json")
    
    faq_data = [
        {"question": "FAQ Вопрос 1", "answer": "FAQ Ответ 1", "url": "faq_url1", "date": "2023-01-01"},
        {"question": "FAQ Вопрос 2", "answer": "FAQ Ответ 2", "url": "faq_url2", "date": "2023-01-02"}
    ]
    with open(faq_path, "w", encoding="utf-8") as f:
        json.dump(faq_data, f, ensure_ascii=False, indent=2)
    return faq_path

@pytest.fixture
def create_dummy_excel_for_loading(temp_data_dir):
    """
    Создает фиктивный Excel-файл в папке конфигурации для load_process_excel.
    """
    excel_dir = os.path.join(temp_data_dir, "УТ", "excel")
    os.makedirs(excel_dir, exist_ok=True)
    excel_path = os.path.join(excel_dir, "test_table.xlsx")
    
    df_data = {
        "ColumnA": ["Value1", "Value2"],
        "ColumnB": ["Value3", "Value4"]
    }
    df = pd.DataFrame(df_data)
    df.to_excel(excel_path, index=False)
    return excel_path

# Здесь мы мокаем глобальные переменные скрипта
@pytest.fixture(autouse=True)
def mock_global_paths():
    with patch('index_data.BASE_DATA_DIR', TEST_BASE_DATA_DIR), \
         patch('index_data.CHROMA_PERSIST_DIR', TEST_CHROMA_PERSIST_DIR), \
         patch('index_data.SQLITE_DB_PATH', TEST_SQLITE_DB_PATH), \
         patch('index_data.PROGRAM_TO_FOLDER_MAP', {"1С:БП": "БП", "1С:УНФ": "УНФ", "1С:Управление торговлей": "УТ"}):
        yield

# Мок PyPDFLoader
@pytest.fixture
def mock_pypdf_loader():
    with patch('index_data.PyPDFLoader') as MockLoader: # Убедитесь, что 'index_data' здесь правильное имя вашего скрипта
        mock_instance = MockLoader.return_value
        mock_instance.load.return_value = [
            # Верните 'page' как 0-индексированную, как это делает реальный PyPDFLoader.
            # Ваша функция load_process_pdfs должна будет добавить +1 к этому значению.
            Document(page_content="Test PDF content on page 1.", metadata={'page': 0}),
            Document(page_content="Test PDF content on page 2.", metadata={'page': 1})
        ]
        yield MockLoader

# Мок UnstructuredExcelLoader
@pytest.fixture
def mock_unstructured_excel_loader():
    with patch('index_data.UnstructuredExcelLoader') as MockLoader:
        mock_instance = MockLoader.return_value
        mock_instance.load.return_value = [
            Document(page_content="Excel table data row 1", metadata={'row': 0}),
            Document(page_content="Excel table data row 2", metadata={'row': 1})
        ]
        yield MockLoader

# Мок shutil.rmtree
@pytest.fixture(autouse=True)
def mock_shutil_rmtree():
    """
    Мокает shutil.rmtree, чтобы она не удаляла реальные файлы во время тестов.
    Однако, в данном случае мы полагаемся на setup_test_environment для очистки.
    Эту фикстуру можно использовать, если нужно проверить, что rmtree вызывается, но не выполнять реальное удаление.
    """
    with patch('shutil.rmtree') as mock_rmtree:
        yield mock_rmtree



##### Для UI


@pytest.fixture
def mock_streamlit_st_and_app_ui():
    """
    Мокирует модуль streamlit.st для тестирования UI.
    Возвращает объект-мок, который можно использовать для настройки поведения st.* функций
    и проверки их вызовов.
    """
    mock_st = MagicMock()
    # Мокирование часто используемых функций Streamlit
    mock_st.set_page_config = MagicMock()
    mock_st.title = MagicMock()
    mock_st.caption = MagicMock()

    # Мокируем st.sidebar как контекстный менеджер
    mock_st.sidebar = MagicMock()
    mock_st.sidebar.__enter__.return_value = mock_st.sidebar # Важно: __enter__ возвращает сам мок
    mock_st.sidebar.__exit__.return_value = None # Стандартное поведение для контекстных менеджеров

    # Теперь мокируем методы непосредственно на mock_st.sidebar
    mock_st.sidebar.header = MagicMock()
    mock_st.sidebar.selectbox = MagicMock(return_value="все") # По умолчанию выбран "все"
    mock_st.sidebar.markdown = MagicMock()
    mock_st.sidebar.subheader = MagicMock()
    mock_st.sidebar.info = MagicMock()
    mock_st.text_area = MagicMock()
    mock_st.button = MagicMock() # Поведение кнопки будет задаваться в каждом тесте
    mock_st.columns = MagicMock(return_value=[MagicMock(), MagicMock()]) # Для st.columns
    mock_st.expander = MagicMock()
    # Важно: для контекстных менеджеров (как st.expander, st.spinner) нужно мокировать __enter__ и __exit__
    mock_st.expander.return_value.__enter__.return_value = MagicMock()
    mock_st.expander.return_value.__exit__.return_value = None
    mock_st.markdown = MagicMock()
    mock_st.subheader = MagicMock()
    mock_st.write = MagicMock()
    mock_st.info = MagicMock()
    mock_st.warning = MagicMock()
    mock_st.error = MagicMock()
    mock_st.json = MagicMock()
    mock_st.spinner = MagicMock()
    mock_st.spinner.return_value.__enter__.return_value = MagicMock()
    mock_st.spinner.return_value.__exit__.return_value = None
    mock_st.text = MagicMock()
    mock_st.rerun = MagicMock() # Для st.rerun

    # Мокируем st.session_state напрямую
    mock_st.session_state = MagicMock() 

    # --- Самый надежный способ мокирования Streamlit (попытка) ---
    # Удаляем 'streamlit' из sys.modules, чтобы он был принудительно переимпортирован
    # после того, как наш мок будет активен.
    import sys
    _original_streamlit = None
    if 'streamlit' in sys.modules:
        _original_streamlit = sys.modules['streamlit'] # Сохраняем оригинал
        del sys.modules['streamlit'] # Удаляем из кэша

    _original_app_ui = None
    # Если 'app_ui' также уже был импортирован, удаляем его тоже,
    # чтобы он мог быть переимпортирован и увидеть замокированный 'st'.
    # Важно: замените 'app_ui' на фактическое имя вашего файла Streamlit UI
    if 'app_ui' in sys.modules:
        _original_app_ui = sys.modules['app_ui'] # Сохраняем оригинал
        del sys.modules['app_ui'] # Удаляем из кэша

    # Патчим сам модуль 'streamlit' (надежнее, чем только app_ui.st)
    with patch.dict(sys.modules, {'streamlit': mock_st}):
        import app_ui as mocked_app_ui_module # Импортируем app_ui здесь, чтобы он увидел мокированный Streamlit
    
     # Восстанавливаем оригинальные модули после теста
    if _original_streamlit:
        sys.modules['streamlit'] = _original_streamlit
    if _original_app_ui:
        sys.modules['app_ui'] = _original_app_ui


@pytest.fixture
def mock_requests_post():
    """
    Мокирует requests.post для перехвата и контроля HTTP-запросов.
    """
    # Важно: замените 'app_ui' на фактическое имя вашего файла Streamlit UI
    with patch('app_ui.requests.post') as mock_post:
        yield mock_post # Возвращаем объект-мок post для использования в тестах
