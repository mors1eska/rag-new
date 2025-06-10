# conftest.py
import importlib
import pytest
import os
import sys
import shutil
import json
import pandas as pd
from unittest.mock import patch, MagicMock # Объединенный импорт patch и MagicMock. ТОЛЬКО ЭТА СТРОКА!
from langchain.schema import Document

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
    os.makedirs(TEST_CHROMA_PERSIST_DIR, exist_ok=True)
    os.makedirs(os.path.join(TEST_BASE_DATA_DIR, "БП", "pdfs"), exist_ok=True)
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
    with patch('index_data.PyPDFLoader') as MockLoader:
        mock_instance = MockLoader.return_value
        mock_instance.load.return_value = [
            # Верните 'page' как 0-индексированную, как это делает реальный PyPDFLoader.
            # Ваша функция load_process_pdfs должна будет добавить +1 к этому значению.
            Document(page_content="Test PDF content on page 1.", metadata={'page': 0}),
            Document(page_content="Test PDF content on page 2.", metadata={'page': 1})
        ]
        yield MockLoader

@pytest.fixture
def mock_unstructured_excel_loader():
    with patch('index_data.UnstructuredExcelLoader') as MockLoader:
        mock_instance = MockLoader.return_value
        mock_instance.load.return_value = [
            Document(page_content="Excel table data row 1", metadata={'row': 0}),
            Document(page_content="Excel table data row 2", metadata={'row': 1})
        ]
        yield MockLoader

@pytest.fixture(autouse=True)
def mock_shutil_rmtree():
    """
    Мокает shutil.rmtree, чтобы она не удаляла реальные файлы во время тестов.
    Однако, в данном случае мы полагаемся на setup_test_environment для очистки.
    Эту фикстуру можно использовать, если нужно проверить, что rmtree вызывается, но не выполнять реальное удаление.
    """
    with patch('shutil.rmtree') as mock_rmtree:
        yield mock_rmtree

@pytest.fixture
def create_empty_dir(temp_data_dir):
    """Создает и возвращает путь к пустой директории."""
    empty_dir = os.path.join(temp_data_dir, "empty_content")
    os.makedirs(empty_dir, exist_ok=True)
    return empty_dir

@pytest.fixture
def create_non_existent_dir(temp_data_dir):
    """Возвращает путь к несуществующей директории."""
    return os.path.join(temp_data_dir, "non_existent_dir")

@pytest.fixture
def create_excel_with_empty_rows(temp_data_dir):
    """Создает Excel-файл с пустыми строками для convert_excel_to_faq_format."""
    excel_import_dir = os.path.join(temp_data_dir, "import")
    os.makedirs(excel_import_dir, exist_ok=True)
    excel_path = os.path.join(excel_import_dir, "kb_import_empty_rows.xlsx")

    df_data = {
        "Наименование": ["Вопрос 1", "", "Вопрос 3"],
        "Номер": ["111", "222", "333"],
        "Ссылка": ["link1", "", "link3"],
        "Описание": ["Ответ 1", "", "Ответ 3"],
        "Компоненты": ["1С:БП", "Неизвестно", "1С:УНФ"],
        "Разделы": ["Базовый", "Пусто", "Прочее"]
    }
    df = pd.DataFrame(df_data)
    df.to_excel(excel_path, index=False)
    return excel_path

@pytest.fixture
def create_faq_with_missing_fields(temp_data_dir):
    """Создает FAQ JSON с отсутствующими/пустыми полями для load_process_faq."""
    faq_dir = os.path.join(temp_data_dir, "БП", "faq")
    os.makedirs(faq_dir, exist_ok=True)
    faq_path = os.path.join(faq_dir, "malformed_faq.json")

    faq_data = [
        {"question": "Valid Q1", "answer": "Valid A1", "url": "url1"},
        {"question": "Q2 no answer", "url": "url2"}, # Нет ответа
        {"answer": "A3 no question", "url": "url3"}, # Нет вопроса
        {"question": "Q4 empty answer", "answer": ""}, # Пустой ответ
        {"question": "", "answer": "A5 empty question"}, # Пустой вопрос
        {"question": "Valid Q6", "answer": "Valid A6"} # Без URL
    ]
    with open(faq_path, "w", encoding="utf-8") as f:
        json.dump(faq_data, f, ensure_ascii=False, indent=2)
    return faq_path

@pytest.fixture
def create_excel_for_metadata_test(temp_data_dir):
    """Создает Excel-файл для тщательной проверки метаданных."""
    excel_dir = os.path.join(temp_data_dir, "БП", "excel")
    os.makedirs(excel_dir, exist_ok=True)
    excel_path = os.path.join(excel_dir, "metadata_excel.xlsx")
    
    df_data = {
        "ColA": ["Content1"],
        "ColB": ["Content2"]
    }
    df = pd.DataFrame(df_data)
    df.to_excel(excel_path, index=False)
    return excel_path

@pytest.fixture
def create_faq_for_metadata_test(temp_data_dir):
    """Создает FAQ JSON для тщательной проверки метаданных."""
    faq_dir = os.path.join(temp_data_dir, "УНФ", "faq")
    os.makedirs(faq_dir, exist_ok=True)
    faq_path = os.path.join(faq_dir, "metadata_faq.json")
    
    faq_data = [
        {
            "question": "Meta Question 1",
            "answer": "Meta Answer 1",
            "url": "http://meta.com/1",
            "date": "2024-05-15",
            "code": "CODE001",
            "subsection": "Sub A"
        }
    ]
    with open(faq_path, "w", encoding="utf-8") as f:
        json.dump(faq_data, f, ensure_ascii=False, indent=2)
    return faq_path

# Мок UnstructuredMarkdownLoader
@pytest.fixture
def mock_unstructured_markdown_loader():
    with patch('index_data.UnstructuredMarkdownLoader') as MockLoader:
        mock_instance = MockLoader.return_value
        mock_instance.load.return_value = [
            Document(page_content="Markdown content 1.", metadata={'header': 'Header 1'}),
            Document(page_content="Markdown content 2.", metadata={'header': 'Header 2'})
        ]
        yield MockLoader

# Мок SQLDatabaseLoader и SQLDatabase (для тестирования load_process_sqlite)
@pytest.fixture
def mock_sqlite_loading():
    # Мокаем зависимости sqlalchemy, которые могут быть в load_process_sqlite
    with patch('index_data.SQLDatabaseLoader') as MockLoader, \
         patch('index_data.SQLDatabase') as MockSQLDatabase, \
         patch('index_data.create_engine') as MockCreateEngine, \
         patch('index_data.text') as MockText:

        # Мок для SQLDatabaseLoader.load()
        mock_loader_instance = MockLoader.return_value
        mock_loader_instance.load.return_value = [
            Document(page_content="SQLite content 1.", metadata={'id': 1, '1c_section': 'БП', 'source_info': 'DB A'}),
            Document(page_content="SQLite content 2.", metadata={'id': 2, '1c_section': 'УНФ', 'source_info': 'DB B'})
        ]

        # Мок для использования sqlalchemy.text (если скрипт использует)
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_result = MagicMock()

        # Настройка возвращаемых значений для имитации запроса
        mock_result.all.return_value = [
            MagicMock(id=1, content_column="SQLite content 1.", _1c_section_column='БП', source_info='DB A'), # Исправлено: 1c_section_column на _1c_section_column
            MagicMock(id=2, content_column="SQLite content 2.", _1c_section_column='УНФ', source_info='DB B')  # Исправлено: 1c_section_column на _1c_section_column
        ]
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        MockCreateEngine.return_value = mock_engine

        yield # <-- Этот yield должен быть внутри блока 'with'

##### Для UI

@pytest.fixture
def mock_streamlit():
    """Создает полноценный мок Streamlit с правильной структурой."""
    mock_st = MagicMock()
    
    # Настройка session_state
    mock_st.session_state = MagicMock()
    mock_st.session_state.chat_history = []
    mock_st.session_state.question_input = ""
    
    # Настройка sidebar как контекстного менеджера
    mock_sidebar = MagicMock()
    mock_sidebar.__enter__.return_value = mock_sidebar  # Ключевая строка!
    mock_sidebar.__exit__.return_value = None
    
    # Настройка методов sidebar
    mock_sidebar.header = MagicMock()
    mock_sidebar.selectbox = MagicMock(return_value="все")
    mock_sidebar.markdown = MagicMock()
    mock_sidebar.subheader = MagicMock()
    mock_sidebar.info = MagicMock()
    
    mock_st.sidebar = mock_sidebar
    
    # Остальные моки
    mock_st.set_page_config = MagicMock()
    mock_st.title = MagicMock()
    mock_st.caption = MagicMock()
    mock_st.text_area = MagicMock(return_value="")
    mock_st.button = MagicMock(side_effect=[False, False])  # Обе кнопки не нажаты
    mock_st.columns = MagicMock(return_value=[MagicMock(), MagicMock()])
    
    # Expander
    mock_expander = MagicMock()
    mock_expander.__enter__.return_value = MagicMock()
    mock_expander.__exit__.return_value = None
    mock_st.expander = MagicMock(return_value=mock_expander)
    
    return mock_st

@pytest.fixture
def mock_streamlit_st_and_app_ui(mock_streamlit):
    """Фикстура, возвращающая мокированный Streamlit и модуль app_ui."""
    mock_st = mock_streamlit
    
    # Удаляем оригинальные модули из кэша
    import sys
    original_streamlit = sys.modules.get('streamlit')
    original_app_ui = sys.modules.get('app_ui')
    
    if 'streamlit' in sys.modules:
        del sys.modules['streamlit']
    if 'app_ui' in sys.modules:
        del sys.modules['app_ui']

    # Патчим модули
    with patch.dict('sys.modules', {'streamlit': mock_st}):
        import app_ui
        importlib.reload(app_ui)
        app_ui.st = mock_st  # Важно!
        
        yield mock_st, app_ui
    
    # Восстанавливаем оригинальные модули
    if original_streamlit:
        sys.modules['streamlit'] = original_streamlit
    if original_app_ui:
        sys.modules['app_ui'] = original_app_ui

@pytest.fixture
def mock_requests_post():
    """
    Мокирует requests.post для перехвата и контроля HTTP-запросов.
    """
    # Важно: замените 'app_ui' на фактическое имя вашего файла Streamlit UI
    with patch('app_ui.requests.post') as mock_post:
        yield mock_post # Возвращаем объект-мок post для использования в тестах
