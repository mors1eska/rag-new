# test_ui.py

import pytest
import os
import sys
import json
from unittest.mock import MagicMock, patch
import requests # Нужен для исключений ConnectionError, HTTPError
import importlib # Для надежной перезагрузки модулей

# --- Тестовые функции (для pytest) ---

# Эти функции будут обнаружены pytest и будут использовать фикстуры из conftest.py
# При запуске напрямую через main(), они будут вызываться с вручную созданными моками.


def test_submit_button_empty_question(mock_streamlit_st_and_app_ui, mock_requests_post):
    """
    Проверяет обработку пустого вопроса.
    """
    mock_st, app_ui_module = mock_streamlit_st_and_app_ui

    print("\n--- Запуск теста (Pytest/Standalone): test_submit_button_empty_question ---")

    mock_st.text_area.return_value = ""
    mock_st.button.side_effect = [True, False]

    # ИСПРАВЛЕНИЕ: Убедимся, что mock_requests_post.side_effect сброшен
    mock_requests_post.side_effect = None
    mock_requests_post.return_value = MagicMock() # Возвращаем простой мок-ответ, так как запрос не должен отправляться

    app_ui_module.main()

    mock_requests_post.assert_not_called()
    mock_st.warning.assert_called_with("Пожалуйста, введите ваш вопрос в текстовое поле выше.")
    print("Test `test_submit_button_empty_question` PASSED.")


def test_clear_button(mock_streamlit_st_and_app_ui):
    """
    Проверяет функциональность кнопки "Очистить".
    """
    mock_st, app_ui_module = mock_streamlit_st_and_app_ui

    print("\n--- Запуск теста (Pytest/Standalone): test_clear_button ---")

    mock_st.session_state.question_input = "Старый вопрос"
    mock_st.session_state.chat_history = [{"question": "Prev Q", "answer": "Prev A"}]
    mock_st.text_area.return_value = "Старый вопрос"

    mock_st.button.side_effect = [False, True]

    app_ui_module.main()

    assert mock_st.session_state.question_input == ""
    mock_st.rerun.assert_called_once()
    print("Test `test_clear_button` PASSED.")


def test_api_connection_error(mock_streamlit_st_and_app_ui, mock_requests_post):
    """Проверяет обработку ошибки подключения к API."""
    mock_st, app_ui_module = mock_streamlit_st_and_app_ui

    print("\n--- Запуск теста (Pytest/Standalone): test_api_connection_error ---")

    mock_st.text_area.return_value = "Тестовый вопрос"
    mock_st.button.side_effect = [True, False]

    # ИСПРАВЛЕНИЕ: Убедимся, что mock_requests_post.side_effect сброшен перед установкой
    mock_requests_post.side_effect = None
    mock_requests_post.side_effect = requests.exceptions.ConnectionError("Тестовая ошибка подключения")

    app_ui_module.main()

    mock_st.error.assert_called_once_with(
        "Ошибка: Не удалось подключиться к API по адресу http://127.0.0.1:8000/query/. Убедитесь, что FastAPI сервер запущен и доступен."
    )
    assert len(mock_st.session_state.chat_history) == 0
    print("Test `test_api_connection_error` PASSED.")


def test_api_http_error(mock_streamlit_st_and_app_ui, mock_requests_post):
    """Проверяет обработку HTTP ошибок от API."""
    mock_st, app_ui_module = mock_streamlit_st_and_app_ui

    print("\n--- Запуск теста (Pytest/Standalone): test_api_http_error ---")

    mock_st.text_area.return_value = "Тестовый вопрос"
    mock_st.button.side_effect = [True, False]

    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.reason = "Not Found"
    # ИСПРАВЛЕНИЕ: side_effect для raise_for_status, а не для самого post
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "404 Client Error: Not Found for url: http://127.0.0.1:8000/query/",
        response=mock_response
    )
    mock_response.json.return_value = {"detail": "Item not found"}

    # ИСПРАВЛЕНИЕ: Убедимся, что mock_requests_post.side_effect сброшен перед установкой
    mock_requests_post.side_effect = None
    mock_requests_post.return_value = mock_response # mock_requests_post должен вернуть mock_response

    app_ui_module.main()

    mock_st.error.assert_called_once()
    mock_st.error.assert_called_with("Ошибка HTTP от API: 404 Not Found")
    mock_st.json.assert_called_once_with({"detail": "Item not found"})
    assert len(mock_st.session_state.chat_history) == 0
    print("Test `test_api_http_error` PASSED.")


def test_api_json_decode_error(mock_streamlit_st_and_app_ui, mock_requests_post):
    """Проверяет обработку ошибок декодирования JSON ответа API."""
    mock_st, app_ui_module = mock_streamlit_st_and_app_ui

    print("\n--- Запуск теста (Pytest/Standalone): test_api_json_decode_error ---")

    mock_st.text_area.return_value = "Тестовый вопрос"
    mock_st.button.side_effect = [True, False]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "Это не JSON ответ"
    # ИСПРАВЛЕНИЕ: side_effect для json(), а не для самого post
    mock_response.json.side_effect = json.JSONDecodeError("Expecting value", "raw_data", 0)

    # ИСПРАВЛЕНИЕ: Убедимся, что mock_requests_post.side_effect сброшен перед установкой
    mock_requests_post.side_effect = None
    mock_requests_post.return_value = mock_response # mock_requests_post должен вернуть mock_response

    app_ui_module.main()

    mock_st.error.assert_called_once_with(
        "Ошибка: Не удалось декодировать JSON из ответа API. Ответ API не является валидным JSON."
    )
    mock_st.text.assert_any_call("Полученный ответ (или его часть):")
    mock_st.text.assert_any_call("Это не JSON ответ")
    assert len(mock_st.session_state.chat_history) == 0
    print("Test `test_api_json_decode_error` PASSED.")


# --- Вспомогательные функции для самостоятельного запуска ---

def setup_standalone_mocks():
    """
    Настраивает и возвращает мок-объекты для streamlit.st и requests.post,
    а также замокированный модуль app_ui.
    Эта функция дублирует логику фикстур из conftest.py для автономного запуска.
    """
    mock_st = MagicMock()
    
    # Инициализация всех моков для st
    mock_st.set_page_config = MagicMock()
    mock_st.title = MagicMock()
    mock_st.caption = MagicMock()
    
    # Настройка sidebar как контекстного менеджера
    mock_st.sidebar = MagicMock()
    mock_st.sidebar.__enter__.return_value = mock_st.sidebar
    mock_st.sidebar.__exit__.return_value = None
    mock_st.sidebar.header = MagicMock()
    # selectbox вернет "все" по умолчанию при первом вызове,
    # затем в тестах его return_value будет переопределяться
    mock_st.sidebar.selectbox = MagicMock(return_value="все") 
    mock_st.sidebar.markdown = MagicMock()
    mock_st.sidebar.subheader = MagicMock()
    mock_st.sidebar.info = MagicMock()

    mock_st.text_area = MagicMock()
    mock_st.button = MagicMock()
    mock_st.columns = MagicMock(return_value=[MagicMock(), MagicMock()])
    mock_st.expander = MagicMock()
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
    mock_st.rerun = MagicMock()
    mock_st.session_state = MagicMock() # Сброс session_state для каждого тестового прогона

    # Сохраняем оригинальные модули и удаляем их из sys.modules
    _original_streamlit = sys.modules.get('streamlit')
    _original_app_ui = sys.modules.get('app_ui')
    
    if 'streamlit' in sys.modules:
        del sys.modules['streamlit']
    if 'app_ui' in sys.modules:
        del sys.modules['app_ui']

    # Добавляем родительскую директорию в sys.path для импорта app_ui
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    print(f"DEBUG: Added {project_root} to sys.path")

    # Патчим сам модуль 'streamlit' в sys.modules
    streamlit_patch = patch.dict(sys.modules, {'streamlit': mock_st})
    streamlit_patch.start() # Запускаем патч вручную

    # Импортируем app_ui. Используем importlib.reload для надежности.
    try:
        # Важно: используем importlib.import_module, если модуль еще не был загружен.
        # Если он уже в sys.modules (например, из-за предыдущих тестов), то reload.
        if 'app_ui' in sys.modules:
            mocked_app_ui_module = importlib.reload(sys.modules['app_ui'])
        else:
            mocked_app_ui_module = importlib.import_module('app_ui')
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to import or reload app_ui: {e}")
        # Если импорт app_ui не удался, возвращаем None для всех связанных объектов,
        # чтобы ошибка распаковки была явно поймана.
        return None, None, None, streamlit_patch, None, _original_streamlit, _original_app_ui
    
    # Принудительно устанавливаем st в app_ui.py на наш мок
    mocked_app_ui_module.st = mock_st 
    
    requests_post_patch = patch('app_ui.requests.post')
    mock_requests_post = requests_post_patch.start()

    # Возвращаем все необходимые моки, модуль и объекты патчей для очистки
    return mock_st, mocked_app_ui_module, mock_requests_post, streamlit_patch, requests_post_patch, _original_streamlit, _original_app_ui


@pytest.mark.skip(reason="Эта функция предназначена для запуска как самостоятельный скрипт")
def run_tests_standalone():
    """
    Основная функция для запуска всех тестов как самостоятельного скрипта.
    """
    print("Запуск всех UI-тестов как самостоятельного скрипта...")
    
    # Настраиваем моки.
    # Если setup_standalone_mocks возвращает None из-за ошибки импорта,
    # мы должны корректно обработать это.
    mocks_and_patches = setup_standalone_mocks()
    if mocks_and_patches[0] is None: # Проверяем mock_st, который будет None при ошибке импорта app_ui
        print("FATAL: Не удалось настроить моки для автономного запуска. Тесты не будут выполнены.")
        # Выполняем очистку, если патчи были запущены, но импорт app_ui провалился
        if mocks_and_patches[3]: # streamlit_patch
             mocks_and_patches[3].stop()
        if mocks_and_patches[4]: # requests_post_patch
             mocks_and_patches[4].stop()
        # Восстанавливаем оригинальные модули
        if mocks_and_patches[5]: # _original_streamlit
            sys.modules['streamlit'] = mocks_and_patches[5]
        if mocks_and_patches[6]: # _original_app_ui
            sys.modules['app_ui'] = mocks_and_patches[6]
        return # Выход из функции, так как моки не настроены
    
    mock_st, app_ui_module, mock_requests_post, streamlit_patch, requests_post_patch, _original_streamlit, _original_app_ui = mocks_and_patches

    tests_to_run = [
        test_initial_ui_elements,
        test_submit_button_success,
        test_submit_button_empty_question,
        test_clear_button,
        test_api_connection_error,
        test_api_http_error,
        test_api_json_decode_error,
    ]

    for test_func in tests_to_run:
        try:
            print(f"\n--- Сброс моков перед тестом: {test_func.__name__} ---")
            # Сброс моков перед каждым тестом для изоляции
            # IMPORTANT: reset_mock(side_effect=True) сбрасывает side_effect
            # (и return_value) только на самом моке, но не на его подмоках
            mock_st.reset_mock(return_value=True, side_effect=True) 
            # Явно сбросим мок sidebar и его return_value
            mock_st.sidebar = MagicMock()
            mock_st.sidebar.__enter__.return_value = mock_st.sidebar
            mock_st.sidebar.__exit__.return_value = None
            # ИСПРАВЛЕНИЕ: Возвращаем "все" по умолчанию для selectbox после сброса
            mock_st.sidebar.selectbox = MagicMock(return_value="все") 

            # ИСПРАВЛЕНИЕ: Сброс и возврат для requests.post
            if hasattr(mock_requests_post, 'reset_mock'):
                mock_requests_post.reset_mock(return_value=True, side_effect=True)
                mock_requests_post.side_effect = None # Очистить побочные эффекты
                mock_requests_post.return_value = MagicMock() # Вернуть простой мок-объект по умолчанию
            
            # Перенастраиваем остальные базовые моки, которые могли быть сброшены
            mock_st.set_page_config = MagicMock()
            mock_st.title = MagicMock()
            mock_st.caption = MagicMock()

            mock_st.sidebar.header = MagicMock()
            mock_st.sidebar.markdown = MagicMock()
            mock_st.sidebar.subheader = MagicMock()
            mock_st.sidebar.info = MagicMock()

            mock_st.text_area = MagicMock(return_value="") # Значение по умолчанию для текстового поля
            mock_st.button = MagicMock(side_effect=[False, False]) # Обе кнопки не нажаты по умолчанию
            mock_st.columns = MagicMock(return_value=[MagicMock(), MagicMock()])
            
            mock_expander = MagicMock()
            mock_expander.__enter__.return_value = MagicMock()
            mock_expander.__exit__.return_value = None
            mock_st.expander = MagicMock(return_value=mock_expander) # Возвращаем мок для контекстного менеджера expander

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
            mock_st.rerun = MagicMock()
            mock_st.session_state = MagicMock() # Сброс session_state для каждого тестового прогона

            # Убеждаемся, что app_ui_module.st всегда указывает на наш текущий mock_st
            app_ui_module.st = mock_st


            # Определяем, какие аргументы принимает функция
            import inspect
            sig = inspect.signature(test_func)
            params = {}
            if 'mock_streamlit_st_and_app_ui' in sig.parameters:
                # Для Pytest-совместимых функций, передаем кортеж как фикстуру
                params['mock_streamlit_st_and_app_ui'] = (mock_st, app_ui_module)
            if 'mock_requests_post' in sig.parameters:
                params['mock_requests_post'] = mock_requests_post
            
            test_func(**params)
        except AssertionError as e:
            print(f"Test `{test_func.__name__}` FAILED: {e}")
        except Exception as e:
            print(f"Test `{test_func.__name__}` encountered an unexpected error: {e}")
            import traceback
            traceback.print_exc() # Вывод полного стека для отладки
    
    # Очистка: останавливаем все патчи и восстанавливаем оригинальные модули
    requests_post_patch.stop()
    streamlit_patch.stop()
    if _original_streamlit:
        sys.modules['streamlit'] = _original_streamlit
    if _original_app_ui:
        sys.modules['app_ui'] = _original_app_ui

    print("\nВсе UI-тесты завершены.")


# --- Точка входа для запуска как Python-скрипт ---
if __name__ == "__main__":
    run_tests_standalone()