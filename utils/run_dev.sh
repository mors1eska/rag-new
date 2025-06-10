#!/bin/bash
cd "$(dirname "$0")/.."  # перейти в корень проекта

# ✅ Активируем виртуальное окружение
source .venv/bin/activate

# 🛑 Завершаем старые процессы, если запущены
echo "🔎 Проверка запущенных FastAPI процессов..."
PORTS=(8000 8502)
for PORT in "${PORTS[@]}"; do
  PID=$(lsof -ti tcp:$PORT)
  if [ -n "$PID" ]; then
    echo "🛑 Завершаем процесс на порту $PORT (PID=$PID)..."
    kill -9 $PID
  else
    echo "✅ Порт $PORT свободен."
  fi
done

# 🚀 Запускаем FastAPI API (main_api.py) на 8000
echo "🚀 Запускаем FastAPI (main_api.py)..."
uvicorn main_api:app --host 127.0.0.1 --port 8000 --reload &
API_PID=$!

# 🚀 Запускаем сервер для статики (serve_static_files.py) на 8502
echo "🧩 Запускаем сервер раздачи статики (serve_static_files.py)..."
uvicorn serve_static_files:app --host 127.0.0.1 --port 8600 --reload &
STATIC_PID=$!

# ⏳ Подождём немного
sleep 2

# 🖥️ Запускаем Streamlit UI
echo "🖥️  Запускаем Streamlit UI (app_ui.py)..."
streamlit run app_ui.py

# 🛑 После закрытия UI — останавливаем API и сервер статики
echo "🛑 Остановка FastAPI и сервера статики..."
kill $API_PID
kill $STATIC_PID