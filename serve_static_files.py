from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI(title="Статический сервер для PDF, Excel, Markdown")

# Путь к исходным данным
DATA_ROOT = "data"

# Подключаем папки для каждой конфигурации (УНФ, БП, ЗУП и т.д.)
for config_name in os.listdir(DATA_ROOT):
    config_path = os.path.join(DATA_ROOT, config_name)
    if not os.path.isdir(config_path):
        continue

    for subfolder in ["pdfs", "excel", "markdown"]:
        sub_path = os.path.join(config_path, subfolder)
        if os.path.isdir(sub_path):
            mount_path = f"/{subfolder}/{config_name}"
            print(f"📂 Монтируем {sub_path} на {mount_path}")
            app.mount(mount_path, StaticFiles(directory=sub_path), name=f"{subfolder}_{config_name}")


@app.get("/")
async def root():
    return {"message": "Статический сервер работает. Используйте /pdfs/, /excel/, /markdown/"}