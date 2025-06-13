from rag_module import main_query

if __name__ == "__main__":
    # пример CLI-запуска:
    resp = main_query("Как настроить резервное копирование в УНФ?", section_filter="УНФ")
    print(resp["answer"])
    for src in resp["sources"]:
        print("-", src["file_name"] or src["url"])