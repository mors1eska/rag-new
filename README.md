# LangChain RAG System for 1C Knowledge Base

## 1. Project Overview

This project implements a Retrieval Augmented Generation (RAG) system using Python and the LangChain library. Its primary purpose is to provide a Question & Answering interface over a knowledge base related to 1C software products (e.g., УНФ, БП, ERP). Users can ask questions in natural language, and the system will retrieve relevant information from indexed documents (PDFs, Markdown files, Excel sheets, and SQLite databases) and generate an answer along with source citations.

**Key Technologies:**
*   **LangChain:** Core framework for building the RAG pipeline.
*   **Python:** Programming language.
*   **OpenAI / Hugging Face Models:** For text embeddings and language generation.
*   **ChromaDB:** Vector store for storing and retrieving document embeddings.
*   **Various Document Loaders:** For PDF, Markdown, Excel, and SQLite data.

## 2. Project Structure

```
.
├── data/
│   ├── pdfs/
│   │   ├── УНФ/
│   │   │   └── example_unf_doc1.pdf
│   │   │   └── example_unf_doc2.pdf
│   │   ├── БП/
│   │   │   └── example_bp_doc1.pdf
│   │   └── ERP/
│   │       └── example_erp_doc1.pdf
│   ├── markdown/
│   │   └── (similar structure for .md files if used)
│   ├── excel/
│   │   └── (similar structure for .xlsx files if used)
│   └── database/
│       └── 1c_knowledge_base.db (if SQLite is used)
├── db_chroma/
│   └── (ChromaDB files will be created here after indexing)
├── .venv/
│   └── (Python virtual environment)
├── .env
├── index_data.py
├── query_data.py
├── requirements.txt
├── setup_env.sh
└── README.md
```

## 3. Step 1: Preparation and Project Setup

1.  **Create Project Folder:**
    Create a main directory for your project (e.g., `my_1c_rag_project`). All subsequent files and folders will be inside this directory.

2.  **Create Data Subfolders:**
    *   Inside the main project folder, create a `data` directory.
    *   Within `data`, create subdirectories for different file types: `pdfs`, `markdown`, `excel`, `database`.
    *   **Crucially**, within `data/pdfs` (and other relevant type-specific folders like `data/markdown`), create subdirectories named after the 1C sections you want to filter by. For example:
        *   `data/pdfs/УНФ`
        *   `data/pdfs/БП`
        *   `data/pdfs/ERP`
        *   `data/pdfs/Розница`
        *   `data/pdfs/ЗУП`
    *   Ensure these folder names exactly match the strings listed in the `KNOWN_1C_SECTIONS` variable in `index_data.py` and `query_data.py` if you want the section filtering to work correctly. The `get_1c_section_from_path` function in `index_data.py` relies on these path segments.

3.  **Place Your Documents:**
    *   Copy your PDF documents into the respective 1C section folders (e.g., all PDFs related to "УНФ" go into `data/pdfs/УНФ/`).
    *   Do the same for Markdown, Excel, or other file types if you implement their loaders.
    *   If using SQLite, place your `.db` file in the `data/database/` directory.

4.  **Create Script and Configuration Files:**
    *   Create the following files in your main project directory:
        *   `requirements.txt`
        *   `setup_env.sh`
        *   `index_data.py`
        *   `query_data.py`
    *   Copy the Python code generated in previous steps into `index_data.py` and `query_data.py`. Copy the requirements list into `requirements.txt` and the shell script content into `setup_env.sh`.

5.  **Create and Configure `.env` File:**
    *   Create a file named `.env` in the root of your project directory.
    *   Add your API keys or configuration. For OpenAI:
        ```env
        OPENAI_API_KEY="your_actual_openai_api_key_here"
        ```
    *   If you choose to use Hugging Face embeddings or models that require an API token, add it here as well (e.g., `HUGGINGFACEHUB_API_TOKEN="your_huggingface_token"`).
    *   The scripts `index_data.py` and `query_data.py` load these variables.

## 4. Step 2: Environment Setup and Dependency Installation

### For Linux/macOS:

1.  **Make `setup_env.sh` executable:**
    ```bash
    chmod +x setup_env.sh
    ```
2.  **Run the setup script:**
    ```bash
    ./setup_env.sh
    ```
    This script will create a Python virtual environment in `.venv/`, activate it (for the script's duration), and install all dependencies from `requirements.txt`.
3.  **Activate the environment in your current shell:**
    After the script finishes, you need to activate the environment in your current terminal session to run the Python scripts:
    ```bash
    source .venv/bin/activate
    ```
    You should see `(.venv)` prefixed to your shell prompt.

### For Windows:

1.  **Create a virtual environment:**
    Open Command Prompt or PowerShell, navigate to your project directory, and run:
    ```bash
    python -m venv .venv
    ```
2.  **Activate the virtual environment:**
    ```bash
    .venv\Scripts\activate
    ```
    You should see `(.venv)` prefixed to your shell prompt.
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
**Note on `unstructured` for Windows:** The `unstructured[local-inference]` package can sometimes have complex dependencies (like `detectron2`) that are challenging to install on Windows. If you encounter issues:
    *   Consider using `unstructured` without `[local-inference]` if you don't need all its capabilities for local model inference (though some loaders might rely on it).
    *   Look for specific Windows installation guides for `unstructured` or its problematic sub-dependencies.
    *   You might need to install tools like Build Tools for Visual Studio.
    *   For PDF, `PyPDFLoader` is a good alternative and is generally easier to install. The `index_data.py` script uses `PyPDFLoader` by default for PDFs.

## 5. Step 3: Data Indexing

1.  **Check Configurations in `index_data.py`:**
    *   Open `index_data.py`.
    *   Verify `DATA_DIR`, `PDF_DIR`, `MD_DIR`, etc., point to your actual data locations.
    *   Ensure `KNOWN_1C_SECTIONS` includes all the section folder names you created.
    *   Confirm your choice of embedding model (`USE_OPENAI_EMBEDDINGS` and `EMBEDDING_MODEL_NAME_HF`). If using OpenAI, ensure your API key is in `.env`.
    *   Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` if needed, though defaults are provided.

2.  **Run the Indexing Script:**
    Make sure your virtual environment is activated (`source .venv/bin/activate` or `.venv\Scripts\activate`).
    ```bash
    python index_data.py
    ```

3.  **What Happens During Indexing:**
    *   The script will scan the specified data directories (starting with PDFs).
    *   It will load documents, extract text, and assign metadata (including `source_type`, `file_name`, `full_path`, and `1c_section` derived from the path).
    *   The text will be split into smaller chunks.
    *   These chunks will be converted into vector embeddings using the chosen embedding model.
    *   The embeddings and their associated text/metadata will be stored in a ChromaDB vector store.
    *   A new directory named `db_chroma` (or as configured in `CHROMA_PERSIST_DIR`) will be created in your project folder, containing the database files.
    *   You'll see progress messages in the console.

## 6. Step 4: Querying and Testing

1.  **Check Configurations in `query_data.py`:**
    *   Open `query_data.py`.
    *   Ensure `CHROMA_PERSIST_DIR` and `CHROMA_COLLECTION_NAME` match those in `index_data.py`.
    *   Verify that `USE_OPENAI_EMBEDDINGS` and `EMBEDDING_MODEL_NAME_HF` are set consistently with how the data was indexed. **Using a different embedding model for querying than for indexing will lead to poor results.**
    *   Confirm your LLM choice (`LLM_PROVIDER`).

2.  **Run the Query Script:**
    Make sure your virtual environment is activated.
    The command format is:
    ```bash
    python query_data.py "YOUR_QUESTION_HERE" [--section SECTION_NAME]
    ```
    *   `"YOUR_QUESTION_HERE"`: Your question in natural language, enclosed in quotes.
    *   `--section SECTION_NAME` (optional): Filter the search to a specific 1C section (e.g., `УНФ`, `БП`). If omitted or set to `все`, it searches across all indexed sections.

3.  **Example Queries:**
    *   **Without section filter (searches all documents):**
        ```bash
        python query_data.py "Что такое основные средства в 1С?"
        ```
    *   **With section filter for "УНФ":**
        ```bash
        python query_data.py "Как сформировать отчет о продажах в УНФ?" --section УНФ
        ```
    *   **With section filter for "БП":**
        ```bash
        python query_data.py "Как закрыть месяц в Бухгалтерии предприятия?" --section БП
        ```

4.  **Analyzing the Output:**
    The script will print:
    *   **The LLM's Answer:** The generated response to your question.
    *   **Sources:** A list of document chunks that the retriever found relevant and were likely used by the LLM to formulate the answer. For each source, you'll see:
        *   A snippet of the content.
        *   Metadata like `file_name`, `full_path`, `source_type`, `1c_section`, and potentially `page_number` (for PDFs) or other DB-specific identifiers. This helps you verify the information and understand its origin.

## 7. Step 5: Evaluation and Iteration

Building a good RAG system is an iterative process. Evaluate the results critically:

*   **Answer Accuracy:** Is the LLM's answer correct and relevant to the question?
*   **Source Relevance:** Are the retrieved source documents actually relevant to the question and the answer?
*   **Filter Correctness:** If using section filters, are only documents from that section being retrieved?
*   **LLM Hallucinations:** Is the LLM inventing information not present in the sources?
*   **Text Extraction Quality:** Are there issues with how text is extracted from PDFs or other files (e.g., garbled text, missed sections)?
*   **Chunk Optimality:** Are the chunks too small (lacking context) or too large (diluting information or exceeding LLM context limits)?

**If results are unsatisfactory, consider tweaking:**

*   **Document Loaders:** Try different PDF loaders in `index_data.py` if PDF extraction is poor (e.g., `UnstructuredPDFLoader`, `PyMuPDFLoader`).
*   **Chunking Strategy:** Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` in `index_data.py`.
*   **Embedding Models:** Experiment with different embedding models (Russian-specific or better multilingual ones) in both `index_data.py` and `query_data.py`. **Remember to re-index if you change the embedding model.**
*   **Retriever Settings:**
    *   `k` value in `query_data.py` (number of documents to retrieve).
    *   `search_type` (e.g., "similarity", "mmr" for Maximum Marginal Relevance).
*   **LLM Model/Temperature:** Try different LLMs or adjust the `temperature` parameter in `query_data.py` to control creativity vs. factuality.
*   **Chain Type:** Experiment with different chain types in `RetrievalQA` (e.g., `map_reduce`, `refine`) if the "stuff" type is not working well for long documents or many retrieved chunks.
*   **Prompt Engineering:** Customize the prompt used by the `RetrievalQA` chain to better guide the LLM.

**Re-index your data (`python index_data.py`) whenever you change data preprocessing steps, chunking strategies, or the embedding model.**

## 8. Further Steps (Future Development)

*   **Implement Other Data Loaders:** Fully implement the placeholder functions in `index_data.py` for Markdown, Excel, and SQLite to incorporate a wider range of knowledge.
*   **Advanced Retrievers:** Explore more sophisticated retrievers in LangChain:
    *   `SelfQueryRetriever`: Allows the LLM to write metadata filters based on the natural language query.
    *   `ContextualCompressionRetriever`: Re-ranks and filters retrieved documents based on the query context.
    *   Parent Document Retriever: Indexes small chunks but retrieves larger parent chunks for better context.
*   **Hybrid Search:** Combine keyword-based search (like BM25) with semantic search for potentially better retrieval.
*   **Chat History:** Implement conversational memory to allow follow-up questions.
*   **User Interface:** Develop a web interface (e.g., using Streamlit or Flask) for easier interaction.
*   **Evaluation Pipeline:** Set up a more formal evaluation process using benchmark questions and metrics.
*   **Logging and Monitoring:** Add robust logging to track system behavior and identify issues.

This README provides a comprehensive guide to setting up and using your 1C RAG system. Remember that experimentation is key to achieving the best results for your specific dataset and use case.
