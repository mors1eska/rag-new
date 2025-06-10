import os
import glob
import json
from collections import defaultdict
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document # For creating Documents from SQLite data or FAQs
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import sqlite3
import pandas as pd # For convert_excel_to_faq_format
from pathlib import Path # For convert_excel_to_faq_format


def convert_excel_to_faq_format(xlsx_path: str, mapping_path: str = None):
    """
    Converts an Excel file (e.g., from a Service Desk system) into a structured
    FAQ JSON format suitable for further processing and ingestion.
    The Excel file is expected to have specific columns (in Russian).
    Mappings from Excel component names to target folder names can be provided via a JSON file.

    Args:
        xlsx_path (str): Path to the input Excel file.
        mapping_path (str, optional): Path to the JSON mapping file. Defaults to None.
    """
    df = pd.read_excel(xlsx_path)
    mapping_data = {}
    if mapping_path and os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)

    file_basename = os.path.basename(xlsx_path)
    mapping_entry = mapping_data.get(file_basename)
    if not mapping_entry:
        print(f"INFO: No specific mapping entry found for {file_basename} in {mapping_path}. Using defaults or skipping if essential info missing.")
        # Decide if you want to proceed with default values or return
        # For this example, we'll assume some default behavior if possible,
        # or the function might implicitly skip items if `folder` isn't found later.

    source_type_label = mapping_entry.get("source_type", "excel_faq") if mapping_entry else "excel_faq"
    program_map = mapping_entry.get("program_map", {}) if mapping_entry else {}

    # Expected column names (in Russian)
    # These should match the columns in your input Excel file.
    name_col = "Наименование" # Name/Title of the FAQ/issue
    number_col = "Номер"       # ID or number
    link_col = "Ссылка"        # URL link
    description_col = "Описание" # Answer or description
    components_col = "Компоненты" # Components/programs related, can be slash-separated
    sections_col = "Разделы"   # Sections/subsections

    required_columns = [name_col, number_col, link_col, description_col, components_col, sections_col]
    for col in required_columns:
        if col not in df.columns:
            print(f"ERROR: Missing required column in Excel file '{xlsx_path}': {col}")
            return

    grouped_data = defaultdict(list)
    for _, row in df.iterrows():
        name = str(row[name_col]).strip()
        answer = str(row[description_col]).strip()
        if not name or not answer: # Skip if essential question or answer is missing
            continue

        raw_components = str(row[components_col]).split('/')
        for comp in raw_components:
            comp = comp.strip()
            # Use program_map to find the target folder for this component
            folder = program_map.get(comp)
            if not folder: # If component not in map, or no map provided, skip this component
                print(f"WARN: Component '{comp}' not found in program_map or program_map not provided. Skipping for this entry.")
                continue

            target_dir = os.path.join(BASE_DATA_DIR, folder, "faq")
            os.makedirs(target_dir, exist_ok=True)
            # Standardized output file name
            output_filename = f"from_excel_{Path(xlsx_path).stem}_{comp.replace(':', '_').replace('/', '_')}.json"
            output_path = os.path.join(target_dir, output_filename)

            grouped_data[output_path].append({
                "question": name,
                "answer": answer,
                "url": str(row[link_col]).strip(),
                "source_type": source_type_label, # From mapping or default
                "code": str(row[number_col]).strip(),
                "subsection": str(row[sections_col]).strip()
            })

    for path, records in grouped_data.items():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"INFO: Saved {len(records)} FAQ records to {path}")


# --- Configuration ---
load_dotenv()  # Load environment variables from .env file

# Base path for data
BASE_DATA_DIR = "data" # Parent directory for configuration folders (UNF, BP, etc.)
SQLITE_DB_PATH = os.path.join(BASE_DATA_DIR, "database", "1c_knowledge_base.db")
SQLITE_FTS_TABLE = "documents_fts" # FTS table name

# ChromaDB Configuration
CHROMA_PERSIST_DIR = "db_chroma"
CHROMA_COLLECTION_NAME = "rag_collection"

# Embedding Model Configuration
EMBEDDING_MODEL_NAME_HF = "sentence-transformers/all-MiniLM-L6-v2" # Example HuggingFace model
USE_OPENAI_EMBEDDINGS = True # Set to False to use HuggingFaceEmbeddings by default

# Text Splitter Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Note: 1C sections are now dynamically determined from folder names within BASE_DATA_DIR.
# e.g., data/UNF/, data/BP/. The script will automatically find these.

# --- Helper Functions ---

def get_embedding_model():
    """Initializes and returns the selected embedding model."""
    if USE_OPENAI_EMBEDDINGS:
        if os.getenv("OPENAI_API_KEY"):
            print("Using OpenAI Embeddings.")
            return OpenAIEmbeddings()
        else:
            print("OPENAI_API_KEY not found. Switching to HuggingFace Embeddings.")
            print(f"Using HuggingFace Embeddings: {EMBEDDING_MODEL_NAME_HF}")
            return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME_HF,
                                       model_kwargs={'device': 'cpu'})
    else:
        print(f"Using HuggingFace Embeddings: {EMBEDDING_MODEL_NAME_HF}")
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME_HF,
                                   model_kwargs={'device': 'cpu'})

def get_1c_section_from_path(file_path: str, base_data_dir: str) -> str:
    """
    Extracts the 1C section name from the file path.
    The 1C section is the name of the first-level directory within base_data_dir.
    Example: base_data_dir="data", file_path="data/UNF/pdfs/doc1.pdf" -> "UNF"
    """
    try:
        normalized_base_path = os.path.join(os.path.normpath(base_data_dir), "")
        normalized_file_path = os.path.normpath(file_path)

        if not normalized_file_path.startswith(normalized_base_path):
            return "Unknown" # File is not within the base data directory

        relative_path = os.path.relpath(normalized_file_path, normalized_base_path)
        path_parts = relative_path.split(os.sep)
        if path_parts:
            return path_parts[0]
    except Exception as e:
        print(f"Error extracting section from path {file_path}: {e}")
    return "Unknown"

# --- Data Loading and Processing Functions ---

def load_process_pdfs(config_specific_dir: str, section_name: str, text_splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    """
    Loads PDF files from the 'pdfs' subdirectory within a specific 1C configuration directory,
    extracts metadata, splits them, and returns document chunks.
    Args:
        config_specific_dir: Path to the configuration folder (e.g., "data/UNF").
        section_name: Name of the 1C section (e.g., "UNF").
        text_splitter: Configured text splitter instance.
    Returns:
        A list of Document chunks.
    """
    all_chunks: List[Document] = []
    pdf_data_subdir = os.path.join(config_specific_dir, "pdfs") # Look for 'pdfs' subfolder

    print(f"  Loading PDFs from: {pdf_data_subdir} for section '{section_name}'")
    if not os.path.isdir(pdf_data_subdir):
        print(f"    PDF directory not found: {pdf_data_subdir}")
        return all_chunks

    for pdf_file_path in glob.glob(os.path.join(pdf_data_subdir, "*.pdf")):
        try:
            print(f"    Processing PDF: {pdf_file_path}")
            loader = PyPDFLoader(pdf_file_path)
            documents = loader.load()

            processed_docs_for_file = []
            for doc in documents:
                doc.metadata["source_type"] = "pdf"
                doc.metadata["file_name"] = os.path.basename(pdf_file_path)
                doc.metadata["full_path"] = pdf_file_path
                doc.metadata["1c_section"] = section_name
                # Ensure page_number is set, PyPDFLoader might use 'page' or other keys
                doc.metadata['page_number'] = doc.metadata.get('page', doc.metadata.get('page_number', -1)) + 1
                processed_docs_for_file.append(doc)

            chunks = text_splitter.split_documents(processed_docs_for_file)
            all_chunks.extend(chunks)
            print(f"      Loaded {len(documents)} pages, split into {len(chunks)} chunks.")
        except Exception as e:
            print(f"      Error processing PDF {pdf_file_path}: {e}")
    return all_chunks

def load_process_markdown(config_specific_dir: str, section_name: str, text_splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    """
    Stub for loading Markdown files from the 'markdown' subdirectory. (To be implemented)
    """
    all_chunks: List[Document] = []
    md_data_subdir = os.path.join(config_specific_dir, "markdown")
    print(f"  Loading Markdown from: {md_data_subdir} for section '{section_name}' (Stub - Not Implemented)")
    if not os.path.isdir(md_data_subdir):
        return all_chunks
    # Example structure (to be implemented):
    # for md_file_path in glob.glob(os.path.join(md_data_subdir, "*.md")):
    #     loader = UnstructuredMarkdownLoader(md_file_path)
    #     # ... process and split ...
    return all_chunks

def load_process_excel(config_specific_dir: str, section_name: str, text_splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    """
    Loads Excel files from the 'excel' subdirectory.
    """
    all_chunks: List[Document] = []
    excel_data_subdir = os.path.join(config_specific_dir, "excel")
    print(f"  Loading Excel from: {excel_data_subdir} for section '{section_name}'")
    if not os.path.isdir(excel_data_subdir):
        return all_chunks

    for excel_file_path in glob.glob(os.path.join(excel_data_subdir, "*.xlsx")):
        try:
            print(f"    Processing Excel: {excel_file_path}")
            # UnstructuredExcelLoader loads each sheet as a document by default.
            # Consider how you want to treat sheets (one doc per sheet or one doc per file).
            loader = UnstructuredExcelLoader(excel_file_path, mode="elements") # mode="elements" or "single"
            documents = loader.load()
            for doc in documents:
                doc.metadata["source_type"] = "excel"
                doc.metadata["file_name"] = os.path.basename(excel_file_path)
                doc.metadata["full_path"] = excel_file_path
                doc.metadata["1c_section"] = section_name
            chunks = text_splitter.split_documents(documents)
            all_chunks.extend(chunks)
            print(f"      Loaded and split into {len(chunks)} chunks.")
        except Exception as e:
            print(f"      Error processing Excel {excel_file_path}: {e}")
    return all_chunks

def load_process_faq(config_specific_dir: str, section_name: str, text_splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    """
    Loads FAQ data from JSON files in the 'faq' subdirectory.
    Each JSON file should contain a list of Q&A pairs.
    """
    all_chunks: List[Document] = []
    faq_data_subdir = os.path.join(config_specific_dir, "faq")
    print(f"  Loading FAQ from: {faq_data_subdir} for section '{section_name}'")
    if not os.path.isdir(faq_data_subdir):
        return all_chunks

    for faq_file_path in glob.glob(os.path.join(faq_data_subdir, "*.json")):
        try:
            print(f"    Processing FAQ JSON: {faq_file_path}")
            with open(faq_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            documents_to_split = []
            for item in data: # Assuming data is a list of Q&A dicts
                q = item.get("question", "").strip()
                a = item.get("answer", "").strip()
                if q and a:
                    # Combine question and answer for context, or treat them separately
                    content = f"Question: {q}\n\nAnswer: {a}"
                    metadata = {
                        "source_type": "faq",
                        "file_name": os.path.basename(faq_file_path),
                        "full_path": faq_file_path,
                        "1c_section": section_name,
                        "url": item.get("url", ""),
                        "date": item.get("date", ""), # Assuming these fields might exist
                        "question": q # Store original question separately in metadata
                    }
                    # Each Q&A pair becomes a Document for LangChain
                    documents_to_split.append(Document(page_content=content, metadata=metadata))

            # Split the combined Q&A content if it's too long
            chunks = text_splitter.split_documents(documents_to_split)
            all_chunks.extend(chunks)
            print(f"      Loaded {len(documents_to_split)} FAQs, split into {len(chunks)} chunks.")
        except Exception as e:
            print(f"      Error processing FAQ JSON {faq_file_path}: {e}")
    return all_chunks

def load_process_sqlite(db_path: str, text_splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    """
    Stub for loading data from SQLite. (To be implemented)
    For SQLite, '1c_section' metadata should ideally come from a dedicated column in the tables,
    not from the database file path itself.
    """
    all_chunks: List[Document] = []
    print(f"\nLoading SQLite data from: {db_path} (Stub - Not Implemented)")
    print("  Note: For SQLite, '1c_section' metadata should be in a table column.")
    if not os.path.isfile(db_path):
        print(f"  SQLite DB file not found: {db_path}")
        return all_chunks
    # Example structure (to be implemented):
    # from sqlalchemy import create_engine, text
    # engine = create_engine(f"sqlite:///{db_path}")
    # with engine.connect() as connection:
    #     result = connection.execute(text("SELECT id, content_column, section_column, source_info FROM your_table"))
    #     for row in result:
    #         doc = Document(page_content=row.content_column, metadata={...})
    #         all_chunks.extend(text_splitter.split_documents([doc]))
    return all_chunks

# --- Main Indexing Logic ---

def split_faq_json_by_program(source_dir: str, base_data_dir: str) -> None:
    """
    Scans for general FAQ JSON files in `source_dir`, splits them by program/product,
    and writes individual files to the corresponding configuration folders within `base_data_dir`.
    The mapping from program name (in JSON) to folder name is defined in `mappings.json`
    within `source_dir`.
    """
    json_mapping_path = os.path.join(source_dir, "mappings.json")
    json_program_map = {}
    if os.path.exists(json_mapping_path):
        try:
            with open(json_mapping_path, "r", encoding="utf-8") as f:
                # Expects a structure like: {"json_program_map": {"1С:БП": "BP", ...}}
                json_program_map = json.load(f).get("json_program_map", {})
        except json.JSONDecodeError:
            print(f"Error decoding {json_mapping_path}. Proceeding without program name mapping for JSON splitting.")

    for json_file in glob.glob(os.path.join(source_dir, "*.json")):
        if os.path.basename(json_file) == "mappings.json":
            continue # Skip the mapping file itself

        try:
            print(f"\n📦 Processing shared FAQ file: {json_file}")
            with open(json_file, "r", encoding="utf-8") as f:
                entries = json.load(f) # Assuming top level is a list of blocks

            program_map_data = defaultdict(list) # program_name -> list_of_qas_for_that_program

            for block in entries: # Iterate through blocks in the shared FAQ file
                program_name_original = block.get("program", "UnknownProgram").strip()
                questions_list = block.get("questions", [])
                if not questions_list:
                    continue

                # Use the mapping to get the target folder name, or derive it
                # Default to stripping "1С:" and spaces if no map entry
                target_folder_name = json_program_map.get(program_name_original,
                                                          program_name_original.replace("1С:", "").strip())

                # Add all questions from this block to the corresponding program's list
                # The questions themselves should be in the format expected by load_process_faq
                # i.e., list of {"question": "...", "answer": "...", ...}
                program_map_data[target_folder_name].extend(questions_list)

            # Save the grouped Q&A lists to their respective target directories
            for folder_name, qlist in program_map_data.items():
                if folder_name == "UnknownProgram" or not qlist:
                    print(f"  Skipping save for '{folder_name}' due to unknown program or empty question list.")
                    continue

                target_faq_dir = os.path.join(base_data_dir, folder_name, "faq")
                os.makedirs(target_faq_dir, exist_ok=True)

                output_filename = f"from_import_{Path(json_file).stem}_{folder_name}.json"
                output_path = os.path.join(target_faq_dir, output_filename)

                with open(output_path, "w", encoding="utf-8") as out_f:
                    json.dump(qlist, out_f, ensure_ascii=False, indent=2)
                print(f"  ✅ Saved {len(qlist)} Q&A pairs to {output_path} for program '{folder_name}'")
            
            # Remove original file after successful processing
            os.remove(json_file)
            print(f"  🗑️ Removed original shared FAQ file: {json_file}")
        except Exception as e:
            print(f"  ⚠️ Error processing shared FAQ file {json_file}: {e}")


def main():
    """
    Main function to manage data loading, processing, and indexing for RAG.
    """
    print("Starting RAG data indexing process...")

    embedding_model = get_embedding_model()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    all_document_chunks: List[Document] = []

    if not os.path.isdir(BASE_DATA_DIR):
        print(f"ERROR: Base data directory '{BASE_DATA_DIR}' not found.")
        return

    # Preprocess general-purpose FAQ files from 'import' directory
    shared_import_dir = os.path.join(BASE_DATA_DIR, "import")
    if os.path.isdir(shared_import_dir):
        print(f"\nChecking for shared FAQ files in: {shared_import_dir}")
        split_faq_json_by_program(shared_import_dir, BASE_DATA_DIR)
        # Also process Excel files from the import directory
        excel_import_mappings = os.path.join(shared_import_dir, "mappings.json")
        for excel_file in glob.glob(os.path.join(shared_import_dir, "*.xlsx")):
            print(f"\nProcessing shared Excel file: {excel_file}")
            convert_excel_to_faq_format(excel_file, mapping_path=excel_import_mappings)
            try:
                # Optionally remove or move the processed Excel file
                # os.remove(excel_file)
                # print(f"  🗑️ Removed imported Excel file: {excel_file}")
                pass # Decide on handling after conversion
            except Exception as e:
                print(f"  ⚠️ Could not remove Excel file {excel_file}: {e}")


    # Iterate through configuration folders (UNF, BP, etc.) within BASE_DATA_DIR
    for section_folder_name in os.listdir(BASE_DATA_DIR):
        if section_folder_name.startswith(".") or section_folder_name == "import" or section_folder_name == "database":
            continue # Skip hidden folders, import, and database utility folders

        config_specific_dir = os.path.join(BASE_DATA_DIR, section_folder_name)
        if os.path.isdir(config_specific_dir):
            print(f"\nProcessing configuration/section: {section_folder_name}")

            all_document_chunks.extend(load_process_pdfs(config_specific_dir, section_folder_name, text_splitter))
            all_document_chunks.extend(load_process_markdown(config_specific_dir, section_folder_name, text_splitter))
            all_document_chunks.extend(load_process_excel(config_specific_dir, section_folder_name, text_splitter))
            all_document_chunks.extend(load_process_faq(config_specific_dir, section_folder_name, text_splitter))

    # Load data from SQLite (processed separately as it's not tied to config folders in the same way)
    all_document_chunks.extend(load_process_sqlite(SQLITE_DB_PATH, text_splitter))

    if not all_document_chunks:
        print("\nNo documents found or processed. Exiting.")
        return

    print(f"\nTotal document chunks to index: {len(all_document_chunks)}")

    # --- SQLite FTS Indexing ---
    print(f"\nStarting SQLite FTS indexing: {SQLITE_DB_PATH}")
    conn = None
    try:
        db_dir = os.path.dirname(SQLITE_DB_PATH)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
            print(f"  Created directory for SQLite DB: {db_dir}")

        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()

        # Create FTS table if it doesn't exist
        # doc_id should be unique for INSERT OR REPLACE to work correctly.
        create_table_sql = f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {SQLITE_FTS_TABLE} (
            doc_id TEXT UNIQUE,
            content TEXT,
            tokenize = 'porter unicode61'
        );
        """
        cursor.execute(create_table_sql)
        print(f"  FTS table '{SQLITE_FTS_TABLE}' is ready/created.")

        # Optional: Clear old records if re-indexing everything
        # cursor.execute(f"DELETE FROM {SQLITE_FTS_TABLE};")
        # print(f"  Old records (if any) deleted from '{SQLITE_FTS_TABLE}'.")

        inserted_count = 0
        if all_document_chunks:
            for i, doc_chunk in enumerate(all_document_chunks):
                content = doc_chunk.page_content
                # Create a unique doc_id for each chunk
                base_path = doc_chunk.metadata.get('full_path', doc_chunk.metadata.get('file_name', f"unknown_doc_{i}"))
                page_num = doc_chunk.metadata.get('page_number', -1)
                chunk_specific_id = f"_chunk_{i}" # Add chunk index for uniqueness

                doc_id = f"{base_path}_page_{page_num}{chunk_specific_id}" if page_num != -1 else f"{base_path}{chunk_specific_id}"

                if not content.strip(): # Skip empty chunks
                    print(f"    Skipping empty chunk for doc_id: {doc_id}")
                    continue
                try:
                    cursor.execute(
                        f"INSERT OR REPLACE INTO {SQLITE_FTS_TABLE} (doc_id, content) VALUES (?, ?)",
                        (doc_id, content)
                    )
                    inserted_count += 1
                except sqlite3.IntegrityError as ie:
                     print(f"    Integrity error inserting doc_id {doc_id}: {ie}. This might indicate a non-unique doc_id if not using 'INSERT OR REPLACE'.")
                except Exception as e_insert:
                    print(f"    Error inserting chunk for doc_id {doc_id}: {e_insert}")
            conn.commit()
            print(f"  Successfully inserted/replaced {inserted_count} chunks into FTS table.")
        else:
            print("  No document chunks to index into SQLite FTS.")

    except sqlite3.Error as e:
        print(f"  SQLite error: {e}")
    except Exception as e_global:
        print(f"  Unexpected error during SQLite FTS indexing: {e_global}")
    finally:
        if conn:
            conn.close()
            print(f"  SQLite connection ({SQLITE_DB_PATH}) closed.")
    print("--- SQLite FTS indexing finished ---")

    # --- ChromaDB and TF-IDF Artifacts ---
    # Clear old ChromaDB and TF-IDF artifacts before creating new ones
    # This is important to avoid conflicts or using stale data.
    # If CHROMA_PERSIST_DIR is used for both, rmtree will clear everything.
    if os.path.exists(CHROMA_PERSIST_DIR):
        print(f"Deleting old ChromaDB/TF-IDF storage directory: {CHROMA_PERSIST_DIR}")
        import shutil
        shutil.rmtree(CHROMA_PERSIST_DIR)

    if not os.path.exists(CHROMA_PERSIST_DIR):
        os.makedirs(CHROMA_PERSIST_DIR)
        print(f"Created directory for ChromaDB and TF-IDF: {CHROMA_PERSIST_DIR}")

    # --- TF-IDF Generation and Saving ---
    print("\nGenerating TF-IDF vectors...")
    try:
        if all_document_chunks:
            texts_for_tfidf = [doc.page_content for doc in all_document_chunks]

            # Ensure unique doc_id for each chunk for TF-IDF, similar to FTS
            doc_ids_for_tfidf = []
            for i, doc_chunk in enumerate(all_document_chunks):
                base_path = doc_chunk.metadata.get('full_path', doc_chunk.metadata.get('file_name', f"unknown_tfidf_doc_{i}"))
                page_num = doc_chunk.metadata.get('page_number', -1)
                chunk_specific_id = f"_chunk_{i}" # Suffix for chunk uniqueness
                doc_id = f"{base_path}_page_{page_num}{chunk_specific_id}" if page_num != -1 else f"{base_path}{chunk_specific_id}"
                doc_ids_for_tfidf.append(doc_id)

            if texts_for_tfidf:
                tfidf_vectorizer = TfidfVectorizer(
                    max_df=0.95,
                    min_df=2,
                    ngram_range=(1, 2),
                    stop_words=None # Consider 'russian' if scikit-learn supports or use an external library
                )
                tfidf_matrix = tfidf_vectorizer.fit_transform(texts_for_tfidf)

                tfidf_model_path = os.path.join(CHROMA_PERSIST_DIR, "tfidf_model.joblib")
                tfidf_vectors_path = os.path.join(CHROMA_PERSIST_DIR, "tfidf_vectors_and_ids.joblib")

                joblib.dump(tfidf_vectorizer, tfidf_model_path)
                print(f"  TF-IDF model saved to: {tfidf_model_path}")

                # Store doc_ids along with vectors. Ensure these IDs match the granularity of the vectors.
                joblib.dump({'ids': doc_ids_for_tfidf, 'vectors': tfidf_matrix}, tfidf_vectors_path)
                print(f"  TF-IDF vectors and IDs saved to: {tfidf_vectors_path}")
                print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")
            else:
                print("  No text data available for TF-IDF generation (text list is empty).")
        else:
            print("  No document chunks available for TF-IDF generation.")
    except Exception as e:
        print(f"  Error during TF-IDF generation or saving: {e}")
    print("--- TF-IDF generation finished ---")

    # --- ChromaDB Indexing ---
    print(f"\nInitializing Chroma vector store at: {CHROMA_PERSIST_DIR}")
    print(f"Using collection name: {CHROMA_COLLECTION_NAME}")

    # At this point, CHROMA_PERSIST_DIR should exist.
    # If it was deleted and not recreated, Chroma.from_documents might fail.
    # The logic above ensures it's created.

    try:
        vector_store = Chroma.from_documents(
            documents=all_document_chunks, # These are LangChain Document objects
            embedding=embedding_model,
            collection_name=CHROMA_COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR
        )
        print(f"\nSuccessfully indexed {len(all_document_chunks)} chunks into Chroma.")
        print(f"Vector store persisted at: {CHROMA_PERSIST_DIR}")
    except Exception as e:
        print(f"Error during ChromaDB indexing: {e}")

    print("\nIndexing process finished.")

if __name__ == "__main__":
    # Example: Ensure data directory structure for testing if it doesn't exist.
    # Users should create these folders and place their files accordingly.
    # os.makedirs(os.path.join(BASE_DATA_DIR, "UNF", "pdfs"), exist_ok=True)
    # os.makedirs(os.path.join(BASE_DATA_DIR, "BP", "pdfs"), exist_ok=True)
    # os.makedirs(os.path.join(BASE_DATA_DIR, "import"), exist_ok=True)
    # os.makedirs(os.path.join(BASE_DATA_DIR, "database"), exist_ok=True)
    # print(f"Ensure '{BASE_DATA_DIR}' directory exists and contains subfolders for configurations (e.g., UNF, BP),")
    # print(f"and within them, subfolders for data types (pdfs, markdown, excel, faq).")
    # print(f"For example: {os.path.join(BASE_DATA_DIR, 'UNF', 'pdfs', 'your_document.pdf')}")
    # print(f"Shared import files can be placed in '{os.path.join(BASE_DATA_DIR, 'import')}'")

    main()
