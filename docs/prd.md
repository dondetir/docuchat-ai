Here is a comprehensive document outlining the project plan for your local, CLI-based RAG system. This document is structured to serve as a guide for a multi-agent coding team, with clear instructions to pause after each phase.

***

### **Project Document: Local RAG (Retrieval-Augmented Generation) System**

**Top-Level Project View**

The goal of this project is to build a complete, self-contained Retrieval-Augmented Generation (RAG) system that runs entirely on a user's local machine. This system will be a command-line interface (CLI) tool capable of answering questions about a collection of local documents. The core functionality involves ingesting documents, converting them into a searchable database of embeddings, and using a local Large Language Model (LLM) to generate answers based on the retrieved information. The entire pipeline, from data ingestion to response generation, will not require any external internet connection after initial setup.

**Target Audience:** A multi-agent coding team.

**Key Directives:** 
- The project must be completed in four distinct phases. **After each phase, stop and ask the user for questions or approval before proceeding to the next phase.**
- Use virtual environment for Python dependencies
- Use parallel agents (maximum 5) whenever needed and possible
- Complete one phase at a time sequentially
- For each phase, create a test script in the `tests/` folder to validate phase completion
- Maintain organized folder structure throughout development
- Create and maintain `CLAUDE.md` with detailed project information
- Maintain `progress.md` with detailed tasks and agent information for every task

### **Project Setup Requirements**

**Virtual Environment Setup:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Folder Structure:**
```
DocuChat/
├── src/
│   ├── __init__.py
│   ├── cli.py
│   ├── document_loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_db.py
│   ├── llm_client.py
│   └── rag_pipeline.py
├── tests/
│   ├── test_phase1.py
│   ├── test_phase2.py
│   ├── test_phase3.py
│   └── test_phase4.py
├── docs/
│   └── prd.md
├── chroma_db/
├── requirements.txt
├── CLAUDE.md
├── progress.md
└── docuchat.py
```

---

### **Phase 1: File Ingestion and Initial Processing**

**Objective:** Create the foundational component of the system that can read and process documents from a local directory.

**Detailed Steps:**
1.  **CLI Setup:** Implement a command-line interface using a library like `argparse` or `click`. The CLI should accept the path to the directory containing the documents and support the following flags:
    - `--chunk-size`: Chunk size in characters (default: 2000)
    - `--top-k`: Number of chunks to retrieve per query (default: 10)
    - `--rebuild`: Force recreation of vector database
    - `--show-sources`: Display source attribution after responses
    - `--verbose`: Show detailed progress during processing
2.  **File Discovery:** Write a function that recursively walks through the provided directory path and all subdirectories. This function should identify and collect all files with specified extensions (`.pdf`, `.txt`, `.md`, `.docx`).
3.  **Document Loading:** For each identified file, use the appropriate library to load its content into a raw text format. Use `pypdf` for PDF files, `python-docx` for Word documents (extracting plain text only), and standard Python file I/O for text and markdown files. Skip files that cannot be processed with a warning message, but continue processing other files.
4.  **Confirmation Output:** After all files have been processed, print a clear, formatted list of the documents that were successfully read. For example: "Successfully read the following documents:" followed by a list of filenames.

5.  **Phase 1 Test Script:** Create `tests/test_phase1.py` that validates:
    - CLI argument parsing works correctly
    - File discovery finds all supported file types recursively
    - Document loading successfully extracts text from each file type
    - Error handling works for problematic files
    - Progress output functions correctly

**STOP AND ASK FOR USER QUESTIONS:**
* `User, Phase 1 is complete. I have read and listed the documents. Do you have any questions or changes before I proceed to Phase 2?`

---

### **Phase 2: Chunking and Vectorization**

**Objective:** Convert the raw text from the documents into a searchable vector database.

**Detailed Steps:**
1.  **Document Chunking:** Take the raw text content from each document processed in Phase 1 and split it into smaller, overlapping chunks. Use a `RecursiveCharacterTextSplitter` with 2000 characters per chunk and 200 character overlap (configurable via `--chunk-size` flag) to ensure that context is maintained within each chunk.
2.  **On-Device Embedding Model:** Load the `all-MiniLM-L6-v2` sentence-transformers model that can run efficiently on a local machine.
3.  **Embedding Generation:** Iterate through each text chunk and use the loaded embedding model to generate a numerical vector (embedding) for it.
4.  **Vector Database Creation:** Initialize a local ChromaDB vector database stored in `./chroma_db/` directory. The database persists between runs by default, but can be recreated using the `--rebuild` flag.
5.  **Database Population:** Add the generated vector embeddings and their corresponding text chunks to the vector database.
6.  **Confirmation Output:** For each document, print a confirmation message like: "Successfully created and stored embeddings for [filename]."

7.  **Phase 2 Test Script:** Create `tests/test_phase2.py` that validates:
    - Text chunking produces correct chunk sizes and overlaps
    - Embedding model loads successfully
    - Vector embeddings are generated correctly
    - ChromaDB database creation and persistence works
    - Database rebuild functionality works with `--rebuild` flag

**STOP AND ASK FOR USER QUESTIONS:**
* `User, Phase 2 is complete. All documents have been converted to embeddings and stored in the vector database. Do you have any questions or changes before I proceed to Phase 3?`

---

### **Phase 3: LLM Integration**

**Objective:** Load and test a local, on-device LLM to ensure it can be prompted and can generate a response.

**Detailed Steps:**
1.  **Model Selection and Loading:** Connect to the Gemma 3 270M model (gemma3:270m) via Ollama's REST API at `http://localhost:11434/api/generate`. This model provides a 32K token context window, perfect for RAG applications with substantial retrieved context.
2.  **Test Prompt Function:** Create a dedicated function that takes a string prompt as input, sends it to the loaded LLM, and returns the generated text response.
3.  **Basic LLM Test:** Call the test prompt function with a simple query, like "What is the capital of France?", and print the LLM's response to verify that the model is loaded and working correctly. This step is independent of the document data.

4.  **Phase 3 Test Script:** Create `tests/test_phase3.py` that validates:
    - Ollama REST API connection works correctly
    - Gemma3:270m model responds to prompts
    - Response generation function works as expected
    - Error handling for Ollama connection issues
    - Response parsing and formatting works correctly

**STOP AND ASK FOR USER QUESTIONS:**
* `User, Phase 3 is complete. The LLM has been successfully loaded and tested. Do you have any questions or changes before I proceed to the final phase?`

---

### **Phase 4: RAG Pipeline Integration and Chat Loop**

**Objective:** Connect the LLM with the vector database to build a complete, interactive RAG chat system.

**Detailed Steps:**
1.  **Query Embedding:** Inside the chat loop, when the user inputs a new question, use the **same embedding model from Phase 2** to convert this question into a vector.
2.  **Vector Search:** Perform a similarity search on the vector database using the user's query vector. Retrieve the top K most relevant text chunks from the documents (configurable via `--top-k` flag, default: 10 chunks).
3.  **Prompt Construction:** Combine the user's original query and the retrieved text chunks into a single, well-structured prompt for the LLM. The prompt must instruct the LLM to use *only the provided context* to answer the question.
4.  **Generation and Response:** Pass the combined prompt to the LLM. The LLM will generate a response based on the provided context.
5.  **Interactive Loop:** Display the final answer to the user. If `--show-sources` flag is enabled, also display which documents/chunks were used as sources. Return to wait for the next question, creating a continuous conversational experience.
6.  **Final Confirmation:** Once the user quits the application, print a final confirmation message.

7.  **Phase 4 Test Script:** Create `tests/test_phase4.py` that validates:
    - Query embedding generation works correctly
    - Vector similarity search retrieves relevant chunks
    - Prompt construction combines context properly
    - End-to-end RAG pipeline produces coherent responses
    - Interactive chat loop functions correctly
    - Source attribution works when `--show-sources` is enabled
    - All CLI flags work as expected in the complete system

**STOP AND ASK FOR USER QUESTIONS:**
* `User, Phase 4 and the entire project are now complete. The RAG system is fully functional. Do you have any final questions or require any modifications?`
