# DocuChat Web Application

A modern, production-ready web interface for DocuChat that provides a Grok-like chat experience for document Q&A.

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.8+** 
- **Ollama** installed and running
- Documents in PDF, DOCX, or TXT format

### 2. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama service (in a separate terminal)
ollama serve

# Pull a recommended model
ollama pull gemma2:2b
```

### 3. Launch Web App

```bash
# Option 1: Use the launcher (recommended)
python web/run_web_app.py

# Option 2: Run directly
python web/web_app.py
```

The web interface will open automatically in your browser at `http://localhost:7860`

## 📋 Features

### 🔥 Core Functionality
- **Document Processing**: Upload documents from any folder path
- **Real-time Progress**: Live updates during document processing 
- **AI-Powered Chat**: Ask questions about your documents
- **Source Citations**: See exactly which documents informed each answer
- **Multi-format Support**: PDF, Word (DOCX), and text files

### 🎨 User Interface
- **Modern Design**: Clean, professional Grok-like interface
- **Mobile Responsive**: Works on desktop, tablet, and mobile
- **Real-time Updates**: Progress bars and status indicators
- **Error Handling**: User-friendly error messages and recovery
- **Settings Panel**: Customize processing parameters

### ⚙️ Advanced Features
- **Parallel Processing**: Fast document ingestion
- **Vector Database**: Efficient semantic search with ChromaDB
- **Configurable Parameters**: Adjust chunk size, retrieval count, etc.
- **Performance Monitoring**: Processing times and throughput stats

## 🖥️ User Guide

### Step 1: Process Documents
1. Go to the **"📁 Process Documents"** tab
2. Enter the full path to your document folder
   - Windows: `C:\Users\YourName\Documents\MyFiles`
   - Mac/Linux: `/home/username/documents/myfiles`
3. Click **"🚀 Start Processing"**
4. Wait for processing to complete (progress bar will show status)

### Step 2: Chat with Documents
1. Switch to the **"💬 Chat"** tab
2. Type your question in the text box
3. Press Enter or click **"Send 📤"**
4. Review the answer with source citations

### Step 3: Customize Settings (Optional)
1. Go to **"⚙️ Settings"** tab
2. Adjust parameters:
   - **Chunk Size**: How text is split (500-2000 recommended)
   - **Top-K Results**: Number of relevant sections to find (3-10)
   - **Ollama Model**: AI model to use for answers
3. Click **"💾 Save Settings"**

## 💡 Tips for Best Results

### Document Preparation
- Use clear, well-formatted documents
- Ensure text is searchable (not just images)
- Organize related documents in the same folder

### Asking Questions
- Be specific about what you're looking for
- Ask conceptual questions, not just keyword searches
- Reference specific topics or document types
- Ask follow-up questions to drill down

### Example Questions
- "What are the main conclusions about climate change?"
- "How does the company handle customer complaints?"
- "What safety procedures are mentioned in the manual?"
- "Can you summarize the financial projections?"

## 🔧 Configuration

### Ollama Models
Recommended models (balance of speed and quality):
- `gemma2:2b` - Fast, good for most tasks
- `llama3.1:8b` - Higher quality, slower
- `mistral:7b` - Good balance

### Processing Settings
- **Chunk Size**: 
  - Small (500): Better for specific facts
  - Large (1500): Better for context and summaries
- **Top-K Results**:
  - Low (3-5): Focused, precise answers
  - High (8-10): More comprehensive context

## 🛠️ Troubleshooting

### Common Issues

**"Ollama service not available"**
```bash
# Start Ollama service
ollama serve

# Check if model is installed
ollama list

# Pull model if needed
ollama pull gemma2:2b
```

**"No supported documents found"**
- Check folder path is correct
- Ensure files are PDF, DOCX, or TXT format
- Check file permissions

**"Processing failed"**
- Check available disk space
- Ensure Python has write permissions
- Try processing fewer documents at once

**Web interface won't load**
- Check if port 7860 is available
- Try running: `python -m gradio --version`
- Reinstall Gradio: `pip install --upgrade gradio`

### Performance Tips
- **For large document collections**: Process in batches
- **For faster processing**: Use smaller chunk sizes
- **For better answers**: Use larger, high-quality models
- **For resource-constrained systems**: Use `gemma2:2b` model

## 📊 System Requirements

### Minimum
- **RAM**: 8GB
- **Storage**: 5GB free space
- **CPU**: 4 cores
- **GPU**: Optional (speeds up processing)

### Recommended
- **RAM**: 16GB+
- **Storage**: 10GB+ free space
- **CPU**: 8+ cores
- **GPU**: NVIDIA with 8GB+ VRAM

## 🔒 Security Notes

- Web interface runs locally by default (`127.0.0.1:7860`)
- No data is sent to external servers
- All processing happens on your machine
- Vector database stored locally in `./chroma_web/`

## 📁 File Structure

```
DocuChat/
├── web/
│   ├── web_app.py          # Main web application
│   ├── run_web_app.py      # Launcher with checks
├── requirements.txt        # Dependencies
├── src/                    # DocuChat components
│   ├── document_loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_db.py
│   ├── llm_client.py
│   ├── rag_pipeline.py
│   └── progress_manager.py
└── chroma_web/            # Vector database (created automatically)
```

## 🆘 Getting Help

1. **Check logs**: Look for error messages in the terminal
2. **Verify setup**: Run `python web/run_web_app.py` for diagnostics
3. **Test components**: Try the CLI version first (`python docuchat.py`)
4. **Check resources**: Monitor CPU, memory, and disk usage

## 🔄 Updates

To update to the latest version:
```bash
git pull origin main
pip install --upgrade -r requirements.txt
```

---

*DocuChat Web Application - Making document Q&A accessible and powerful* 🚀