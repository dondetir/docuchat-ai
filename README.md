# 🚀 DocuChat - AI-Powered Document Intelligence

> **Chat with your documents using AI** - Transform any collection of documents into an intelligent, conversational knowledge base that runs entirely on your local machine.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Powered by Ollama](https://img.shields.io/badge/LLM-Ollama-green.svg)](https://ollama.ai/)

---

## ✨ What is DocuChat?

DocuChat is a complete **Retrieval-Augmented Generation (RAG)** system that lets you have natural conversations with your document collections. Upload PDFs, Word documents, or text files, and instantly get precise, contextual answers powered by cutting-edge AI - all running locally for maximum privacy and control.

### 🎯 Perfect For:
- **Researchers** analyzing academic papers and reports
- **Legal professionals** reviewing contracts and case files  
- **Business teams** extracting insights from company documents
- **Students** studying textbooks and course materials
- **Anyone** who needs to quickly find information in large document collections

---

## 🎬 Quick Demo

> **Coming Soon**: Interactive demo GIF showing document upload → processing → chat in action

---

## ⚡ Quick Start (3 Steps)

Get DocuChat running in under 10 minutes:

### 1️⃣ Install Dependencies
```bash
# Clone the repository
git clone https://github.com/dondetir/docuchat-ai.git
cd docuchat-ai

# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2️⃣ Set up Ollama
```bash
# Install Ollama (https://ollama.ai)
curl -fsSL https://ollama.ai/install.sh | sh  # Linux/Mac
# Windows: Download from https://ollama.ai

# Start Ollama service
ollama serve

# Pull recommended model (in new terminal)
ollama pull gemma3:270m  # Fast, efficient model
```

### 3️⃣ Launch DocuChat
```bash
# Option A: Web Interface (Recommended)
python web/run_web_app.py
# Opens at http://localhost:7860

# Option B: Command Line
python docuchat.py /path/to/your/documents --chat
```

**That's it!** 🎉 You're now ready to chat with your documents.

---

## 🌟 Key Features

### 📄 **Multi-Format Document Support**
- **PDF** documents (with OCR capability)
- **Microsoft Word** (.docx) files
- **Plain text** (.txt) and Markdown (.md) files
- **Batch processing** of entire document folders

### 🔒 **100% Local & Private**
- No data sent to external servers
- All processing happens on your machine
- Your documents stay completely private
- Works offline after initial setup

### 🎨 **Modern Web Interface**
- Clean, intuitive Grok-like chat interface
- Real-time document processing progress
- Source attribution with every answer
- Mobile-responsive design
- Dark/light theme support

### ⚡ **High Performance**
- Parallel document processing
- Optimized embedding generation
- Smart chunk management
- Efficient vector similarity search

### 🐳 **Multiple Deployment Options**
- Standalone Python application
- Docker containerization
- Cloud deployment scripts
- Cross-platform compatibility

---

## 💻 Usage Examples

### Web Interface Workflow
1. **Upload Documents**: Enter your document folder path
2. **Process**: Watch real-time progress as documents are analyzed
3. **Chat**: Ask questions and get instant, sourced answers

### CLI Commands
```bash
# Process documents and start interactive chat
python docuchat.py ./my-documents --chat

# Process with custom settings
python docuchat.py ./docs --chunk-size 1500 --top-k 7 --verbose

# Just process documents (no chat)
python docuchat.py ./documents --rebuild
```

### Example Questions You Can Ask
- *"What are the main conclusions about climate change in these reports?"*
- *"How does the company handle customer complaints according to these policies?"*
- *"What safety procedures are mentioned across these manuals?"*
- *"Can you summarize the financial projections from Q3?"*
- *"What are the common themes in these research papers?"*

---

## 📋 Requirements & Compatibility

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB+ recommended
- **Storage**: 2GB for dependencies + your document storage
- **CPU**: Multi-core recommended for faster processing

### Supported Platforms
- ✅ **Windows** 10/11
- ✅ **macOS** 10.15+
- ✅ **Linux** (Ubuntu, CentOS, Debian, etc.)
- ✅ **Docker** environments

### Dependencies
- **Ollama**: Local LLM inference engine
- **ChromaDB**: Vector database for embeddings
- **Gradio**: Modern web interface framework
- **PyTorch**: Deep learning backend

---

## 🚀 Installation & Deployment

### Standard Installation
```bash
git clone https://github.com/dondetir/docuchat-ai.git
cd docuchat-ai
pip install -r requirements.txt
python web/run_web_app.py
```

### Docker Deployment
```bash
# Quick start with Docker Compose
docker-compose up -d

# Access at http://localhost:7860
# Ollama runs on http://localhost:11434
```

### Environment Configuration
```bash
# Optional: Configure via environment variables
export OLLAMA_BASE_URL="http://localhost:11434"
export DOCUCHAT_MODEL="gemma3:270m"
export GRADIO_SERVER_PORT=7860
```

### Recommended Models

| Model | Speed | Quality | RAM Usage | Best For |
|-------|-------|---------|-----------|----------|
| `gemma3:270m` | ⚡⚡⚡ | ⭐⭐⭐ | ~2GB | Quick answers, fast responses |
| `llama3.1:8b` | ⚡⚡ | ⭐⭐⭐⭐ | ~8GB | Balanced performance |
| `mistral:7b` | ⚡⚡ | ⭐⭐⭐⭐ | ~7GB | Complex reasoning |

---

## 🛠️ Troubleshooting

### Common Issues & Solutions

**🚫 "Ollama service not available"**
```bash
# Start Ollama service
ollama serve

# Verify model is installed
ollama list
ollama pull gemma3:270m
```

**🚫 "No supported documents found"**
- Verify folder path exists and contains PDF/DOCX/TXT files
- Check file permissions (readable by current user)
- Ensure files aren't corrupted or password-protected

**🚫 "Port 7860 already in use"**
```bash
# Use different port
python web/web_app.py --port 7861

# Or find and kill process using port
lsof -ti:7860 | xargs kill
```

**⚠️ Slow Performance**
- Use smaller chunk size (`--chunk-size 500`)
- Reduce retrieval count (`--top-k 3`)
- Switch to faster model (`gemma3:270m`)
- Close other resource-intensive applications

### Performance Optimization Tips
- **For large document sets**: Process in batches of 50-100 files
- **For faster responses**: Use `gemma3:270m` model
- **For better accuracy**: Use `llama3.1:8b` with `--top-k 8`
- **For memory constraints**: Set `--chunk-size 800`

---

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │───▶│  DocuChat RAG    │───▶│   Web/CLI UI    │
│  (PDF/DOCX/TXT) │    │     Pipeline     │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Core Components │
                    └──────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌─────────────┐    ┌─────────────────┐    ┌─────────────┐
│ Document    │    │ Embedding &     │    │ LLM Client  │
│ Loader      │    │ Vector Database │    │ (Ollama)    │
│             │    │                 │    │             │
│ • PDF       │    │ • Chunking      │    │ • Chat      │
│ • DOCX      │    │ • Embeddings    │    │ • Q&A       │
│ • TXT       │    │ • ChromaDB      │    │ • Context   │
└─────────────┘    └─────────────────┘    └─────────────┘
```

### Key Components
- **Document Loader**: Multi-format file parsing and text extraction
- **Chunking Engine**: Intelligent text segmentation with overlap
- **Embedding Generator**: Semantic vector generation using sentence-transformers
- **Vector Database**: High-performance similarity search with ChromaDB
- **LLM Client**: Local language model integration via Ollama
- **RAG Pipeline**: Orchestrated retrieval-augmented generation workflow

---

## 🤝 Contributing

We welcome contributions! Whether you're fixing bugs, adding features, or improving documentation, your help makes DocuChat better for everyone.

### Quick Contribution Guide
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

📖 **Read our full [Contributing Guide](CONTRIBUTING.md)** for detailed development setup, coding standards, and review process.

### Development Setup
```bash
# Clone and setup for development
git clone https://github.com/dondetir/docuchat-ai.git
cd docuchat-ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Start development server
python web/web_app.py
```

---

## 📄 License

DocuChat is released under the **MIT License**, making it free for both personal and commercial use.

**🙏 Attribution Appreciated**: While not legally required, we'd appreciate a mention of this repository if you use DocuChat in your projects. It helps others discover this tool!

See the [LICENSE](LICENSE) file for full details.

---

## 🎯 Roadmap

### Upcoming Features
- 📱 **Mobile App**: Native iOS and Android applications
- 🌐 **Multi-Language**: Support for non-English documents
- 🔗 **API Integration**: RESTful API for programmatic access
- 📊 **Analytics Dashboard**: Usage insights and performance metrics
- 🔄 **Document Synchronization**: Auto-update when documents change

### Community Requests
- 💬 **Conversation Memory**: Multi-turn contextual conversations
- 🎨 **Theme Customization**: Personalized UI themes
- 📈 **Advanced Search**: Filters, date ranges, document types
- 🔐 **User Authentication**: Multi-user support with access controls

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=dondetir/docuchat-ai&type=Date)](https://star-history.com/#dondetir/docuchat-ai&Date)

---

## 📞 Support & Community

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/dondetir/docuchat-ai/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/dondetir/docuchat-ai/discussions)
- 📚 **Documentation**: [Wiki](https://github.com/dondetir/docuchat-ai/wiki)
- 💬 **Community Chat**: Coming Soon

---

## 🙏 Acknowledgments

DocuChat is built on the shoulders of giants. Special thanks to:

- **[Ollama](https://ollama.ai/)** - Local LLM inference engine
- **[ChromaDB](https://www.trychroma.com/)** - Vector database for embeddings
- **[Sentence Transformers](https://www.sbert.net/)** - Semantic embedding models
- **[Gradio](https://gradio.app/)** - Beautiful web interfaces for ML
- **[PyPDF](https://pypdf.readthedocs.io/)** - PDF parsing library
- **[python-docx](https://python-docx.readthedocs.io/)** - Word document processing

---

<div align="center">

**Built with ❤️ for the AI community**

[⬆ Back to Top](#-docuchat---ai-powered-document-intelligence)

</div>