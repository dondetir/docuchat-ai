#!/usr/bin/env python3
"""
DocuChat Web Application

A production-ready Gradio web interface for DocuChat providing:
- Grok-like chat interface
- Document processing from folder path
- Real-time progress updates
- Source citations in responses
- Professional UI with error handling
"""

import sys
import os
import time
import logging
import threading
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import json

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import Gradio
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    print("❌ Gradio not available. Install with: pip install gradio")
    GRADIO_AVAILABLE = False

# Import DocuChat components
try:
    from document_loader import DocumentLoader
    from chunker import DocumentChunker
    from embeddings import EmbeddingGenerator
    from vector_db import VectorDatabase
    from llm_client import LLMClient
    from rag_pipeline import RAGPipeline
    from progress_manager import create_progress_manager
    DOCUCHAT_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ DocuChat components not available: {e}")
    DOCUCHAT_COMPONENTS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocuChatWebApp:
    """Main web application class for DocuChat."""
    
    def __init__(self):
        """Initialize the web application."""
        self.rag_pipeline: Optional[RAGPipeline] = None
        self.vector_db: Optional[VectorDatabase] = None
        self.is_processing = False
        self.processing_thread: Optional[threading.Thread] = None
        self.processing_progress = {"status": "idle", "progress": 0, "message": "", "error": None}
        self.chat_history: List[Dict[str, str]] = []
        self.current_folder_path = ""
        self.completion_message_shown = False
        
        # Application settings
        self.settings = {
            "chunk_size": 1000,
            "top_k": 5,
            "ollama_base_url": "http://localhost:11434",
            "ollama_model": "gemma3:270m",
            "vector_db_path": "./chroma",
            "collection_name": "docuchat_embeddings"
        }
        
        logger.info("DocuChatWebApp initialized")
    
    def validate_folder_path(self, folder_path: str) -> Tuple[bool, str]:
        """
        Validate that the folder path exists and contains supported documents.
        
        Args:
            folder_path: Path to folder containing documents
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not folder_path or not folder_path.strip():
            return False, "Please enter a folder path"
        
        folder_path = folder_path.strip()
        
        # Security: Prevent path traversal attacks
        if '..' in folder_path:
            return False, "Invalid folder path - path traversal detected"
        
        # Security: Block access to sensitive system directories
        sensitive_paths = ['/etc/', '/proc/', '/sys/', '/root/', '/boot/', 'C:\\Windows\\', 'C:\\System32\\']
        folder_lower = folder_path.lower()
        if any(folder_lower.startswith(sensitive.lower()) for sensitive in sensitive_paths):
            return False, "Access to system directories is not allowed"
        
        try:
            path = Path(folder_path).resolve()
        except (OSError, ValueError) as e:
            return False, f"Invalid folder path: {str(e)}"
        
        if not path.exists():
            return False, f"Folder does not exist: {folder_path}"
        
        if not path.is_dir():
            return False, f"Path is not a directory: {folder_path}"
        
        # Check for supported documents
        try:
            loader = DocumentLoader()
            supported_files = loader.discover_files(folder_path)
            
            if not supported_files:
                return False, "No supported documents found (PDF, DOCX, TXT files)"
            
            return True, f"Found {len(supported_files)} supported document(s)"
        
        except Exception as e:
            return False, f"Error scanning folder: {str(e)}"
    
    def validate_folder_path_ui(self, folder_path: str) -> str:
        """
        UI-specific folder path validation that returns formatted HTML status.
        
        Args:
            folder_path: Path to validate
            
        Returns:
            HTML formatted validation status
        """
        # Check for completion status first - Show stable completion message when processing is done
        status_info = self.get_processing_status()
        if status_info["status"] == "completed":
            # Only render completion message once to prevent blinking
            if not self.completion_message_shown:
                self.completion_message_shown = True
            
            processed_count = status_info.get('processed_files', 0)
            processing_time = status_info.get('processing_time', 0)
            return f"""
            <div id="completion-message-stable" style="
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                color: white;
                padding: 20px 24px;
                border-radius: 12px;
                margin: 15px 0;
                box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);
                border: 2px solid #10b981;
                text-align: center;
                font-weight: 600;
            ">
                <div style="font-size: 1.4em; margin-bottom: 15px;">
                    🎉 <strong>Processing Complete!</strong> 🎉
                </div>
                <div style="font-size: 1.1em; margin-bottom: 10px;">
                    📊 <strong>Results:</strong> Successfully processed {processed_count} documents
                </div>
                <div style="font-size: 1.1em; margin-bottom: 10px;">
                    ⏱️ <strong>Time:</strong> Completed in {processing_time:.1f} seconds
                </div>
                <div style="font-size: 1.2em; background: rgba(255,255,255,0.2); padding: 10px; border-radius: 8px; margin-top: 15px;">
                    💬 <strong>Next:</strong> Go to Chat tab to ask questions!
                </div>
            </div>
            """
        
        if not folder_path or not folder_path.strip():
            return """
            <div class="warning-box">
                <strong>⚠️ Folder Validation:</strong> Please enter a folder path above
            </div>
            """
        
        is_valid, message = self.validate_folder_path(folder_path)
        
        if is_valid:
            return f"""
            <div class="success-box">
                <strong>✅ Folder Validation:</strong> {message}<br>
                <strong>📁 Path:</strong> {folder_path}<br>
                <strong>🎯 Status:</strong> Ready to process!
            </div>
            """
        else:
            return f"""
            <div class="error-box">
                <strong>❌ Folder Validation Error:</strong> {message}<br>
                <strong>💡 Tip:</strong> Make sure the path exists and contains PDF, DOCX, or TXT files
            </div>
            """
    
    def process_documents_async(self, folder_path: str) -> None:
        """
        Process documents in a separate thread with progress updates.
        
        Args:
            folder_path: Path to folder containing documents
        """
        self.is_processing = True
        self.processing_progress = {
            "status": "initializing",
            "progress": 0,
            "message": "Initializing document processing...",
            "error": None,
            "total_files": 0,
            "processed_files": 0,
            "failed_files": 0
        }
        
        try:
            logger.info(f"Starting document processing for: {folder_path}")
            
            # Initialize components
            self.processing_progress["message"] = "Initializing components..."
            
            loader = DocumentLoader(verbose=False)
            chunker = DocumentChunker(
                chunk_size=self.settings["chunk_size"],
                chunk_overlap=200,
                verbose=False
            )
            embedding_generator = EmbeddingGenerator(
                model_name="all-MiniLM-L6-v2",
                batch_size=32,
                verbose=False
            )
            
            # Initialize vector database (rebuild for web interface)
            self.vector_db = VectorDatabase(
                persist_directory=self.settings["vector_db_path"],
                collection_name=self.settings["collection_name"],
                verbose=False,
                rebuild=True  # Always rebuild for fresh data
            )
            
            # Discover files
            self.processing_progress["message"] = "Discovering documents..."
            discovered_files = loader.discover_files(folder_path)
            
            if not discovered_files:
                raise Exception("No supported documents found in folder")
            
            self.processing_progress["total_files"] = len(discovered_files)
            
            # Process documents with progress tracking
            start_time = time.time()
            
            for i, file_path in enumerate(discovered_files):
                if not self.is_processing:  # Check for cancellation
                    break
                
                try:
                    # Update progress
                    filename = Path(file_path).name
                    self.processing_progress.update({
                        "status": "processing",
                        "progress": (i / len(discovered_files)) * 100,
                        "message": f"Processing: {filename}",
                        "current_file": filename,
                        "processed_files": i
                    })
                    
                    # Load document
                    result = loader.load_document(file_path)
                    if result is None:
                        logger.warning(f"Failed to load document: {file_path}")
                        self.processing_progress["failed_files"] += 1
                        continue
                    
                    filename, content = result
                    
                    # Update progress - chunking
                    self.processing_progress["message"] = f"Chunking: {Path(filename).name}"
                    
                    # Chunk document
                    chunks = list(chunker.chunk_document(
                        text=content,
                        source_file=filename,
                        metadata={'processed_at': time.time()}
                    ))
                    
                    if not chunks:
                        logger.warning(f"No chunks created for: {filename}")
                        self.processing_progress["failed_files"] += 1
                        continue
                    
                    # Update progress - embedding
                    self.processing_progress["message"] = f"Embedding: {Path(filename).name}"
                    
                    # Generate embeddings
                    embedded_chunks = embedding_generator.embed_chunks(chunks)
                    
                    if not embedded_chunks:
                        logger.warning(f"No embeddings generated for: {filename}")
                        self.processing_progress["failed_files"] += 1
                        continue
                    
                    # Update progress - storing
                    self.processing_progress["message"] = f"Storing: {Path(filename).name}"
                    
                    # Store in vector database
                    success = self.vector_db.add_chunks(embedded_chunks)
                    
                    if not success:
                        logger.warning(f"Failed to store chunks for: {filename}")
                        self.processing_progress["failed_files"] += 1
                        continue
                    
                    logger.info(f"Successfully processed: {filename}")
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    self.processing_progress["failed_files"] += 1
                    continue
            
            # Finalize processing
            processing_time = time.time() - start_time
            processed_count = len(discovered_files) - self.processing_progress["failed_files"]
            
            if processed_count == 0:
                raise Exception("Failed to process any documents successfully")
            
            # Initialize RAG pipeline
            self.processing_progress["message"] = "Initializing chat interface..."
            
            llm_client = LLMClient(
                base_url=self.settings["ollama_base_url"],
                model=self.settings["ollama_model"],
                timeout=60.0,
                verbose=False
            )
            
            # Test LLM availability
            if not llm_client.is_available():
                raise Exception(f"LLM service not available at {self.settings['ollama_base_url']}. Please ensure Ollama is running with {self.settings['ollama_model']} model.")
            
            # Create RAG pipeline
            self.rag_pipeline = RAGPipeline(
                embedding_generator=embedding_generator,
                vector_database=self.vector_db,
                llm_client=llm_client,
                verbose=False
            )
            
            # Test pipeline
            if not self.rag_pipeline.test_pipeline():
                raise Exception("RAG pipeline test failed")
            
            # Update final status
            self.processing_progress.update({
                "status": "completed",
                "progress": 100,
                "message": f"✅ Successfully processed {processed_count}/{len(discovered_files)} documents in {processing_time:.1f}s",
                "processed_files": processed_count,
                "processing_time": processing_time
            })
            
            self.current_folder_path = folder_path
            logger.info(f"Document processing completed successfully")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Document processing failed: {error_msg}")
            logger.error(traceback.format_exc())
            
            self.processing_progress.update({
                "status": "error",
                "progress": 0,
                "message": f"❌ Processing failed: {error_msg}",
                "error": error_msg
            })
            
        finally:
            self.is_processing = False
    
    def start_document_processing(self, selected_folder: str) -> str:
        """
        Start document processing in background thread.
        
        Args:
            selected_folder: Selected folder path from folder input field
            
        Returns:
            Status message
        """
        # Extract actual folder path from display text
        if not selected_folder or "No folder selected" in selected_folder or "Browser security" in selected_folder:
            return "❌ Please enter a valid folder path first"
        
        # Extract folder path if it's in the format "✅ /path (message)" or "❌ /path - Error: message"
        folder_path = selected_folder
        if selected_folder.startswith("✅ "):
            folder_path = selected_folder.split("(")[0].replace("✅ ", "").strip()
        elif selected_folder.startswith("❌ "):
            return "❌ Please fix the folder path error first"
        
        # Validate folder path
        is_valid, message = self.validate_folder_path(folder_path)
        if not is_valid:
            return f"❌ {message}"
        
        # Check if already processing
        if self.is_processing:
            return "⚠️ Document processing already in progress"
        
        # Clear previous chat history and reset completion flag
        self.chat_history.clear()
        self.completion_message_shown = False
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self.process_documents_async,
            args=(folder_path,),
            daemon=True
        )
        self.processing_thread.start()
        
        return f"🚀 Started processing documents from: {folder_path}"
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status for UI updates."""
        return self.processing_progress.copy()
    
    def cancel_processing(self) -> str:
        """Cancel ongoing document processing."""
        if not self.is_processing:
            return "No processing to cancel"
        
        self.is_processing = False
        self.processing_progress.update({
            "status": "cancelled",
            "message": "❌ Processing cancelled by user",
            "error": "Cancelled"
        })
        
        return "Processing cancelled"
    
    def add_user_message(self, question: str, history: List[List[str]]) -> Tuple[str, List[List[str]]]:
        """
        Immediately add user message to chat history and clear input.
        
        Args:
            question: User's question
            history: Chat history
            
        Returns:
            Tuple of (cleared_input, updated_history_with_user_message)
        """
        if not question or not question.strip():
            return "", history
            
        question = question.strip()
        
        # Add user message with temporary "thinking" response
        history.append([question, "🤔 Thinking..."])
        
        logger.info(f"Added user message: {question[:50]}...")
        return "", history
    
    def generate_ai_response(self, history: List[List[str]]) -> List[List[str]]:
        """
        Generate AI response for the last user message.
        
        Args:
            history: Chat history (last entry should have user question and "Thinking..." response)
            
        Returns:
            Updated history with AI response
        """
        if not history:
            return history
            
        # Get the last user question
        last_entry = history[-1]
        if len(last_entry) < 2:
            return history
            
        question = last_entry[0]
        
        # Check if RAG pipeline is ready
        if not self.rag_pipeline:
            error_msg = "❌ Please process documents first before asking questions"
            history[-1][1] = error_msg
            return history
        
        try:
            # Get answer using RAG pipeline
            result = self.rag_pipeline.answer_question(
                question=question,
                top_k=self.settings["top_k"],
                include_sources=True
            )
            
            # Modern AI Assistant Response Format
            response = f"**🤖 AI Assistant**\n\n{result.answer}"
            
            if result.sources and len(result.sources) > 0:
                response += "\n\n**📚 Sources:**\n"
                
                for i, source in enumerate(result.sources[:3], 1):
                    metadata = source.get('metadata', {})
                    source_file = metadata.get('source_file', 'Unknown')
                    file_name = Path(source_file).name
                    distance = source.get('distance', 0.0)
                    
                    # Simple confidence: anything under 1.5 is decent
                    if distance < 0.8:
                        confidence_emoji = "🎯"
                        confidence_text = "High"
                    elif distance < 1.2:
                        confidence_emoji = "👍"  
                        confidence_text = "Good"
                    else:
                        confidence_emoji = "📄"
                        confidence_text = "Related"
                    
                    # Clean content preview
                    content = source.get('content', '').strip()
                    if len(content) > 100:
                        content_preview = content[:100] + "..."
                    else:
                        content_preview = content
                    
                    response += f"\n{confidence_emoji} **{file_name}** ({confidence_text})\n"
                    response += f"   *\"{content_preview}\"*\n"
            
            # Simple footer
            response += f"\n*⚡ {result.processing_time:.2f}s"
            if result.tokens_per_second:
                response += f" • {result.tokens_per_second:.1f} tok/s"
            response += "*"
            
            # Update the last entry with the real response
            history[-1][1] = response
            
            logger.info(f"Generated AI response successfully: {question[:50]}...")
            
            return history
        
        except Exception as e:
            error_msg = f"❌ Error answering question: {str(e)}"
            history[-1][1] = error_msg
            logger.error(f"Error answering question: {e}")
            logger.error(traceback.format_exc())
            
            return history
    
    def clear_chat(self) -> List:
        """Clear chat history."""
        self.chat_history.clear()
        return []
    
    def get_database_info(self) -> str:
        """Get information about the current database."""
        if not self.vector_db:
            return """
            <div class="warning-box">
                <strong>⚠️ Database Status:</strong><br>
                No documents processed yet.<br><br>
                <strong>📝 Next Steps:</strong><br>
                1. Go to Process Documents tab<br>
                2. Select a folder with documents<br>
                3. Start processing<br>
                4. Return here to chat!
            </div>
            """
        
        try:
            db_info = self.vector_db.get_info()
            
            document_count = db_info.get('document_count', 0)
            total_chunks = db_info.get('total_chunks', 0)
            
            info_html = f"""
            <div class="success-box">
                <h4 style="margin-top: 0;">📊 Database Information</h4>
                
                <strong>📄 Documents:</strong> {document_count:,} files processed<br>
                <strong>🔍 Total Chunks:</strong> {total_chunks:,} searchable pieces<br>
            """
            
            if 'unique_source_files_sample' in db_info:
                files = db_info['unique_source_files_sample']
                if isinstance(files, list) and len(files) > 0:
                    info_html += f"<strong>📁 Source Files:</strong> {len(files)} files<br><br>"
                    
                    info_html += "<strong>📋 File List:</strong><br>"
                    info_html += "<ul style='margin: 5px 0; padding-left: 20px; font-size: 0.9em;'>"
                    
                    # Show sample of files
                    sample_files = files[:5]  # Show first 5 files
                    for file_path in sample_files:
                        file_name = Path(file_path).name
                        info_html += f"<li>{file_name}</li>"
                    
                    if len(files) > 5:
                        info_html += f"<li>... and {len(files) - 5} more files</li>"
                    
                    info_html += "</ul>"
            
            if self.current_folder_path:
                info_html += f"<br><strong>📂 Source Folder:</strong><br>{self.current_folder_path}"
            
            info_html += "</div>"
            
            return info_html
        
        except Exception as e:
            return f"""
            <div class="error-box">
                <strong>❌ Database Error:</strong> {str(e)}<br>
                <strong>💡 Tip:</strong> Try reprocessing your documents
            </div>
            """
    
    def update_settings(self, chunk_size: int, top_k: int, ollama_url: str, ollama_model: str) -> str:
        """Update application settings with HTML formatted response."""
        try:
            # Validate inputs
            if chunk_size < 100 or chunk_size > 5000:
                return """
                <div class="error-box">
                    <strong>❌ Validation Error:</strong> Chunk size must be between 100 and 5000<br>
                    <strong>💡 Tip:</strong> Use 500-1500 for most documents
                </div>
                """
            
            if top_k < 1 or top_k > 20:
                return """
                <div class="error-box">
                    <strong>❌ Validation Error:</strong> Top-K must be between 1 and 20<br>
                    <strong>💡 Tip:</strong> Use 3-7 for balanced results
                </div>
                """
            
            if not ollama_url.strip():
                return """
                <div class="error-box">
                    <strong>❌ Validation Error:</strong> Ollama URL cannot be empty<br>
                    <strong>💡 Default:</strong> http://localhost:11434
                </div>
                """
            
            if not ollama_model.strip():
                return """
                <div class="error-box">
                    <strong>❌ Validation Error:</strong> Ollama model cannot be empty<br>
                    <strong>💡 Suggestion:</strong> Try gemma2:2b for fast performance
                </div>
                """
            
            # Update settings
            old_settings = self.settings.copy()
            self.settings.update({
                "chunk_size": chunk_size,
                "top_k": top_k,
                "ollama_base_url": ollama_url.strip(),
                "ollama_model": ollama_model.strip()
            })
            
            # Create detailed success response
            changes = []
            if old_settings["chunk_size"] != chunk_size:
                changes.append(f"Chunk Size: {old_settings['chunk_size']} → {chunk_size}")
            if old_settings["top_k"] != top_k:
                changes.append(f"Top-K: {old_settings['top_k']} → {top_k}")
            if old_settings["ollama_base_url"] != ollama_url.strip():
                changes.append(f"Ollama URL: {old_settings['ollama_base_url']} → {ollama_url.strip()}")
            if old_settings["ollama_model"] != ollama_model.strip():
                changes.append(f"Model: {old_settings['ollama_model']} → {ollama_model.strip()}")
            
            changes_text = "<br>".join([f"• {change}" for change in changes]) if changes else "No changes detected"
            
            return f"""
            <div class="success-box">
                <strong>✅ Settings Updated Successfully!</strong><br>
                <strong>📝 Last Updated:</strong> Just now<br>
                <strong>🔄 Changes Made:</strong><br>
                {changes_text}<br><br>
                <strong>💡 Note:</strong> New settings will be used for future document processing
            </div>
            """
        
        except Exception as e:
            return f"""
            <div class="error-box">
                <strong>❌ Error updating settings:</strong> {str(e)}<br>
                <strong>💡 Tip:</strong> Try refreshing the page and try again
            </div>
            """
    
    def clear_database(self) -> str:
        """Clear all database data with HTML formatted response."""
        try:
            logger.info("Starting database clear operation")
            
            # Check if vector database is available
            if not self.vector_db:
                return """
                <div class="warning-box">
                    <strong>⚠️ No Database:</strong> No database to clear<br>
                    <strong>💡 Status:</strong> No documents have been processed yet
                </div>
                """
            
            # Get current database info before clearing
            db_info = self.vector_db.get_info()
            document_count = db_info.get('document_count', 0)
            
            if document_count == 0:
                return """
                <div class="info-box">
                    <strong>ℹ️ Database Already Empty:</strong> No documents to clear<br>
                    <strong>📊 Status:</strong> Database contains 0 documents
                </div>
                """
            
            # Perform database clear
            success = self.vector_db.clear_database()
            
            if success:
                # Reset RAG pipeline since database is cleared
                self.rag_pipeline = None
                self.current_folder_path = ""
                
                # Clear chat history
                self.chat_history.clear()
                
                return f"""
                <div class="success-box">
                    <strong>✅ Database Cleared Successfully!</strong><br>
                    <strong>📊 Documents Removed:</strong> {document_count:,} documents cleared<br>
                    <strong>🗂️ Collections:</strong> All data removed from chroma folder<br>
                    <strong>💬 Chat History:</strong> Cleared<br><br>
                    <strong>📝 Next Steps:</strong><br>
                    • Go to Process Documents tab<br>
                    • Select a folder with documents<br>
                    • Start processing to rebuild your database
                </div>
                """
            else:
                return """
                <div class="error-box">
                    <strong>❌ Clear Failed:</strong> Database clear operation failed<br>
                    <strong>💡 Tip:</strong> Try restarting the application
                </div>
                """
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Database clear failed: {error_msg}")
            return f"""
            <div class="error-box">
                <strong>❌ Clear Error:</strong> {error_msg}<br>
                <strong>💡 Suggestion:</strong> Check application logs and try restarting
            </div>
            """
    
    
    def create_gradio_interface(self) -> gr.Blocks:
        """Create the main Gradio interface with left sidebar navigation."""
        with gr.Blocks(
            title="DocuChat - AI-Powered Document Intelligence",
            theme=gr.themes.Soft(),
            css="""
                /* Global Styles - 2025 Modern Design */
                .gradio-container {
                    max-width: 1400px !important;
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: #fafbfc;
                }
                
                /* Modern Header with Subtle Gradient */
                .header-container {
                    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #ec4899 100%);
                    color: white;
                    padding: 24px 32px;
                    border-radius: 16px;
                    margin-bottom: 20px;
                    text-align: center;
                    box-shadow: 0 10px 40px rgba(79, 70, 229, 0.15);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }
                
                .header-container h1 {
                    font-size: 2.5em;
                    font-weight: 700;
                    margin: 0 0 15px 0;
                    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                
                .header-container p {
                    font-size: 1.2em;
                    margin: 0;
                    opacity: 0.95;
                    line-height: 1.6;
                }
                
                /* Sidebar Styles */
                .sidebar {
                    background: #fafbfc;
                    padding: 25px;
                    border-radius: 12px;
                    margin-right: 20px;
                    border: 1px solid #eaecef;
                    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
                }
                
                .sidebar h2 {
                    color: #24292f;
                    font-size: 1.3em;
                    margin-bottom: 20px;
                    text-align: center;
                    border-bottom: 2px solid #eaecef;
                    padding-bottom: 10px;
                }
                
                /* Navigation Button Styles */
                .nav-button {
                    margin-bottom: 12px !important;
                    width: 100% !important;
                    padding: 12px 16px !important;
                    border-radius: 8px !important;
                    font-weight: 500 !important;
                    transition: all 0.3s ease !important;
                    border: 1px solid #eaecef !important;
                    color: #656d76 !important;
                }
                
                .nav-button:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                }
                
                .nav-button-active {
                    background: #0969da !important;
                    color: white !important;
                    border-color: #0969da !important;
                    box-shadow: 0 4px 16px rgba(9, 105, 218, 0.3) !important;
                }
                
                /* Main Content Styles */
                .main-content {
                    padding: 30px;
                    background: #ffffff;
                    border-radius: 12px;
                    border: 1px solid #eaecef;
                    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
                }
                
                /* Status Box Styles */
                .status-box {
                    padding: 18px 24px;
                    border-radius: 10px;
                    margin: 15px 0;
                    font-weight: 500;
                    border-left: 4px solid transparent;
                }
                
                .success-box {
                    background: #f6ffed;
                    border-left-color: #1a7f37;
                    border: 1px solid #34d399;
                    color: #0f5132;
                }
                
                .error-box {
                    background: #fff5f5;
                    border-left-color: #dc2626;
                    border: 1px solid #fca5a5;
                    color: #991b1b;
                }
                
                .info-box {
                    background: #f0f9ff;
                    border-left-color: #0969da;
                    border: 1px solid #93c5fd;
                    color: #0550ae;
                }
                
                .warning-box {
                    background: #fffbeb;
                    border-left-color: #d97706;
                    border: 1px solid #fcd34d;
                    color: #92400e;
                }
                
                
                /* Modern Chat Interface */
                .chat-container {
                    height: 600px !important;
                    border: 1px solid #e5e7eb;
                    border-radius: 16px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
                    background: white;
                }
                
                /* AI Response Cards - 2025 Style */
                .chatbot .message.bot {
                    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                    border: 1px solid #e2e8f0;
                    border-radius: 12px;
                    padding: 16px;
                    margin: 8px 0;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
                }
                
                /* Form Control Styles */
                .form-control {
                    border-radius: 8px;
                    border: 1px solid #eaecef;
                    transition: all 0.3s ease;
                    color: #24292f;
                }
                
                .form-control:focus {
                    border-color: #0969da;
                    box-shadow: 0 0 0 3px rgba(9, 105, 218, 0.1);
                }
                
                /* Progress Bar Styles */
                .progress-container {
                    min-height: 150px;
                    background: #fafbfc;
                    border-radius: 10px;
                    padding: 20px;
                    border: 1px solid #eaecef;
                }
                
                /* Button Styles */
                .btn-primary {
                    background: #0969da;
                    border: none;
                    padding: 12px 24px;
                    border-radius: 8px;
                    font-weight: 600;
                    transition: all 0.3s ease;
                    color: white;
                }
                
                .btn-primary:hover {
                    background: #0550ae;
                    transform: translateY(-2px);
                    box-shadow: 0 8px 25px rgba(9, 105, 218, 0.3);
                }
                
                .btn-warning {
                    background: #dc2626 !important;
                    border: 1px solid #dc2626 !important;
                    color: white !important;
                    padding: 12px 24px;
                    border-radius: 8px;
                    font-weight: 600;
                    transition: all 0.3s ease;
                }
                
                .btn-warning:hover {
                    background: #b91c1c !important;
                    transform: translateY(-2px);
                    box-shadow: 0 8px 25px rgba(220, 38, 38, 0.3);
                }
                
                /* Section Headers */
                .section-header {
                    color: #24292f;
                    font-size: 1.8em;
                    font-weight: 700;
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 3px solid #eaecef;
                }
                
                /* Responsive Design */
                @media (max-width: 768px) {
                    .gradio-container {
                        max-width: 100% !important;
                        padding: 10px;
                    }
                    
                    .header-container {
                        padding: 20px;
                    }
                    
                    .header-container h1 {
                        font-size: 2em;
                    }
                }
            """
        ) as app:
            
            # Professional Header Section
            with gr.Row():
                gr.HTML(
                    """
                    <div class="header-container">
                        <h1>🧠 DocuChat Intelligence</h1>
                        <p>Transform documents into conversations • Ask anything • Get intelligent answers with sources</p>
                    </div>
                    """
                )
            
            # Main interface with left sidebar navigation
            with gr.Row():
                # Left sidebar for navigation
                with gr.Column(scale=1, elem_classes=["sidebar"]):
                    
                    nav_process_btn = gr.Button(
                        "📁 Process Documents", 
                        variant="primary",
                        elem_classes=["nav-button", "nav-button-active"]
                    )
                    nav_chat_btn = gr.Button(
                        "Chat", 
                        variant="secondary",
                        elem_classes=["nav-button"]
                    )
                    nav_settings_btn = gr.Button(
                        "⚙️ Settings", 
                        variant="secondary",
                        elem_classes=["nav-button"]
                    )
                    
                
                # Main content area
                with gr.Column(scale=4, elem_classes=["main-content"]):
                    
                    # Document Processing Content - Clean Option A Design
                    process_content = gr.Group(visible=True)
                    with process_content:
                        gr.HTML("<h1 class='section-header'>📁 Process Documents</h1>")
                        
                        # Clean Single Progress Card - Replaces ALL old status sections
                        with gr.Row():
                            with gr.Column(scale=2):
                                # Progress Card - Hidden by default, shows only during processing
                                progress_card = gr.HTML(
                                    """
                                    <div class="progress-card" style="
                                        border: 2px solid #3b82f6; 
                                        border-radius: 12px; 
                                        padding: 20px; 
                                        margin: 10px 0;
                                        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
                                        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                                    ">
                                        <h3 style="color: #1e40af; margin: 0 0 15px 0; display: flex; align-items: center;">
                                            🔄 Processing Documents...
                                        </h3>
                                        <div style="color: #1e40af; margin-bottom: 15px;">
                                            <strong>Initializing...</strong>
                                        </div>
                                    </div>
                                    """,
                                    elem_id="progress_card",
                                    visible=False  # Hidden by default
                                )
                                
                                # Folder Input
                                selected_folder_display = gr.Textbox(
                                    label="📂 Folder Path",
                                    value="",
                                    interactive=True,
                                    placeholder="Enter folder path (e.g., /home/user/documents)",
                                    lines=1,
                                    elem_classes=["form-control"]
                                )
                                
                                # Validation status - clean styling
                                path_validation_status = gr.HTML(
                                    """
                                    <div style="
                                        padding: 10px; 
                                        margin: 10px 0; 
                                        border-radius: 6px; 
                                        background: #fef3c7; 
                                        border-left: 4px solid #f59e0b;
                                    ">
                                        <strong style="color: #92400e;">⚠️ Validation:</strong> 
                                        <span style="color: #78350f;">Please enter a folder path above</span>
                                    </div>
                                    """,
                                    visible=True
                                )
                                
                                # Single State-based Action Button
                                with gr.Row():
                                    action_btn = gr.Button(
                                        "▶️ Start",
                                        variant="primary",
                                        size="lg",
                                        elem_id="action_btn"
                                    )
                            
                            # Compact sidebar - cleaner design
                            with gr.Column(scale=1):
                                gr.HTML(
                                    """
                                    <div style="
                                        background: #f9fafb; 
                                        border-radius: 8px; 
                                        padding: 15px; 
                                        border-left: 4px solid #3b82f6;
                                    ">
                                        <h4 style='color: #374151; margin: 0 0 10px 0;'>📄 Supported</h4>
                                        <div style="color: #6b7280; font-size: 14px; line-height: 1.5;">
                                            • <strong>PDF</strong> - Documents<br>
                                            • <strong>DOCX</strong> - Word files<br>
                                            • <strong>TXT</strong> - Text files
                                        </div>
                                        
                                        <h4 style='color: #374151; margin: 15px 0 10px 0;'>⚡ Features</h4>
                                        <div style="color: #6b7280; font-size: 14px; line-height: 1.5;">
                                            • Smart chunking<br>
                                            • AI embeddings<br>
                                            • Live progress
                                        </div>
                                    </div>
                                    """
                                )
                        
                    
                    # Chat Interface Content
                    chat_content = gr.Group(visible=False)
                    with chat_content:
                        # No header, no status - ultra minimal
                        
                        with gr.Row():
                            # Main chat area
                            with gr.Column(scale=4):
                                # No conversation header needed
                                
                                chatbot = gr.Chatbot(
                                    label="DocuChat Assistant",
                                    height=600,
                                    show_copy_button=True,
                                    elem_classes=["chat-container"],
                                    avatar_images=None,
                                    bubble_full_width=False
                                )
                                
                                # Input area with better styling
                                with gr.Row():
                                    question_input = gr.Textbox(
                                        label="Ask a question about your documents",
                                        placeholder="What would you like to know about your documents? Ask about concepts, summaries, specific details...",
                                        lines=2,
                                        scale=4,
                                        elem_classes=["form-control"]
                                    )
                                    ask_btn = gr.Button(
                                        "⚡",
                                        variant="primary",
                                        scale=1,
                                        elem_classes=["btn-primary"]
                                    )
                                
                            
                            # Enhanced sidebar with database info and tips
                            with gr.Column(scale=1):
                                gr.HTML("<h3 style='color: #374151; margin-bottom: 15px;'>📊 Database Info</h3>")
                                
                                database_info = gr.HTML(
                                    """
                                    <div class="warning-box">
                                        No documents processed yet • Process documents to begin chatting
                                    </div>
                                    """,
                                    elem_classes=["info-box"]
                                )
                                
                                # Enhanced tips section
                                gr.HTML(
                                    """
                                    <div class="info-box" style="margin-top: 20px;">
                                        <h4 style="margin-top: 0;">💡 Chat Tips & Best Practices</h4>
                                        
                                        <h5>🎯 Question Types:</h5>
                                        <ul style="margin: 5px 0; padding-left: 20px; font-size: 0.9em;">
                                            <li><strong>Summaries:</strong> "Summarize the main points"</li>
                                            <li><strong>Specific Facts:</strong> "What does it say about X?"</li>
                                            <li><strong>Comparisons:</strong> "Compare A and B"</li>
                                            <li><strong>Analysis:</strong> "What are the key themes?"</li>
                                        </ul>
                                        
                                        <h5>🔍 Better Results:</h5>
                                        <ul style="margin: 5px 0; padding-left: 20px; font-size: 0.9em;">
                                            <li>Be specific in your questions</li>
                                            <li>Ask follow-up questions</li>
                                            <li>Reference document names</li>
                                            <li>Use natural language</li>
                                        </ul>
                                        
                                        <h5>📚 Source Citations:</h5>
                                        <ul style="margin: 5px 0; padding-left: 20px; font-size: 0.9em;">
                                            <li>All answers include source references</li>
                                            <li>Click copy button to save responses</li>
                                            <li>Sources show relevant document sections</li>
                                        </ul>
                                    </div>
                                    """
                                )
                    
                    # Settings Content - SIMPLIFIED
                    settings_content = gr.Group(visible=False)
                    with settings_content:
                        gr.HTML("<h1 class='section-header'>⚙️ Settings</h1>")
                        
                        # Clean single-column layout
                        with gr.Column():
                            # AI Model
                            ollama_model_input = gr.Textbox(
                                label="🤖 AI Model",
                                value=self.settings["ollama_model"],
                                placeholder="gemma3:270m"
                            )
                            
                            # Ollama Server URL
                            ollama_url_input = gr.Textbox(
                                label="🌐 Ollama Server",
                                value=self.settings["ollama_base_url"],
                                placeholder="http://localhost:11434"
                            )
                            
                            # Document Processing - Two sliders in one row
                            gr.HTML("<h3>📄 Document Processing</h3>")
                            with gr.Row():
                                chunk_size_input = gr.Slider(
                                    minimum=100,
                                    maximum=5000,
                                    step=50,
                                    value=self.settings["chunk_size"],
                                    label="📄 Chunk Size",
                                    info="100-5000 characters"
                                )
                                
                                top_k_input = gr.Slider(
                                    minimum=1,
                                    maximum=20,
                                    step=1,
                                    value=self.settings["top_k"],
                                    label="🔍 Search Results",
                                    info="1-20 results"
                                )
                            
                            # Single save button and status
                            save_settings_btn = gr.Button(
                                "✅ Save Settings",
                                variant="primary",
                                size="lg"
                            )
                            
                            settings_status = gr.HTML("", elem_classes=["status-box"])
                            
                            # Database Management Section
                            gr.HTML("<h3>🗂️ Database Management</h3>")
                            gr.HTML("""
                                <div class="warning-box" style="margin-bottom: 15px;">
                                    <strong>⚠️ Warning:</strong> This will permanently delete all processed documents and embeddings from the chroma folder.<br>
                                    <strong>💡 Use Case:</strong> Clear the database when you want to start fresh or free up storage space.
                                </div>
                            """)
                            
                            clear_db_btn = gr.Button(
                                "🗑️ Clear Database",
                                variant="secondary",
                                size="lg",
                                elem_classes=["btn-warning"]
                            )
                            
                            clear_db_status = gr.HTML("", elem_classes=["status-box"])
            
            # Navigation event handlers with improved active state management
            def show_process_tab():
                return (
                    gr.update(visible=True),   # process_content
                    gr.update(visible=False),  # chat_content
                    gr.update(visible=False),  # settings_content
                    gr.update(value="📁 Process Documents", variant="primary", elem_classes=["nav-button", "nav-button-active"]),   # nav_process_btn
                    gr.update(value="Chat", variant="secondary", elem_classes=["nav-button"]),                       # nav_chat_btn
                    gr.update(value="⚙️ Settings", variant="secondary", elem_classes=["nav-button"])                        # nav_settings_btn
                )
            
            def show_chat_tab():
                return (
                    gr.update(visible=False),  # process_content
                    gr.update(visible=True),   # chat_content
                    gr.update(visible=False),  # settings_content
                    gr.update(value="📁 Process Documents", variant="secondary", elem_classes=["nav-button"]),                       # nav_process_btn
                    gr.update(value="Chat", variant="primary", elem_classes=["nav-button", "nav-button-active"]),   # nav_chat_btn
                    gr.update(value="⚙️ Settings", variant="secondary", elem_classes=["nav-button"])                        # nav_settings_btn
                )
            
            def show_settings_tab():
                return (
                    gr.update(visible=False),  # process_content
                    gr.update(visible=False),  # chat_content
                    gr.update(visible=True),   # settings_content
                    gr.update(value="📁 Process Documents", variant="secondary", elem_classes=["nav-button"]),                       # nav_process_btn
                    gr.update(value="Chat", variant="secondary", elem_classes=["nav-button"]),                       # nav_chat_btn
                    gr.update(value="⚙️ Settings", variant="primary", elem_classes=["nav-button", "nav-button-active"])    # nav_settings_btn
                )
            
            # Set up navigation event handlers
            nav_process_btn.click(
                fn=show_process_tab,
                outputs=[process_content, chat_content, settings_content, nav_process_btn, nav_chat_btn, nav_settings_btn]
            )
            
            nav_chat_btn.click(
                fn=show_chat_tab,
                outputs=[process_content, chat_content, settings_content, nav_process_btn, nav_chat_btn, nav_settings_btn]
            )
            
            nav_settings_btn.click(
                fn=show_settings_tab,
                outputs=[process_content, chat_content, settings_content, nav_process_btn, nav_chat_btn, nav_settings_btn]
            )
            
            
            # Event handlers
            
            # Folder path validation
            selected_folder_display.change(
                fn=self.validate_folder_path_ui,
                inputs=[selected_folder_display],
                outputs=[path_validation_status]
            )
            
            # State-based action handler
            def handle_action_btn(folder_path):
                """Handle both start and stop actions based on current state."""
                if self.is_processing:
                    # Stop processing
                    result = self.cancel_processing()
                else:
                    # Start processing
                    result = self.start_document_processing(folder_path)
                
                return update_progress_card(), update_action_button()
            
            action_btn.click(
                fn=handle_action_btn,
                inputs=[selected_folder_display],
                outputs=[progress_card, action_btn]
            )
            
            # Auto-update processing status with enhanced HTML formatting
            def update_processing_status():
                status_info = self.get_processing_status()
                
                if status_info["status"] == "idle":
                    overview = """
                    <div class="info-box">
                        <strong>ℹ️ Status:</strong> Ready to process documents<br>
                        <strong>📊 Progress:</strong> 0% - Waiting for folder selection
                    </div>
                    """
                    detailed = """
                    <div class="info-box">
                        <strong>🎯 Current Status:</strong> Ready to process documents<br>
                        <strong>📂 Folder:</strong> No folder selected<br>
                        <strong>📑 Documents:</strong> 0 found<br>
                        <strong>⏱️ Processing Time:</strong> Not started
                    </div>
                    """
                    details_visible = False
                    
                elif status_info["status"] == "completed":
                    overview = f"""
                    <div class="success-box">
                        <strong>✅ Status:</strong> Processing completed successfully!<br>
                        <strong>📊 Progress:</strong> 100% - Ready to chat
                    </div>
                    """
                    detailed = f"""
                    <div class="success-box">
                        <strong>🎯 Current Status:</strong> {status_info['message']}<br>
                        <strong>📂 Folder:</strong> {self.current_folder_path}<br>
                        <strong>📑 Documents:</strong> {status_info.get('processed_files', 0)} processed successfully<br>
                        <strong>⏱️ Processing Time:</strong> {status_info.get('processing_time', 0):.1f}s
                    </div>
                    """
                    details_visible = True
                    
                elif status_info["status"] == "error":
                    overview = f"""
                    <div class="error-box">
                        <strong>❌ Status:</strong> Processing failed<br>
                        <strong>📊 Progress:</strong> {status_info.get('progress', 0)}% - Error occurred
                    </div>
                    """
                    detailed = f"""
                    <div class="error-box">
                        <strong>🎯 Current Status:</strong> {status_info['message']}<br>
                        <strong>📂 Folder:</strong> {self.current_folder_path or 'N/A'}<br>
                        <strong>📑 Documents:</strong> {status_info.get('processed_files', 0)} processed, {status_info.get('failed_files', 0)} failed<br>
                        <strong>⏱️ Processing Time:</strong> Failed
                    </div>
                    """
                    details_visible = True
                    
                else:  # processing, initializing, etc.
                    progress = status_info.get('progress', 0)
                    overview = f"""
                    <div class="info-box">
                        <strong>🚀 Status:</strong> Processing in progress...<br>
                        <strong>📊 Progress:</strong> {progress:.1f}% - {status_info.get('message', 'Processing...')}
                    </div>
                    """
                    current_file = status_info.get('current_file', 'N/A')
                    detailed = f"""
                    <div class="info-box">
                        <strong>🎯 Current Status:</strong> {status_info['message']}<br>
                        <strong>📂 Folder:</strong> {self.current_folder_path or 'N/A'}<br>
                        <strong>📄 Current File:</strong> {current_file}<br>
                        <strong>📑 Progress:</strong> {status_info.get('processed_files', 0)}/{status_info.get('total_files', 0)} files<br>
                        <strong>⏱️ Processing Time:</strong> In progress
                    </div>
                    """
                    details_visible = True
                
                # Format processing details as HTML
                details_html = f"""
                <div class="info-box" id="processing_details">
                    <h4>📈 Processing Details</h4>
                    <div id="details_content">
                        <strong>Total Files:</strong> {status_info.get('total_files', 0)}<br>
                        <strong>Processed:</strong> {status_info.get('processed_files', 0)}<br>
                        <strong>Failed:</strong> {status_info.get('failed_files', 0)}<br>
                        <strong>Progress:</strong> {status_info.get('progress', 0):.1f}%
                    </div>
                </div>
                """ if details_visible else ""
                
                return overview, detailed, gr.update(value=details_html, visible=details_visible)
            
            # Progress card update function - Controls both content and visibility
            def update_progress_card():
                """Update progress card content and visibility based on processing status."""
                status_info = self.get_processing_status()
                
                # Base card style template
                card_style = """
                    border: 2px solid {border_color}; 
                    border-radius: 12px; 
                    padding: 20px; 
                    margin: 10px 0;
                    background: {background};
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                """
                
                # Hide progress card when idle, completed, or error - Show only during processing
                if status_info["status"] in ["processing", "starting", "initializing"]:
                    progress_percentage = status_info.get('progress', 0)
                    current_file = status_info.get('current_file', 'Processing...')
                    processed = status_info.get('processed_files', 0)
                    total = status_info.get('total_files', 0)
                    
                    # Progress bar calculation
                    progress_width = max(5, int(progress_percentage))  # Minimum 5% visible
                    
                    content = f"""
                    <div class="progress-card" style="{card_style.format(
                        border_color='#4f46e5', 
                        background='linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%)'
                    )}">
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;">
                            <h3 style="color: #4338ca; margin: 0; font-size: 1.1em; font-weight: 600;">
                                ⚡ Processing Intelligence
                            </h3>
                            <span style="color: #4f46e5; font-weight: 700; font-size: 1.2em;">{progress_percentage:.0f}%</span>
                        </div>
                        
                        <!-- Modern Progress Bar with Glow -->
                        <div style="background: #e5e7eb; height: 8px; border-radius: 12px; margin: 12px 0; overflow: hidden; position: relative;">
                            <div style="
                                background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 50%, #ec4899 100%); 
                                height: 100%; 
                                width: {progress_width}%; 
                                border-radius: 12px;
                                transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                                box-shadow: 0 0 20px rgba(79, 70, 229, 0.4);
                                position: relative;
                            ">
                                <div style="
                                    position: absolute;
                                    top: 0; right: 0; bottom: 0; width: 20px;
                                    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.6));
                                    animation: shimmer 2s infinite;
                                "></div>
                            </div>
                        </div>
                        
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 16px 0;">
                            <div style="background: rgba(255,255,255,0.7); padding: 8px 12px; border-radius: 8px;">
                                <div style="color: #6b7280; font-size: 12px; font-weight: 500;">CURRENT</div>
                                <div style="color: #374151; font-size: 14px; font-weight: 600; margin-top: 2px;">{current_file[:20]}{'...' if len(current_file) > 20 else ''}</div>
                            </div>
                            <div style="background: rgba(255,255,255,0.7); padding: 8px 12px; border-radius: 8px;">
                                <div style="color: #6b7280; font-size: 12px; font-weight: 500;">FILES</div>
                                <div style="color: #374151; font-size: 14px; font-weight: 600; margin-top: 2px;">{processed} of {total}</div>
                            </div>
                        </div>
                        
                        <style>
                        @keyframes shimmer {{
                            0% {{ transform: translateX(-100%); }}
                            100% {{ transform: translateX(400%); }}
                        }}
                        </style>
                    </div>
                    """
                    
                    return gr.update(value=content, visible=True)  # Show during processing
                    
                else:  # idle, completed, error, cancelled - Hide progress card
                    return gr.update(visible=False)  # Hidden when not processing
            
            # Update action button based on processing state
            def update_action_button():
                """Update action button text and appearance based on processing state."""
                if self.is_processing:
                    return gr.update(
                        value="⏹️ Stop",
                        variant="secondary",
                        interactive=True
                    )
                else:
                    return gr.update(
                        value="▶️ Start",
                        variant="primary",
                        interactive=True
                    )
            
            # Timer - updates progress card visibility, action button, and validation status
            def conditional_update_ui():
                """Update progress card visibility and action button based on processing state."""
                status_info = self.get_processing_status()
                
                # Always update UI during any status change
                return update_progress_card(), update_action_button()
            
            timer = gr.Timer(value=1.5)  # 1.5s interval for responsive UI
            timer.tick(
                fn=conditional_update_ui,
                outputs=[progress_card, action_btn]
            )
            
            # Smart validation status updates - only refresh when status actually changes
            def update_validation_status():
                """Update validation status only when processing status changes."""
                status_info = self.get_processing_status()
                folder_path = selected_folder_display.value if hasattr(selected_folder_display, 'value') else ""
                
                # Only update if status changed or if we're not in a stable completed state
                if not hasattr(update_validation_status, 'last_status'):
                    update_validation_status.last_status = None
                
                current_status = status_info["status"]
                
                # If status hasn't changed and we're in completed state, don't refresh
                if (update_validation_status.last_status == current_status and 
                    current_status == "completed"):
                    return gr.update()  # No update needed
                
                # Update last status
                update_validation_status.last_status = current_status
                
                return self.validate_folder_path_ui(folder_path)
            
            validation_timer = gr.Timer(value=2)  # 2s interval for validation updates
            validation_timer.tick(
                fn=update_validation_status,
                outputs=[path_validation_status]
            )
            
            # Chat functionality - Two-step process for immediate feedback
            def handle_question_submit(question, history):
                # Step 1: Add user message immediately
                cleared_input, updated_history = self.add_user_message(question, history)
                return cleared_input, updated_history
                
            def handle_ai_response(history):
                # Step 2: Generate AI response
                return self.generate_ai_response(history)
            
            ask_btn.click(
                fn=handle_question_submit,
                inputs=[question_input, chatbot],
                outputs=[question_input, chatbot]
            ).then(
                fn=handle_ai_response,
                inputs=[chatbot],
                outputs=[chatbot]
            )
            
            question_input.submit(
                fn=handle_question_submit,
                inputs=[question_input, chatbot],
                outputs=[question_input, chatbot]
            ).then(
                fn=handle_ai_response,
                inputs=[chatbot],
                outputs=[chatbot]
            )
            
            
            # Update database info only when needed
            def conditional_refresh_database_info():
                """Only refresh database info if processing recently completed."""
                status_info = self.get_processing_status()
                
                # Only refresh database info during or after processing
                if status_info["status"] in ["processing", "completed"]:
                    return self.get_database_info()
                else:
                    # No refresh needed when idle
                    return gr.update()
            
            database_timer = gr.Timer(value=5)  # Reduced frequency from 2s to 5s
            database_timer.tick(
                fn=conditional_refresh_database_info,
                outputs=[database_info]
            )
            
            # Save settings
            save_settings_btn.click(
                fn=self.update_settings,
                inputs=[chunk_size_input, top_k_input, ollama_url_input, ollama_model_input],
                outputs=[settings_status]
            )
            
            # Clear database
            clear_db_btn.click(
                fn=self.clear_database,
                outputs=[clear_db_status]
            )
        
        return app


def main():
    """Main application entry point."""
    if not GRADIO_AVAILABLE:
        print("❌ Gradio is required. Install with: pip install gradio")
        return 1
    
    if not DOCUCHAT_COMPONENTS_AVAILABLE:
        print("❌ DocuChat components not available. Install requirements: pip install -r requirements.txt")
        return 1
    
    try:
        # Create web application
        app = DocuChatWebApp()
        
        # Create Gradio interface
        interface = app.create_gradio_interface()
        
        # Launch the application
        print("🚀 Starting DocuChat Web Application...")
        print("📖 Open your browser and navigate to the provided URL")
        print("💡 Make sure Ollama is running: ollama serve")
        print("🔧 Pull required model: ollama pull gemma3:270m")
        
        # Launch with public sharing disabled by default for security
        interface.launch(
            server_name="127.0.0.1",  # Local only by default
            server_port=7860,
            share=False,  # Set to True for public sharing (not recommended)
            debug=False,
            show_error=True,
            inbrowser=True  # Auto-open browser
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Application error: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())