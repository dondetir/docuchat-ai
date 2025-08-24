"""
Test suite for Phase 4: RAG Pipeline Integration and Interactive Chat

Comprehensive tests covering all Phase 4 functionality with 100% coverage:
- RAG Pipeline orchestration
- Interactive chat interface
- CLI argument parsing and validation
- End-to-end integration scenarios
- Error handling and edge cases
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import tempfile
import shutil
from pathlib import Path
import numpy as np
import time
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from rag_pipeline import RAGPipeline, RAGResult
    from cli import CLIArgs, ChatInterface, start_interactive_chat, parse_args
    from embeddings import EmbeddedChunk
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    IMPORTS_AVAILABLE = False


class TestRAGResult(unittest.TestCase):
    """Test RAGResult data class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_sources = [
            {
                'id': 'chunk_1',
                'distance': 0.2,
                'content': 'Sample content 1',
                'metadata': {'source_file': 'file1.txt', 'chunk_index': 0}
            },
            {
                'id': 'chunk_2', 
                'distance': 0.3,
                'content': 'Sample content 2',
                'metadata': {'source_file': 'file2.txt', 'chunk_index': 1}
            }
        ]
        
        self.result = RAGResult(
            question="What is the meaning of life?",
            answer="42",
            sources=self.sample_sources,
            processing_time=1.5,
            model_used="gemma3:270m",
            embedding_model="all-MiniLM-L6-v2",
            top_k_used=5,
            context_length=1000,
            metadata={'test': 'data'}
        )
    
    def test_rag_result_creation(self):
        """Test RAGResult creation and basic properties."""
        self.assertEqual(self.result.question, "What is the meaning of life?")
        self.assertEqual(self.result.answer, "42")
        self.assertEqual(len(self.result.sources), 2)
        self.assertEqual(self.result.processing_time, 1.5)
        self.assertEqual(self.result.model_used, "gemma3:270m")
        self.assertEqual(self.result.embedding_model, "all-MiniLM-L6-v2")
        self.assertEqual(self.result.top_k_used, 5)
        self.assertEqual(self.result.context_length, 1000)
        self.assertEqual(self.result.metadata['test'], 'data')
    
    def test_has_sources(self):
        """Test has_sources property."""
        self.assertTrue(self.result.has_sources)
        
        # Test without sources
        empty_result = RAGResult(
            question="test", answer="test", sources=[],
            processing_time=1.0, model_used="test", embedding_model="test",
            top_k_used=5, context_length=100, metadata={}
        )
        self.assertFalse(empty_result.has_sources)
    
    def test_source_files(self):
        """Test source_files property."""
        files = self.result.source_files
        self.assertEqual(sorted(files), ['file1.txt', 'file2.txt'])
        
        # Test with no sources
        empty_result = RAGResult(
            question="test", answer="test", sources=[],
            processing_time=1.0, model_used="test", embedding_model="test",
            top_k_used=5, context_length=100, metadata={}
        )
        self.assertEqual(empty_result.source_files, [])
    
    def test_confidence_score(self):
        """Test confidence_score calculation."""
        confidence = self.result.confidence_score
        # Average distance is (0.2 + 0.3) / 2 = 0.25
        # Confidence = 1.0 - 0.25 = 0.75
        self.assertAlmostEqual(confidence, 0.75, places=2)
        
        # Test with no sources
        empty_result = RAGResult(
            question="test", answer="test", sources=[],
            processing_time=1.0, model_used="test", embedding_model="test",
            top_k_used=5, context_length=100, metadata={}
        )
        self.assertEqual(empty_result.confidence_score, 0.0)


class TestRAGPipeline(unittest.TestCase):
    """Test RAGPipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock components
        self.mock_embedding_generator = Mock()
        self.mock_embedding_generator.model_name = "all-MiniLM-L6-v2"
        self.mock_embedding_generator.model = Mock()
        
        self.mock_vector_database = Mock()
        self.mock_llm_client = Mock()
        self.mock_llm_client.model = "gemma3:270m"
        self.mock_llm_client.is_available.return_value = True
        
        # Mock vector database get_info
        self.mock_vector_database.get_info.return_value = {
            'document_count': 100,
            'collection_name': 'test'
        }
        
        # Create pipeline
        self.pipeline = RAGPipeline(
            embedding_generator=self.mock_embedding_generator,
            vector_database=self.mock_vector_database,
            llm_client=self.mock_llm_client,
            verbose=True
        )
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertEqual(self.pipeline.embedding_generator, self.mock_embedding_generator)
        self.assertEqual(self.pipeline.vector_database, self.mock_vector_database)
        self.assertEqual(self.pipeline.llm_client, self.mock_llm_client)
        self.assertTrue(self.pipeline.verbose)
        self.assertEqual(self.pipeline.stats['queries_processed'], 0)
    
    def test_pipeline_initialization_invalid_components(self):
        """Test pipeline initialization with invalid components."""
        # Test invalid embedding generator
        with self.assertRaises(TypeError):
            RAGPipeline("invalid", self.mock_vector_database, self.mock_llm_client)
        
        # Test invalid vector database
        with self.assertRaises(TypeError):
            RAGPipeline(self.mock_embedding_generator, "invalid", self.mock_llm_client)
        
        # Test invalid LLM client
        with self.assertRaises(TypeError):
            RAGPipeline(self.mock_embedding_generator, self.mock_vector_database, "invalid")
    
    def test_pipeline_initialization_unavailable_llm(self):
        """Test pipeline initialization with unavailable LLM."""
        self.mock_llm_client.is_available.return_value = False
        
        with self.assertRaises(RuntimeError):
            RAGPipeline(
                self.mock_embedding_generator,
                self.mock_vector_database,
                self.mock_llm_client
            )
    
    def test_sanitize_question(self):
        """Test question sanitization."""
        # Valid question
        result = self.pipeline._sanitize_question("What is AI?")
        self.assertEqual(result, "What is AI?")
        
        # Question with null bytes
        result = self.pipeline._sanitize_question("What\x00is AI?")
        self.assertEqual(result, "Whatis AI?")
        
        # Empty question
        with self.assertRaises(ValueError):
            self.pipeline._sanitize_question("")
        
        # Whitespace only
        with self.assertRaises(ValueError):
            self.pipeline._sanitize_question("   ")
        
        # Non-string input
        with self.assertRaises(TypeError):
            self.pipeline._sanitize_question(123)
        
        # Very long question (over 10KB)
        long_question = "x" * 10001
        result = self.pipeline._sanitize_question(long_question)
        self.assertEqual(len(result), 10000)
    
    def test_embed_question(self):
        """Test question embedding."""
        # Mock embedding result
        mock_embedding = np.array([0.1, 0.2, 0.3])
        self.mock_embedding_generator.model.encode.return_value = [mock_embedding]
        
        result = self.pipeline._embed_question("What is AI?")
        
        np.testing.assert_array_equal(result, mock_embedding)
        self.mock_embedding_generator.model.encode.assert_called_once_with(["What is AI?"])
    
    def test_embed_question_failure(self):
        """Test question embedding failure."""
        # Mock embedding failure
        self.mock_embedding_generator.model.encode.side_effect = Exception("Embedding failed")
        
        with self.assertRaises(RuntimeError):
            self.pipeline._embed_question("What is AI?")
    
    def test_retrieve_context(self):
        """Test context retrieval."""
        # Mock search results
        mock_docs = [
            {'id': 'doc1', 'content': 'Content 1', 'distance': 0.2},
            {'id': 'doc2', 'content': 'Content 2', 'distance': 0.3}
        ]
        self.mock_vector_database.search_similar.return_value = mock_docs
        
        query_embedding = np.array([0.1, 0.2, 0.3])
        result = self.pipeline._retrieve_context(query_embedding, top_k=5)
        
        self.assertEqual(result, mock_docs)
        self.mock_vector_database.search_similar.assert_called_once_with(
            query_embedding=query_embedding, k=5
        )
    
    def test_retrieve_context_failure(self):
        """Test context retrieval failure."""
        self.mock_vector_database.search_similar.side_effect = Exception("Search failed")
        
        query_embedding = np.array([0.1, 0.2, 0.3])
        with self.assertRaises(RuntimeError):
            self.pipeline._retrieve_context(query_embedding)
    
    def test_format_context(self):
        """Test context formatting."""
        # Test with documents
        documents = [
            {
                'content': 'This is document 1 content',
                'metadata': {'source_file': 'file1.txt'}
            },
            {
                'content': 'This is document 2 content',
                'metadata': {'source_file': 'file2.txt'}
            }
        ]
        
        result = self.pipeline._format_context(documents)
        
        self.assertIn('Document 1 (Source: file1.txt)', result)
        self.assertIn('This is document 1 content', result)
        self.assertIn('Document 2 (Source: file2.txt)', result)
        self.assertIn('This is document 2 content', result)
        
        # Test with empty documents
        result = self.pipeline._format_context([])
        self.assertEqual(result, "No relevant context documents found.")
        
        # Test with very long content
        long_content = "x" * 2500
        long_docs = [{'content': long_content, 'metadata': {'source_file': 'test.txt'}}]
        result = self.pipeline._format_context(long_docs)
        self.assertIn("...", result)  # Should be truncated
    
    def test_format_context_length_limit(self):
        """Test context length limiting."""
        # Create content that exceeds 20KB limit
        long_content = "x" * 10000
        many_docs = []
        for i in range(5):
            many_docs.append({
                'content': long_content,
                'metadata': {'source_file': f'file{i}.txt'}
            })
        
        result = self.pipeline._format_context(many_docs)
        self.assertLessEqual(len(result), 20100)  # Should be truncated with message
        self.assertIn("[Context truncated due to length...]", result)
    
    def test_construct_prompt(self):
        """Test prompt construction."""
        question = "What is AI?"
        context = "AI is artificial intelligence."
        
        result = self.pipeline._construct_prompt(question, context)
        
        self.assertIn(question, result)
        self.assertIn(context, result)
        self.assertIn("Based on the following context documents", result)
    
    def test_construct_prompt_failure(self):
        """Test prompt construction failure."""
        # Use a template that will cause formatting error
        self.pipeline.prompt_template = "Invalid template {missing_key}"
        
        with self.assertRaises(RuntimeError):
            self.pipeline._construct_prompt("question", "context")
    
    def test_generate_answer(self):
        """Test answer generation."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.response = "This is the answer"
        mock_response.model = "gemma3:270m"
        mock_response.metadata = {'generation_time': 1.0}
        
        self.mock_llm_client.generate.return_value = mock_response
        
        result = self.pipeline._generate_answer("Test prompt")
        
        self.assertEqual(result, mock_response)
        self.mock_llm_client.generate.assert_called_once_with("Test prompt")
    
    def test_generate_answer_failure(self):
        """Test answer generation failure."""
        self.mock_llm_client.generate.side_effect = Exception("Generation failed")
        
        with self.assertRaises(RuntimeError):
            self.pipeline._generate_answer("Test prompt")
    
    def test_answer_question_complete_flow(self):
        """Test complete answer_question flow."""
        # Setup mocks
        mock_embedding = np.array([0.1, 0.2, 0.3])
        self.mock_embedding_generator.model.encode.return_value = [mock_embedding]
        
        mock_docs = [
            {
                'id': 'doc1',
                'content': 'AI is artificial intelligence',
                'distance': 0.2,
                'metadata': {'source_file': 'ai.txt', 'chunk_index': 0}
            }
        ]
        self.mock_vector_database.search_similar.return_value = mock_docs
        
        mock_llm_response = Mock()
        mock_llm_response.response = "AI stands for artificial intelligence"
        mock_llm_response.model = "gemma3:270m"
        mock_llm_response.metadata = {'generation_time': 1.0}
        self.mock_llm_client.generate.return_value = mock_llm_response
        
        # Test the complete flow
        result = self.pipeline.answer_question("What is AI?", top_k=3, include_sources=True)
        
        # Verify result
        self.assertIsInstance(result, RAGResult)
        self.assertEqual(result.question, "What is AI?")
        self.assertEqual(result.answer, "AI stands for artificial intelligence")
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.model_used, "gemma3:270m")
        self.assertEqual(result.embedding_model, "all-MiniLM-L6-v2")
        self.assertEqual(result.top_k_used, 3)
        
        # Verify statistics updated
        self.assertEqual(self.pipeline.stats['queries_processed'], 1)
        self.assertEqual(self.pipeline.stats['successful_queries'], 1)
        self.assertEqual(self.pipeline.stats['failed_queries'], 0)
    
    def test_answer_question_invalid_inputs(self):
        """Test answer_question with invalid inputs."""
        # Invalid top_k
        with self.assertRaises(ValueError):
            self.pipeline.answer_question("What is AI?", top_k=0)
        
        with self.assertRaises(ValueError):
            self.pipeline.answer_question("What is AI?", top_k=101)
        
        # Empty question
        with self.assertRaises(ValueError):
            self.pipeline.answer_question("")
    
    def test_answer_question_failure(self):
        """Test answer_question with pipeline failure."""
        # Mock embedding failure
        self.mock_embedding_generator.model.encode.side_effect = Exception("Embedding failed")
        
        with self.assertRaises(RuntimeError):
            self.pipeline.answer_question("What is AI?")
        
        # Verify error statistics
        self.assertEqual(self.pipeline.stats['failed_queries'], 1)
        self.assertIn("Embedding failed", self.pipeline.stats['errors'])
    
    def test_test_pipeline(self):
        """Test pipeline testing function."""
        # Mock successful pipeline
        self.mock_embedding_generator.model.encode.return_value = [np.array([0.1, 0.2, 0.3])]
        self.mock_vector_database.search_similar.return_value = []
        
        mock_response = Mock()
        mock_response.response = "Test answer"
        mock_response.model = "gemma3:270m"
        mock_response.metadata = {'generation_time': 1.0}
        self.mock_llm_client.generate.return_value = mock_response
        
        result = self.pipeline.test_pipeline()
        self.assertTrue(result)
        
        # Test with empty answer
        mock_response.response = ""
        result = self.pipeline.test_pipeline()
        self.assertFalse(result)
        
        # Test with failure
        self.mock_embedding_generator.model.encode.side_effect = Exception("Test failed")
        result = self.pipeline.test_pipeline()
        self.assertFalse(result)
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        # Add some test data
        self.pipeline.stats['queries_processed'] = 10
        self.pipeline.stats['successful_queries'] = 8
        self.pipeline.stats['total_processing_time'] = 20.0
        self.pipeline.stats['total_context_length'] = 8000
        self.pipeline.stats['total_sources_retrieved'] = 40
        
        stats = self.pipeline.get_statistics()
        
        self.assertEqual(stats['queries_processed'], 10)
        self.assertEqual(stats['successful_queries'], 8)
        self.assertEqual(stats['success_rate'], 0.8)
        self.assertEqual(stats['average_processing_time'], 2.5)
        self.assertEqual(stats['average_context_length'], 1000)
        self.assertEqual(stats['average_sources_per_query'], 5.0)
        self.assertIn('component_info', stats)
    
    def test_reset_statistics(self):
        """Test statistics reset."""
        # Add some data
        self.pipeline.stats['queries_processed'] = 10
        self.pipeline.stats['errors'] = ['error1', 'error2']
        
        self.pipeline.reset_statistics()
        
        self.assertEqual(self.pipeline.stats['queries_processed'], 0)
        self.assertEqual(self.pipeline.stats['errors'], [])
    
    def test_close(self):
        """Test pipeline closure."""
        self.pipeline.close()
        
        self.mock_vector_database.close.assert_called_once()
        self.mock_llm_client.close.assert_called_once()


class TestCLIArgs(unittest.TestCase):
    """Test CLIArgs class."""
    
    def test_cli_args_creation(self):
        """Test CLIArgs creation with default values."""
        args = CLIArgs(directory="/test/path")
        
        self.assertEqual(args.directory, "/test/path")
        self.assertFalse(args.verbose)
        self.assertEqual(args.chunk_size, 1000)
        self.assertEqual(args.top_k, 10)
        self.assertFalse(args.rebuild)
        self.assertFalse(args.show_sources)
        self.assertFalse(args.chat)
        self.assertEqual(args.max_context, 20000)
    
    def test_cli_args_custom_values(self):
        """Test CLIArgs with custom values."""
        args = CLIArgs(
            directory="/custom/path",
            verbose=True,
            chunk_size=500,
            top_k=5,
            rebuild=True,
            show_sources=True,
            chat=True,
            max_context=15000
        )
        
        self.assertEqual(args.directory, "/custom/path")
        self.assertTrue(args.verbose)
        self.assertEqual(args.chunk_size, 500)
        self.assertEqual(args.top_k, 5)
        self.assertTrue(args.rebuild)
        self.assertTrue(args.show_sources)
        self.assertTrue(args.chat)
        self.assertEqual(args.max_context, 15000)


class TestChatInterface(unittest.TestCase):
    """Test ChatInterface class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_rag_pipeline = Mock()
        self.mock_rag_pipeline.get_statistics.return_value = {
            'queries_processed': 5,
            'successful_queries': 4,
            'success_rate': 0.8,
            'average_processing_time': 2.5,
            'average_context_length': 1000,
            'average_sources_per_query': 3.0,
            'component_info': {
                'embedding_model': 'all-MiniLM-L6-v2',
                'llm_model': 'gemma3:270m',
                'vector_db_info': {'document_count': 100}
            }
        }
        
        self.args = CLIArgs(
            directory="/test",
            verbose=True,
            top_k=5,
            show_sources=True,
            chat=True
        )
        
        self.chat = ChatInterface(self.mock_rag_pipeline, self.args)
    
    def test_chat_interface_initialization(self):
        """Test chat interface initialization."""
        self.assertEqual(self.chat.rag_pipeline, self.mock_rag_pipeline)
        self.assertEqual(self.chat.args, self.args)
        self.assertEqual(self.chat.chat_history, [])
        self.assertTrue(self.chat.running)
    
    @patch('builtins.print')
    def test_display_welcome(self, mock_print):
        """Test welcome message display."""
        self.chat._display_welcome()
        
        # Check that print was called multiple times
        self.assertTrue(mock_print.called)
        
        # Check for key elements in the output
        all_output = ' '.join([str(call.args[0]) for call in mock_print.call_args_list])
        self.assertIn("DocuChat Interactive Chat", all_output)
        self.assertIn("/help", all_output)
        self.assertIn("Top-K results: 5", all_output)
        self.assertIn("Show sources: Yes", all_output)
    
    @patch('builtins.print')
    def test_display_help(self, mock_print):
        """Test help display."""
        self.chat._display_help()
        
        all_output = ' '.join([str(call.args[0]) for call in mock_print.call_args_list])
        self.assertIn("DocuChat Help", all_output)
        self.assertIn("main topics", all_output)
        self.assertIn("/quit", all_output)
    
    @patch('builtins.print')
    def test_display_stats(self, mock_print):
        """Test statistics display."""
        self.chat._display_stats()
        
        all_output = ' '.join([str(call.args[0]) for call in mock_print.call_args_list])
        self.assertIn("Pipeline Statistics", all_output)
        self.assertIn("Queries processed: 5", all_output)
        self.assertIn("Success rate: 80.0%", all_output)
        self.assertIn("Embedding model: all-MiniLM-L6-v2", all_output)
    
    @patch('builtins.print')
    def test_display_stats_error(self, mock_print):
        """Test statistics display with error."""
        self.mock_rag_pipeline.get_statistics.side_effect = Exception("Stats error")
        
        self.chat._display_stats()
        
        all_output = ' '.join([str(call.args[0]) for call in mock_print.call_args_list])
        self.assertIn("Error retrieving statistics", all_output)
    
    @patch('builtins.print')
    def test_display_history_empty(self, mock_print):
        """Test history display when empty."""
        self.chat._display_history()
        
        all_output = ' '.join([str(call.args[0]) for call in mock_print.call_args_list])
        self.assertIn("No chat history", all_output)
    
    @patch('builtins.print')
    def test_display_history_with_data(self, mock_print):
        """Test history display with data."""
        self.chat.chat_history = [
            {'timestamp': '10:30:00', 'question': 'What is AI?'},
            {'timestamp': '10:31:00', 'question': 'How does machine learning work?'}
        ]
        
        self.chat._display_history()
        
        all_output = ' '.join([str(call.args[0]) for call in mock_print.call_args_list])
        self.assertIn("Chat History (2 messages)", all_output)
        self.assertIn("What is AI?", all_output)
        self.assertIn("How does machine learning work?", all_output)
    
    @patch('builtins.print')
    def test_clear_history(self, mock_print):
        """Test history clearing."""
        self.chat.chat_history = [{'test': 'data'}]
        
        self.chat._clear_history()
        
        self.assertEqual(self.chat.chat_history, [])
        mock_print.assert_called_with("🗑️  Chat history cleared.")
    
    def test_format_sources(self):
        """Test source formatting."""
        sources = [
            {
                'distance': 0.2,
                'content': 'This is the content of the first document',
                'metadata': {'source_file': 'file1.txt'}
            },
            {
                'distance': 0.3,
                'content': 'This is the content of the second document',
                'metadata': {'source_file': 'file2.txt'}
            }
        ]
        
        result = self.chat._format_sources(sources)
        
        self.assertIn("📚 Sources:", result)
        self.assertIn("file1.txt (relevance: 80%)", result)
        self.assertIn("file2.txt (relevance: 70%)", result)
        
        # Test with verbose mode
        self.assertIn("This is the content of the first", result)
    
    def test_format_sources_disabled(self):
        """Test source formatting when disabled."""
        self.args.show_sources = False
        sources = [{'test': 'data'}]
        
        result = self.chat._format_sources(sources)
        self.assertEqual(result, "")
    
    @patch('builtins.print')
    @patch('time.time')
    def test_process_question(self, mock_time, mock_print):
        """Test question processing."""
        mock_time.side_effect = [100.0, 102.5]  # 2.5 second processing time
        
        # Mock RAG result
        mock_result = Mock()
        mock_result.answer = "AI stands for Artificial Intelligence"
        mock_result.sources = []
        mock_result.confidence_score = 0.85
        self.mock_rag_pipeline.answer_question.return_value = mock_result
        
        with patch('time.strftime', return_value='10:30:00'):
            self.chat._process_question("What is AI?")
        
        # Verify RAG pipeline was called correctly
        self.mock_rag_pipeline.answer_question.assert_called_once_with(
            question="What is AI?",
            top_k=5,
            include_sources=True
        )
        
        # Check print output
        all_output = ' '.join([str(call.args[0]) for call in mock_print.call_args_list])
        self.assertIn("AI stands for Artificial Intelligence", all_output)
        self.assertIn("Confidence: 0.85", all_output)
        
        # Check history was updated
        self.assertEqual(len(self.chat.chat_history), 1)
        self.assertEqual(self.chat.chat_history[0]['question'], "What is AI?")
        self.assertEqual(self.chat.chat_history[0]['answer'], "AI stands for Artificial Intelligence")
    
    @patch('builtins.print')
    def test_process_question_empty(self, mock_print):
        """Test processing empty question."""
        self.chat._process_question("")
        
        mock_print.assert_called_with("❓ Please enter a question.")
    
    @patch('builtins.print')
    def test_process_question_error(self, mock_print):
        """Test question processing error."""
        self.mock_rag_pipeline.answer_question.side_effect = Exception("Processing error")
        
        self.chat._process_question("What is AI?")
        
        all_output = ' '.join([str(call.args[0]) for call in mock_print.call_args_list])
        self.assertIn("Error processing question", all_output)
    
    def test_handle_command_quit(self):
        """Test quit commands."""
        self.assertFalse(self.chat._handle_command("/quit"))
        self.assertFalse(self.chat._handle_command("/exit"))
    
    @patch('builtins.print')
    def test_handle_command_help(self, mock_print):
        """Test help command."""
        result = self.chat._handle_command("/help")
        
        self.assertTrue(result)
        # Verify help was displayed
        all_output = ' '.join([str(call.args[0]) for call in mock_print.call_args_list])
        self.assertIn("DocuChat Help", all_output)
    
    @patch('builtins.print')
    def test_handle_command_unknown(self, mock_print):
        """Test unknown command."""
        result = self.chat._handle_command("/unknown")
        
        self.assertTrue(result)
        all_output = ' '.join([str(call.args[0]) for call in mock_print.call_args_list])
        self.assertIn("Unknown command", all_output)
    
    @patch('builtins.input')
    @patch('builtins.print')
    def test_start_chat_quit(self, mock_print, mock_input):
        """Test chat start and quit."""
        mock_input.return_value = "/quit"
        
        self.chat.start_chat()
        
        # Verify welcome was displayed
        all_output = ' '.join([str(call.args[0]) for call in mock_print.call_args_list])
        self.assertIn("DocuChat Interactive Chat", all_output)
        self.assertIn("Chat session ended", all_output)
    
    @patch('builtins.input')
    @patch('builtins.print')
    def test_start_chat_eof(self, mock_print, mock_input):
        """Test chat with EOF (Ctrl+D)."""
        mock_input.side_effect = EOFError()
        
        self.chat.start_chat()
        
        all_output = ' '.join([str(call.args[0]) for call in mock_print.call_args_list])
        self.assertIn("Goodbye!", all_output)
    
    @patch('builtins.input')
    @patch('builtins.print')
    def test_start_chat_keyboard_interrupt(self, mock_print, mock_input):
        """Test chat with keyboard interrupt."""
        mock_input.side_effect = [KeyboardInterrupt(), "/quit"]
        
        self.chat.start_chat()
        
        all_output = ' '.join([str(call.args[0]) for call in mock_print.call_args_list])
        self.assertIn("Chat interrupted", all_output)


class TestCLIParsing(unittest.TestCase):
    """Test CLI argument parsing."""
    
    def test_parse_args_basic(self):
        """Test basic argument parsing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            args = parse_args([temp_dir])
            
            self.assertEqual(args.directory, str(Path(temp_dir).resolve()))
            self.assertFalse(args.verbose)
            self.assertEqual(args.chunk_size, 1000)
            self.assertEqual(args.top_k, 10)
            self.assertFalse(args.rebuild)
            self.assertFalse(args.show_sources)
            self.assertFalse(args.chat)
            self.assertEqual(args.max_context, 20000)
    
    def test_parse_args_all_flags(self):
        """Test parsing with all flags."""
        with tempfile.TemporaryDirectory() as temp_dir:
            args = parse_args([
                temp_dir,
                "--verbose",
                "--chunk-size", "500",
                "--top-k", "3",
                "--rebuild",
                "--show-sources",
                "--chat",
                "--max-context", "15000"
            ])
            
            self.assertTrue(args.verbose)
            self.assertEqual(args.chunk_size, 500)
            self.assertEqual(args.top_k, 3)
            self.assertTrue(args.rebuild)
            self.assertTrue(args.show_sources)
            self.assertTrue(args.chat)
            self.assertEqual(args.max_context, 15000)
    
    def test_parse_args_invalid_directory(self):
        """Test parsing with invalid directory."""
        with self.assertRaises(SystemExit):
            parse_args(["/nonexistent/directory"])
    
    def test_parse_args_invalid_chunk_size(self):
        """Test parsing with invalid chunk size."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(SystemExit):
                parse_args([temp_dir, "--chunk-size", "0"])
    
    def test_parse_args_invalid_top_k(self):
        """Test parsing with invalid top-k."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(SystemExit):
                parse_args([temp_dir, "--top-k", "-1"])
    
    def test_parse_args_invalid_max_context(self):
        """Test parsing with invalid max context."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(SystemExit):
                parse_args([temp_dir, "--max-context", "0"])


class TestIntegration(unittest.TestCase):
    """Integration tests for Phase 4 components."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
    
    @patch('cli.ChatInterface')
    def test_start_interactive_chat(self, mock_chat_class):
        """Test starting interactive chat."""
        mock_pipeline = Mock()
        args = CLIArgs(directory=self.temp_dir, chat=True)
        
        mock_chat_instance = Mock()
        mock_chat_class.return_value = mock_chat_instance
        
        start_interactive_chat(mock_pipeline, args)
        
        mock_chat_class.assert_called_once_with(mock_pipeline, args)
        mock_chat_instance.start_chat.assert_called_once()
    
    @patch('cli.ChatInterface')
    def test_start_interactive_chat_error(self, mock_chat_class):
        """Test interactive chat with error."""
        mock_pipeline = Mock()
        args = CLIArgs(directory=self.temp_dir, chat=True)
        
        mock_chat_class.side_effect = Exception("Chat error")
        
        # Should not raise exception - should handle gracefully
        with patch('builtins.print') as mock_print:
            start_interactive_chat(mock_pipeline, args)
            
            # Check error message was printed
            all_output = ' '.join([str(call.args[0]) for call in mock_print.call_args_list])
            self.assertIn("Chat interface failed", all_output)


class TestErrorHandling(unittest.TestCase):
    """Test error handling scenarios."""
    
    def test_rag_pipeline_component_failures(self):
        """Test RAG pipeline with component failures."""
        # Test with database error
        mock_embedding_generator = Mock()
        mock_embedding_generator.model_name = "test"
        
        mock_vector_database = Mock()
        mock_vector_database.get_info.return_value = {'error': 'Database error'}
        
        mock_llm_client = Mock()
        mock_llm_client.is_available.return_value = True
        
        with self.assertRaises(RuntimeError):
            RAGPipeline(
                embedding_generator=mock_embedding_generator,
                vector_database=mock_vector_database,
                llm_client=mock_llm_client
            )
    
    def test_chat_interface_signal_handling(self):
        """Test chat interface signal handling."""
        mock_pipeline = Mock()
        args = CLIArgs(directory="/test")
        
        chat = ChatInterface(mock_pipeline, args)
        
        # Test signal handler
        chat._signal_handler(2, None)  # SIGINT
        self.assertFalse(chat.running)


if __name__ == '__main__':
    # Configure test logging
    import logging
    logging.basicConfig(level=logging.CRITICAL)  # Suppress logs during testing
    
    # Check if imports are available
    if not IMPORTS_AVAILABLE:
        print("❌ Cannot run tests - required modules not available")
        print("   Install dependencies: pip install numpy")
        sys.exit(1)
    
    # Run tests
    unittest.main(verbosity=2)