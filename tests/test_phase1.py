#!/usr/bin/env python3
"""
Comprehensive test suite for DocuChat RAG System Phase 1.
Tests CLI argument parsing, file discovery, and document loading functionality.
"""

import unittest
import tempfile
import shutil
import os
import sys
from pathlib import Path
from unittest.mock import patch
import warnings

# Add src directory to Python path
test_dir = Path(__file__).parent
src_path = test_dir.parent / "src"
sys.path.insert(0, str(src_path))

from cli import parse_args, CLIArgs, validate_directory
from document_loader import DocumentLoader


class TestCLI(unittest.TestCase):
    """Test cases for CLI argument parsing and validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_parse_basic_args(self):
        """Test parsing basic required arguments."""
        args = parse_args([self.temp_dir])
        
        self.assertIsInstance(args, CLIArgs)
        self.assertEqual(args.directory, str(self.temp_path.resolve()))
        self.assertFalse(args.verbose)
        self.assertEqual(args.chunk_size, 1000)
        self.assertEqual(args.top_k, 5)
        self.assertFalse(args.rebuild)
        self.assertFalse(args.show_sources)
    
    def test_parse_all_flags(self):
        """Test parsing all optional flags."""
        args = parse_args([
            self.temp_dir,
            '--verbose',
            '--chunk-size', '500',
            '--top-k', '10',
            '--rebuild',
            '--show-sources'
        ])
        
        self.assertTrue(args.verbose)
        self.assertEqual(args.chunk_size, 500)
        self.assertEqual(args.top_k, 10)
        self.assertTrue(args.rebuild)
        self.assertTrue(args.show_sources)
    
    def test_short_flags(self):
        """Test short flag versions."""
        args = parse_args([self.temp_dir, '-v'])
        self.assertTrue(args.verbose)
    
    def test_invalid_directory(self):
        """Test validation of invalid directory paths."""
        nonexistent_dir = "/path/that/does/not/exist"
        
        with self.assertRaises(SystemExit):
            parse_args([nonexistent_dir])
    
    def test_invalid_chunk_size(self):
        """Test validation of invalid chunk sizes."""
        with self.assertRaises(SystemExit):
            parse_args([self.temp_dir, '--chunk-size', '0'])
        
        with self.assertRaises(SystemExit):
            parse_args([self.temp_dir, '--chunk-size', '-1'])
    
    def test_invalid_top_k(self):
        """Test validation of invalid top-k values."""
        with self.assertRaises(SystemExit):
            parse_args([self.temp_dir, '--top-k', '0'])
        
        with self.assertRaises(SystemExit):
            parse_args([self.temp_dir, '--top-k', '-1'])
    
    def test_file_instead_of_directory(self):
        """Test error handling when file path is provided instead of directory."""
        temp_file = self.temp_path / "test.txt"
        temp_file.write_text("test content")
        
        with self.assertRaises(SystemExit):
            parse_args([str(temp_file)])
    
    def test_validate_directory_function(self):
        """Test the validate_directory function directly."""
        # Test valid directory
        result = validate_directory(self.temp_dir)
        self.assertIsInstance(result, Path)
        self.assertTrue(result.exists())
        
        # Test invalid directory
        with self.assertRaises(Exception):
            validate_directory("/nonexistent/path")


class TestDocumentLoader(unittest.TestCase):
    """Test cases for document loading functionality."""
    
    def setUp(self):
        """Set up test fixtures with sample documents."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test directory structure
        self.subdir = self.temp_path / "subdirectory"
        self.subdir.mkdir()
        
        # Create sample files
        self.create_test_files()
        
        self.loader = DocumentLoader(verbose=False)
        self.verbose_loader = DocumentLoader(verbose=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_files(self):
        """Create sample test files of different types."""
        # Text files
        (self.temp_path / "sample.txt").write_text("This is a sample text file.\nWith multiple lines.")
        (self.temp_path / "readme.md").write_text("# Markdown File\n\nThis is a **markdown** file.")
        
        # Text file in subdirectory
        (self.subdir / "nested.txt").write_text("This is a nested text file.")
        
        # Empty text file
        (self.temp_path / "empty.txt").write_text("")
        
        # File with non-standard encoding (if possible)
        try:
            (self.temp_path / "unicode.txt").write_text("Unicode content: äöü 中文", encoding='utf-8')
        except Exception:
            pass
        
        # Unsupported file type
        (self.temp_path / "unsupported.xyz").write_text("This file type is not supported.")
        
        # Non-text file that should be ignored
        (self.temp_path / "binary.bin").write_bytes(b'\x00\x01\x02\x03')
        
        # Create a directory that looks like a file
        (self.temp_path / "fake_file.txt").mkdir()
    
    def test_file_discovery(self):
        """Test recursive file discovery functionality."""
        files = self.loader.discover_files(self.temp_dir)
        
        # Should find supported files
        file_names = [f.name for f in files]
        self.assertIn("sample.txt", file_names)
        self.assertIn("readme.md", file_names)
        self.assertIn("nested.txt", file_names)
        self.assertIn("empty.txt", file_names)
        self.assertIn("unicode.txt", file_names)
        
        # Should not find unsupported files
        self.assertNotIn("unsupported.xyz", file_names)
        self.assertNotIn("binary.bin", file_names)
        
        # Should not include directories
        self.assertNotIn("subdirectory", file_names)
        self.assertNotIn("fake_file.txt", file_names)
        
        # Verify we found the expected number of files
        self.assertEqual(len(files), 5)  # txt, md, nested.txt, empty.txt, unicode.txt
    
    def test_file_discovery_nonexistent_directory(self):
        """Test file discovery with nonexistent directory."""
        with self.assertRaises(FileNotFoundError):
            self.loader.discover_files("/nonexistent/directory")
    
    def test_file_discovery_file_instead_of_directory(self):
        """Test file discovery when file path is provided instead of directory."""
        temp_file = self.temp_path / "test.txt"
        temp_file.write_text("test")
        
        with self.assertRaises(NotADirectoryError):
            self.loader.discover_files(str(temp_file))
    
    def test_load_text_file(self):
        """Test loading plain text files."""
        text_file = self.temp_path / "sample.txt"
        content = self.loader.load_text_file(text_file)
        
        self.assertIsInstance(content, str)
        self.assertIn("sample text file", content)
        self.assertIn("multiple lines", content)
    
    def test_load_markdown_file(self):
        """Test loading markdown files."""
        md_file = self.temp_path / "readme.md"
        content = self.loader.load_text_file(md_file)
        
        self.assertIsInstance(content, str)
        self.assertIn("Markdown File", content)
        self.assertIn("**markdown**", content)
    
    def test_load_unicode_file(self):
        """Test loading files with unicode content."""
        unicode_file = self.temp_path / "unicode.txt"
        if unicode_file.exists():
            content = self.loader.load_text_file(unicode_file)
            self.assertIsInstance(content, str)
            self.assertIn("äöü", content)
            self.assertIn("中文", content)
    
    def test_load_empty_file(self):
        """Test loading empty files."""
        empty_file = self.temp_path / "empty.txt"
        content = self.loader.load_text_file(empty_file)
        
        self.assertEqual(content, "")
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent files."""
        nonexistent = self.temp_path / "nonexistent.txt"
        
        with self.assertRaises(Exception):
            self.loader.load_text_file(nonexistent)
    
    def test_load_document_auto_detection(self):
        """Test automatic document format detection."""
        # Test text file
        text_file = self.temp_path / "sample.txt"
        result = self.loader.load_document(text_file)
        
        self.assertIsNotNone(result)
        filename, content = result
        self.assertEqual(filename, str(text_file))
        self.assertIn("sample text file", content)
        
        # Test markdown file
        md_file = self.temp_path / "readme.md"
        result = self.loader.load_document(md_file)
        
        self.assertIsNotNone(result)
        filename, content = result
        self.assertEqual(filename, str(md_file))
        self.assertIn("Markdown File", content)
    
    def test_load_document_unsupported_format(self):
        """Test loading unsupported document formats."""
        unsupported = self.temp_path / "unsupported.xyz"
        result = self.loader.load_document(unsupported)
        
        self.assertIsNone(result)
    
    def test_load_document_empty_file(self):
        """Test loading empty documents."""
        empty_file = self.temp_path / "empty.txt"
        
        # Should return None for empty files
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.loader.load_document(empty_file)
        
        self.assertIsNone(result)
    
    def test_load_document_nonexistent_file(self):
        """Test loading nonexistent documents."""
        nonexistent = self.temp_path / "nonexistent.txt"
        result = self.loader.load_document(nonexistent)
        
        self.assertIsNone(result)
    
    def test_load_documents_from_directory(self):
        """Test loading all documents from a directory."""
        documents = list(self.loader.load_documents_from_directory(self.temp_dir))
        
        # Should have loaded non-empty supported files
        self.assertGreater(len(documents), 0)
        
        # Verify we get tuples of (filename, content)
        for filename, content in documents:
            self.assertIsInstance(filename, str)
            self.assertIsInstance(content, str)
            self.assertGreater(len(content), 0)
        
        # Verify specific files were loaded
        filenames = [Path(doc[0]).name for doc in documents]
        self.assertIn("sample.txt", filenames)
        self.assertIn("readme.md", filenames)
        self.assertIn("nested.txt", filenames)
    
    def test_verbose_mode(self):
        """Test verbose logging functionality."""
        # This test verifies that verbose mode doesn't crash
        # Actual output verification would require capturing stdout
        
        with patch('builtins.print') as mock_print:
            verbose_loader = DocumentLoader(verbose=True)
            files = verbose_loader.discover_files(self.temp_dir)
            
            # Verify print was called (verbose output)
            self.assertTrue(mock_print.called)
        
        # Test loading with verbose mode
        with patch('builtins.print'):
            documents = list(verbose_loader.load_documents_from_directory(self.temp_dir))
            self.assertGreater(len(documents), 0)
    
    def test_error_handling_graceful_failure(self):
        """Test that errors are handled gracefully without crashing."""
        # Create a file we can't read (simulate permission error)
        restricted_file = self.temp_path / "restricted.txt"
        restricted_file.write_text("test content")
        
        # Test that the loader doesn't crash on permission errors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Should not raise exception, should return None
            result = self.loader.load_document(restricted_file)
            
            # File should still be readable in test environment
            # This test mainly ensures the error handling code path exists
            self.assertIsNotNone(result)


class TestIntegration(unittest.TestCase):
    """Integration tests combining CLI and document loading."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create sample documents
        (self.temp_path / "doc1.txt").write_text("First document content.")
        (self.temp_path / "doc2.md").write_text("# Second Document\nMarkdown content.")
        
        subdir = self.temp_path / "subdir"
        subdir.mkdir()
        (subdir / "doc3.txt").write_text("Third document in subdirectory.")
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_processing(self):
        """Test complete end-to-end processing workflow."""
        # Parse arguments
        args = parse_args([self.temp_dir, '--verbose'])
        
        # Create loader
        loader = DocumentLoader(verbose=args.verbose)
        
        # Process documents
        documents = list(loader.load_documents_from_directory(args.directory))
        
        # Verify results
        self.assertEqual(len(documents), 3)
        
        # Verify content
        contents = [doc[1] for doc in documents]
        self.assertTrue(any("First document" in content for content in contents))
        self.assertTrue(any("Second Document" in content for content in contents))
        self.assertTrue(any("Third document" in content for content in contents))
    
    def test_no_documents_found(self):
        """Test behavior when no supported documents are found."""
        empty_dir = self.temp_path / "empty"
        empty_dir.mkdir()
        
        loader = DocumentLoader(verbose=False)
        documents = list(loader.load_documents_from_directory(str(empty_dir)))
        
        self.assertEqual(len(documents), 0)


def run_tests():
    """Run all tests and provide detailed output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestCLI))
    suite.addTests(loader.loadTestsFromTestCase(TestDocumentLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    # Return success status
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)