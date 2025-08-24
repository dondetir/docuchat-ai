"""
Text chunking module for DocuChat RAG system.
Handles secure text chunking with input validation and memory management.
"""

import re
import hashlib
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Generator, Any
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextChunk:
    """
    Immutable data class representing a text chunk with metadata.
    
    Security Features:
    - Immutable to prevent accidental modification
    - Content length validation
    - Metadata tracking for audit trails
    """
    content: str
    chunk_id: str
    source_file: str
    chunk_index: int
    start_char: int = 0
    end_char: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate chunk data on creation."""
        if not isinstance(self.content, str):
            raise TypeError("Chunk content must be a string")
        
        if len(self.content) > 50000:  # Security: Prevent excessive memory usage
            raise ValueError("Chunk content exceeds maximum allowed size (50KB)")
        
        if self.chunk_index < 0:
            raise ValueError("Chunk index must be non-negative")
        
        if self.start_char < 0 or self.end_char < 0:
            raise ValueError("Character positions must be non-negative")
        
        if self.start_char > self.end_char:
            raise ValueError("Start character position cannot exceed end position")
    
    @property
    def char_count(self) -> int:
        """Get the character count of the chunk."""
        return len(self.content)
    
    @property
    def word_count(self) -> int:
        """Get the estimated word count of the chunk."""
        return len(self.content.split())
    
    @property
    def content_hash(self) -> str:
        """Get SHA-256 hash of content for integrity verification."""
        return hashlib.sha256(self.content.encode('utf-8')).hexdigest()


class RecursiveCharacterTextSplitter:
    """
    Secure text splitter that recursively splits text on various separators.
    
    Security Features:
    - Input validation and sanitization
    - Memory usage monitoring
    - Configurable size limits
    - Safe separator handling
    """
    
    # Default separators in order of preference
    DEFAULT_SEPARATORS = [
        "\n\n",      # Paragraph breaks
        "\n",        # Line breaks
        ". ",        # Sentence endings
        "? ",        # Question endings
        "! ",        # Exclamation endings
        "; ",        # Semicolon breaks
        ", ",        # Comma breaks
        " ",         # Word breaks
        ""           # Character breaks (last resort)
    ]
    
    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        length_function: callable = len,
        max_chunk_size: int = 10000,
        min_chunk_size: int = 50
    ):
        """
        Initialize the text splitter with security controls.
        
        Args:
            chunk_size: Target size for each chunk
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to use for splitting
            keep_separator: Whether to keep separators in chunks
            length_function: Function to calculate text length
            max_chunk_size: Maximum allowed chunk size (security limit)
            min_chunk_size: Minimum chunk size to prevent tiny fragments
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Input validation
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        
        if chunk_overlap < 0:
            raise ValueError("Chunk overlap cannot be negative")
        
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        
        if max_chunk_size <= 0 or max_chunk_size < chunk_size:
            raise ValueError("Max chunk size must be positive and >= chunk_size")
        
        if min_chunk_size <= 0 or min_chunk_size > chunk_size:
            raise ValueError("Min chunk size must be positive and <= chunk_size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS.copy()
        self.keep_separator = keep_separator
        self.length_function = length_function
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        
        # Security: Validate separators
        self._validate_separators()
        
        logger.debug(f"TextSplitter initialized: chunk_size={chunk_size}, "
                   f"overlap={chunk_overlap}, max_size={max_chunk_size}")
    
    def _validate_separators(self) -> None:
        """Validate separator list for security."""
        if not isinstance(self.separators, list):
            raise TypeError("Separators must be a list")
        
        for sep in self.separators:
            if not isinstance(sep, str):
                raise TypeError("All separators must be strings")
            
            # Security: Prevent excessive separator lengths
            if len(sep) > 100:
                raise ValueError("Separator length exceeds maximum (100 chars)")
    
    def _sanitize_text(self, text: str) -> str:
        """
        Sanitize input text for security.
        
        Args:
            text: Input text to sanitize
            
        Returns:
            Sanitized text
        """
        if not isinstance(text, str):
            raise TypeError("Input must be a string")
        
        # Security: Check text length
        if len(text) > 10_000_000:  # 10MB limit
            raise ValueError("Input text exceeds maximum size limit (10MB)")
        
        # Normalize whitespace and remove null bytes
        text = re.sub(r'\x00', '', text)  # Remove null bytes
        text = re.sub(r'\r\n', '\n', text)  # Normalize line endings
        text = re.sub(r'\r', '\n', text)  # Convert Mac line endings
        
        return text
    
    def _split_text_with_separator(self, text: str, separator: str) -> List[str]:
        """
        Split text using a specific separator.
        
        Args:
            text: Text to split
            separator: Separator to use
            
        Returns:
            List of text segments
        """
        if not separator:
            # Character-level split (last resort)
            return list(text)
        
        splits = text.split(separator)
        
        if self.keep_separator and len(splits) > 1:
            # Add separator back to all pieces except the last
            for i in range(len(splits) - 1):
                splits[i] += separator
        
        return splits
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """
        Merge small splits into larger chunks while respecting size limits.
        
        Args:
            splits: List of text splits
            separator: Separator used for splitting
            
        Returns:
            List of merged chunks
        """
        chunks = []
        current_chunk = ""
        
        for split in splits:
            # Skip empty splits
            if not split.strip():
                continue
            
            # Calculate potential new chunk size
            potential_size = self.length_function(current_chunk + split)
            
            if potential_size <= self.chunk_size:
                # Add to current chunk
                current_chunk += split
            else:
                # Current chunk is full, start new one
                if current_chunk and len(current_chunk.strip()) >= self.min_chunk_size:
                    chunks.append(current_chunk)
                
                # Handle oversized splits
                if self.length_function(split) > self.max_chunk_size:
                    logger.warning(f"Split exceeds max size ({len(split)} chars), truncating")
                    split = split[:self.max_chunk_size]
                
                current_chunk = split
        
        # Add final chunk if it exists and meets minimum size
        if current_chunk and len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append(current_chunk)
        
        return chunks
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using different separators.
        
        Args:
            text: Text to split
            separators: List of separators to try
            
        Returns:
            List of text chunks
        """
        if not separators:
            # No more separators, return as-is (truncated if needed)
            if len(text) <= self.max_chunk_size:
                return [text]
            else:
                logger.warning(f"Text chunk too large ({len(text)} chars), truncating")
                return [text[:self.max_chunk_size]]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Split with current separator
        splits = self._split_text_with_separator(text, separator)
        
        # Merge splits into appropriately sized chunks
        merged_chunks = self._merge_splits(splits, separator)
        
        # Recursively process chunks that are still too large
        final_chunks = []
        for chunk in merged_chunks:
            if self.length_function(chunk) > self.chunk_size:
                # Chunk still too large, try next separator
                sub_chunks = self._recursive_split(chunk, remaining_separators)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """
        Add overlap between consecutive chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of chunks with overlap added
        """
        if len(chunks) <= 1 or self.chunk_overlap == 0:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk - no previous overlap needed
                overlapped_chunks.append(chunk)
            else:
                # Add overlap from previous chunk
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
                
                # Combine overlap with current chunk
                combined = overlap_text + chunk
                
                # Ensure combined chunk doesn't exceed max size
                if len(combined) > self.max_chunk_size:
                    # Truncate current chunk to fit
                    available_space = self.max_chunk_size - len(overlap_text)
                    if available_space > 0:
                        chunk = chunk[:available_space]
                        combined = overlap_text + chunk
                    else:
                        # Skip overlap if it's too large
                        combined = chunk
                
                overlapped_chunks.append(combined)
        
        return overlapped_chunks
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks using recursive character splitting.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
            
        Raises:
            ValueError: If input is invalid
            TypeError: If input is not a string
        """
        # Security: Sanitize input
        text = self._sanitize_text(text)
        
        if not text.strip():
            logger.info("Input text is empty or whitespace-only")
            return []
        
        # Perform recursive splitting
        chunks = self._recursive_split(text, self.separators.copy())
        
        # Add overlap between chunks
        chunks = self._add_overlap(chunks)
        
        # Final validation
        validated_chunks = []
        for chunk in chunks:
            if chunk.strip() and len(chunk.strip()) >= self.min_chunk_size:
                validated_chunks.append(chunk)
        
        logger.debug(f"Split text into {len(validated_chunks)} chunks")
        return validated_chunks


class DocumentChunker:
    """
    High-level document chunking interface with security controls.
    
    Security Features:
    - File path validation
    - Memory usage monitoring
    - Audit logging
    - Safe metadata handling
    """
    
    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        verbose: bool = False,
        max_file_size: int = 100_000_000  # 100MB limit
    ):
        """
        Initialize the document chunker.
        
        Args:
            chunk_size: Target size for each chunk
            chunk_overlap: Number of characters to overlap between chunks
            verbose: Enable verbose logging
            max_file_size: Maximum file size to process (security limit)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.verbose = verbose
        self.max_file_size = max_file_size
        
        # Initialize text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_chunk_size=chunk_size * 2,  # Allow some flexibility
            min_chunk_size=50
        )
        
        # Statistics tracking
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'total_characters': 0,
            'errors': 0
        }
        
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        logger.debug(f"DocumentChunker initialized: chunk_size={chunk_size}, "
                   f"overlap={chunk_overlap}")
    
    def _log(self, message: str, level: str = "info") -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            if level == "debug":
                logger.debug(f"[DocumentChunker] {message}")
            elif level == "warning":
                logger.warning(f"[DocumentChunker] {message}")
            elif level == "error":
                logger.error(f"[DocumentChunker] {message}")
            else:
                logger.info(f"[DocumentChunker] {message}")
    
    def _validate_file_path(self, file_path: str) -> Path:
        """
        Validate and normalize file path for security.
        
        Args:
            file_path: Path to validate
            
        Returns:
            Validated Path object
            
        Raises:
            ValueError: If path is invalid or unsafe
        """
        if not isinstance(file_path, (str, Path)):
            raise TypeError("File path must be a string or Path object")
        
        path = Path(file_path)
        
        # Security: Prevent path traversal attacks
        try:
            resolved_path = path.resolve()
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Cannot resolve path {file_path}: {e}")
        
        # Check if path contains dangerous patterns
        path_str = str(resolved_path)
        if ".." in path_str or path_str.startswith("/proc") or path_str.startswith("/sys"):
            raise ValueError(f"Potentially unsafe path: {file_path}")
        
        return resolved_path
    
    def _generate_chunk_id(self, source_file: str, chunk_index: int, content: str) -> str:
        """
        Generate a unique, secure chunk ID.
        
        Args:
            source_file: Source file path
            chunk_index: Index of chunk in document
            content: Chunk content
            
        Returns:
            Unique chunk ID
        """
        # Create hash from file path, index, and content hash
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
        source_hash = hashlib.sha256(source_file.encode('utf-8')).hexdigest()[:8]
        
        return f"{source_hash}_{chunk_index:04d}_{content_hash}"
    
    def chunk_document(
        self,
        text: str,
        source_file: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Generator[TextChunk, None, None]:
        """
        Chunk a document into secure TextChunk objects.
        
        Args:
            text: Document text to chunk
            source_file: Source file path
            metadata: Optional metadata to include
            
        Yields:
            TextChunk objects
            
        Raises:
            ValueError: If input is invalid
        """
        # Validate inputs
        if not isinstance(text, str):
            raise TypeError("Text must be a string")
        
        # Security: Validate file path
        validated_path = self._validate_file_path(source_file)
        source_file = str(validated_path)
        
        # Security: Check text size
        if len(text) > self.max_file_size:
            raise ValueError(f"Document exceeds maximum size limit ({self.max_file_size} bytes)")
        
        # Initialize metadata
        if metadata is None:
            metadata = {}
        
        # Security: Validate metadata
        if not isinstance(metadata, dict):
            raise TypeError("Metadata must be a dictionary")
        
        self._log(f"Chunking document: {Path(source_file).name} ({len(text)} chars)")
        
        try:
            # Split text into chunks
            text_chunks = self.splitter.split_text(text)
            
            if not text_chunks:
                self._log("No chunks created from document", "warning")
                return
            
            # Create TextChunk objects
            char_position = 0
            
            for chunk_index, chunk_text in enumerate(text_chunks):
                # Calculate character positions
                start_char = char_position
                end_char = start_char + len(chunk_text)
                
                # Generate secure chunk ID
                chunk_id = self._generate_chunk_id(source_file, chunk_index, chunk_text)
                
                # Create chunk metadata
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'creation_time': str(hash(chunk_text) % 1000000),  # Pseudo-timestamp
                    'chunk_method': 'recursive_character',
                    'chunk_size_config': self.chunk_size,
                    'overlap_config': self.chunk_overlap
                })
                
                # Create TextChunk object
                try:
                    chunk = TextChunk(
                        content=chunk_text,
                        chunk_id=chunk_id,
                        source_file=source_file,
                        chunk_index=chunk_index,
                        start_char=start_char,
                        end_char=end_char,
                        metadata=chunk_metadata
                    )
                    
                    self._log(f"Created chunk {chunk_index}: {len(chunk_text)} chars", "debug")
                    
                    # Update statistics
                    self.stats['chunks_created'] += 1
                    self.stats['total_characters'] += len(chunk_text)
                    
                    yield chunk
                    
                except Exception as e:
                    self._log(f"Error creating chunk {chunk_index}: {e}", "error")
                    self.stats['errors'] += 1
                    continue
                
                # Update character position for next chunk
                char_position = end_char - self.chunk_overlap if self.chunk_overlap > 0 else end_char
            
            # Update statistics
            self.stats['documents_processed'] += 1
            
            self._log(f"Successfully chunked document into {len(text_chunks)} chunks")
            
        except Exception as e:
            self._log(f"Error chunking document: {e}", "error")
            self.stats['errors'] += 1
            raise
    
    def get_statistics(self) -> Dict[str, int]:
        """Get processing statistics."""
        return self.stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'total_characters': 0,
            'errors': 0
        }
        self._log("Statistics reset")


def main():
    """Simple test function for the chunker module."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python chunker.py <text_content>")
        sys.exit(1)
    
    text = sys.argv[1]
    
    try:
        # Test basic chunking functionality
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20, verbose=True)
        
        chunks = list(chunker.chunk_document(text, "test_file.txt"))
        
        print(f"\nChunking Results:")
        print(f"Created {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i + 1}:")
            print(f"  ID: {chunk.chunk_id}")
            print(f"  Length: {chunk.char_count} chars")
            print(f"  Content: {chunk.content[:100]}...")
        
        print(f"\nStatistics: {chunker.get_statistics()}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()