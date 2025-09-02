"""
Content-aware chunking enhancement for DocuChat RAG system.
Extends the existing DocumentChunker with intelligent content type detection and boundary analysis.
"""

import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Generator, Any, Tuple
from pathlib import Path
import logging

# Import existing chunker components
from chunker import DocumentChunker, RecursiveCharacterTextSplitter, TextChunk

# Configure logging
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContentAwareChunk(TextChunk):
    """
    Enhanced chunk with content-awareness metadata.
    Extends TextChunk with semantic and content-type information.
    """
    content_type: str = "unknown"
    content_confidence: float = 0.0
    boundary_quality: float = 0.0
    semantic_keywords: List[str] = field(default_factory=list)
    topic_shift_score: float = 0.0


class ContentTypeDetector:
    """
    Fast, pattern-based content type detection.
    Uses heuristics to classify content as narrative, technical, or other.
    """
    
    # Content type patterns with confidence weights
    CONTENT_PATTERNS = {
        'narrative': [
            (r'"[^"]*"[^"]*said|replied|asked', 0.8),  # Dialogue patterns
            (r'\b(he|she|they)\s+(said|walked|looked|felt|thought)', 0.7),  # Narrative actions
            (r'\b(once upon|long ago|in the|there was)', 0.9),  # Story beginnings
            (r'[.!?]\s+[A-Z][a-z]+\s+(was|were|had|could|would)', 0.6),  # Story flow
        ],
        'technical': [
            (r'\b(function|class|method|variable|parameter|return)\b', 0.8),  # Programming terms
            (r'\b(API|HTTP|JSON|XML|SQL|database)\b', 0.7),  # Technical acronyms
            (r'```|\bcode\b|<[^>]+>', 0.9),  # Code blocks or HTML
            (r'\b(configure|install|setup|documentation)\b', 0.6),  # Technical actions
        ],
        'conversational': [
            (r'\b(hello|hi|thanks|please|sorry)\b', 0.7),  # Polite expressions
            (r'\b(I think|in my opinion|personally)\b', 0.8),  # Personal opinions
            (r'[?!]{2,}|[.]{3,}', 0.6),  # Emotional punctuation
        ]
    }
    
    def __init__(self):
        self.confidence_threshold = 0.3
    
    def detect_content_type(self, text: str) -> Tuple[str, float]:
        """
        Detect content type using pattern matching.
        
        Args:
            text: Text sample to analyze (first 2000 chars is sufficient)
            
        Returns:
            Tuple of (content_type, confidence_score)
        """
        if not text or len(text.strip()) < 50:
            return "unknown", 0.0
        
        # Use first 2000 chars for speed
        sample = text[:2000].lower()
        
        type_scores = {}
        
        for content_type, patterns in self.CONTENT_PATTERNS.items():
            score = 0.0
            pattern_matches = 0
            
            for pattern, weight in patterns:
                matches = len(re.findall(pattern, sample, re.IGNORECASE))
                if matches > 0:
                    # Normalize by text length and add weight
                    normalized_score = min(matches / 10.0, 1.0) * weight
                    score += normalized_score
                    pattern_matches += 1
            
            if pattern_matches > 0:
                # Average score across patterns that matched
                type_scores[content_type] = score / len(patterns)
        
        if not type_scores:
            return "unknown", 0.0
        
        # Get best match
        best_type = max(type_scores.keys(), key=lambda k: type_scores[k])
        confidence = type_scores[best_type]
        
        # Only return confident results
        if confidence >= self.confidence_threshold:
            return best_type, confidence
        else:
            return "unknown", confidence


class SmartBoundaryAnalyzer:
    """
    Enhanced boundary analysis for content-aware chunking.
    Improves upon existing sentence detection with semantic awareness.
    """
    
    def __init__(self):
        # Enhanced separators for different content types
        self.narrative_separators = [
            "\n\n",           # Paragraph breaks (highest priority for narrative)
            '".\n',           # End of dialogue
            '. "',            # Dialogue transitions  
            "\n",             # Line breaks
            ". ",             # Sentence endings
            "! ",             # Exclamations
            "? ",             # Questions
            "; ",             # Strong pauses
            ", ",             # Comma breaks
            " ",              # Word breaks
            ""                # Character breaks
        ]
        
        self.technical_separators = [
            "\n\n",           # Section breaks
            "```\n",          # Code block boundaries
            "\n#",            # Markdown headings
            "\n-",            # List items
            "\n*",            # Bullet points
            "\n",             # Line breaks
            ". ",             # Sentence endings
            "; ",             # Code statement ends
            "{",              # Code block starts
            "}",              # Code block ends
            " ",              # Word breaks
            ""                # Character breaks
        ]
        
        self.default_separators = [
            "\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""
        ]
    
    def get_optimal_separators(self, content_type: str) -> List[str]:
        """Get optimal separator list for content type."""
        if content_type == "narrative":
            return self.narrative_separators
        elif content_type == "technical":
            return self.technical_separators
        else:
            return self.default_separators
    
    def calculate_boundary_quality(self, chunk_text: str, separator_used: str) -> float:
        """
        Calculate quality score for chunk boundary.
        Higher scores indicate cleaner semantic boundaries.
        """
        if not chunk_text or not chunk_text.strip():
            return 0.0
        
        # Base quality by separator type
        separator_quality = {
            "\n\n": 1.0,      # Perfect: paragraph boundary
            '".\n': 0.95,     # Excellent: dialogue end
            '. "': 0.95,      # Excellent: dialogue transition
            ". ": 0.9,        # Great: sentence boundary
            "! ": 0.9,        # Great: exclamation end
            "? ": 0.9,        # Great: question end
            "\n": 0.7,        # Good: line break
            "; ": 0.6,        # OK: clause break
            ", ": 0.4,        # Poor: comma break
            " ": 0.2,         # Bad: word break
            "": 0.0           # Terrible: character break
        }
        
        base_quality = separator_quality.get(separator_used, 0.1)
        
        # Bonus for ending at complete sentences
        if chunk_text.strip().endswith(('.', '!', '?', '"')):
            base_quality += 0.1
        
        # Penalty for incomplete sentences at start
        if re.match(r'^[a-z]', chunk_text.strip()):
            base_quality -= 0.2
        
        return max(0.0, min(1.0, base_quality))


class ContentAwareRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    """
    Enhanced text splitter with content-type awareness.
    Extends the existing RecursiveCharacterTextSplitter with intelligent separator selection.
    """
    
    def __init__(self, content_type: str = "unknown", **kwargs):
        self.content_type = content_type
        self.boundary_analyzer = SmartBoundaryAnalyzer()
        
        # Get content-appropriate separators
        optimal_separators = self.boundary_analyzer.get_optimal_separators(content_type)
        
        # Override separators if not explicitly provided
        if 'separators' not in kwargs:
            kwargs['separators'] = optimal_separators
        
        super().__init__(**kwargs)
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Enhanced merge with boundary quality tracking."""
        # Use parent implementation but track boundary quality
        chunks = super()._merge_splits(splits, separator)
        
        # Store boundary quality for each chunk (for debugging/optimization)
        if hasattr(self, '_chunk_qualities'):
            for chunk in chunks:
                quality = self.boundary_analyzer.calculate_boundary_quality(chunk, separator)
                self._chunk_qualities[chunk] = quality
        
        return chunks


class ContentAwareDocumentChunker(DocumentChunker):
    """
    Content-aware document chunker that extends DocumentChunker.
    Maintains full backward compatibility while adding content intelligence.
    """
    
    def __init__(self, enable_content_awareness: bool = True, **kwargs):
        """
        Initialize content-aware chunker.
        
        Args:
            enable_content_awareness: Enable content-aware features (default: True)
            **kwargs: All existing DocumentChunker arguments are supported
        """
        super().__init__(**kwargs)
        
        self.enable_content_awareness = enable_content_awareness
        self.content_detector = ContentTypeDetector()
        
        # Enhanced statistics
        self.stats.update({
            'content_types_detected': {},
            'boundary_quality_avg': 0.0,
            'content_aware_chunks': 0
        })
        
        self._log(f"ContentAwareDocumentChunker initialized (content_awareness={enable_content_awareness})")
    
    def _create_content_aware_splitter(self, text: str) -> RecursiveCharacterTextSplitter:
        """
        Create an appropriate text splitter based on content analysis.
        
        Args:
            text: Full document text for analysis
            
        Returns:
            Configured text splitter
        """
        if not self.enable_content_awareness:
            # Fall back to existing behavior
            return self.splitter
        
        # Detect content type
        content_type, confidence = self.content_detector.detect_content_type(text)
        
        self._log(f"Detected content type: {content_type} (confidence: {confidence:.2f})", "debug")
        
        # Update statistics
        if content_type not in self.stats['content_types_detected']:
            self.stats['content_types_detected'][content_type] = 0
        self.stats['content_types_detected'][content_type] += 1
        
        # Create content-aware splitter
        content_splitter = ContentAwareRecursiveTextSplitter(
            content_type=content_type,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            max_chunk_size=self.chunk_size * 2,
            min_chunk_size=50
        )
        
        # Enable boundary quality tracking
        content_splitter._chunk_qualities = {}
        
        return content_splitter
    
    def chunk_document(
        self,
        text: str,
        source_file: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Generator[ContentAwareChunk, None, None]:
        """
        Enhanced chunking with content-awareness.
        Maintains full backward compatibility with DocumentChunker.
        """
        # Validate inputs (reuse parent validation)
        if not isinstance(text, str):
            raise TypeError("Text must be a string")
        
        validated_path = self._validate_file_path(source_file)
        source_file = str(validated_path)
        
        if len(text) > self.max_file_size:
            raise ValueError(f"Document exceeds maximum size limit ({self.max_file_size} bytes)")
        
        if metadata is None:
            metadata = {}
        
        self._log(f"Chunking document: {Path(source_file).name} ({len(text)} chars)")
        
        try:
            # Create content-aware splitter
            splitter = self._create_content_aware_splitter(text)
            
            # Detect content type for chunk metadata
            content_type, content_confidence = "unknown", 0.0
            if self.enable_content_awareness:
                content_type, content_confidence = self.content_detector.detect_content_type(text)
            
            # Split text into chunks
            text_chunks = splitter.split_text(text)
            
            if not text_chunks:
                self._log("No chunks created from document", "warning")
                return
            
            # Create ContentAwareChunk objects
            char_position = 0
            boundary_qualities = []
            
            for chunk_index, chunk_text in enumerate(text_chunks):
                # Calculate character positions
                start_char = char_position
                end_char = start_char + len(chunk_text)
                
                # Generate secure chunk ID
                chunk_id = self._generate_chunk_id(source_file, chunk_index, chunk_text)
                
                # Get boundary quality if available
                boundary_quality = 0.5  # Default
                if hasattr(splitter, '_chunk_qualities') and chunk_text in splitter._chunk_qualities:
                    boundary_quality = splitter._chunk_qualities[chunk_text]
                    boundary_qualities.append(boundary_quality)
                
                # Extract semantic keywords (simple approach)
                semantic_keywords = self._extract_keywords(chunk_text) if self.enable_content_awareness else []
                
                # Create enhanced metadata
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'creation_time': str(hash(chunk_text) % 1000000),
                    'chunk_method': 'content_aware' if self.enable_content_awareness else 'recursive_character',
                    'chunk_size_config': self.chunk_size,
                    'overlap_config': self.chunk_overlap,
                    'content_type': content_type,
                    'content_confidence': content_confidence,
                    'boundary_quality': boundary_quality
                })
                
                # Create ContentAwareChunk object
                try:
                    chunk = ContentAwareChunk(
                        content=chunk_text,
                        chunk_id=chunk_id,
                        source_file=source_file,
                        chunk_index=chunk_index,
                        start_char=start_char,
                        end_char=end_char,
                        metadata=chunk_metadata,
                        content_type=content_type,
                        content_confidence=content_confidence,
                        boundary_quality=boundary_quality,
                        semantic_keywords=semantic_keywords
                    )
                    
                    self._log(f"Created content-aware chunk {chunk_index}: {len(chunk_text)} chars, "
                            f"type={content_type}, quality={boundary_quality:.2f}", "debug")
                    
                    # Update statistics
                    self.stats['chunks_created'] += 1
                    self.stats['total_characters'] += len(chunk_text)
                    if self.enable_content_awareness:
                        self.stats['content_aware_chunks'] += 1
                    
                    yield chunk
                    
                except Exception as e:
                    self._log(f"Error creating chunk {chunk_index}: {e}", "error")
                    self.stats['errors'] += 1
                    continue
                
                # Update character position for next chunk
                char_position = end_char - self.chunk_overlap if self.chunk_overlap > 0 else end_char
            
            # Update average boundary quality
            if boundary_qualities:
                self.stats['boundary_quality_avg'] = sum(boundary_qualities) / len(boundary_qualities)
            
            # Update document statistics
            self.stats['documents_processed'] += 1
            
            self._log(f"Successfully chunked document into {len(text_chunks)} content-aware chunks")
            
        except Exception as e:
            self._log(f"Error chunking document: {e}", "error")
            self.stats['errors'] += 1
            raise
    
    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """
        Extract semantic keywords from chunk text.
        Simple implementation for initial version.
        """
        if not text or len(text.strip()) < 20:
            return []
        
        # Simple keyword extraction (can be enhanced later)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Filter common stop words
        stop_words = {'that', 'with', 'have', 'this', 'will', 'been', 'from', 'they', 'know', 
                     'want', 'were', 'said', 'each', 'which', 'their', 'time', 'would'}
        
        keywords = [word for word in words if word not in stop_words]
        
        # Return most common keywords
        from collections import Counter
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(max_keywords)]
    
    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Get enhanced statistics including content-aware metrics."""
        stats = super().get_statistics()
        
        if self.enable_content_awareness:
            stats.update({
                'content_types_detected': self.stats['content_types_detected'],
                'boundary_quality_avg': round(self.stats['boundary_quality_avg'], 3),
                'content_aware_chunks': self.stats['content_aware_chunks'],
                'content_awareness_enabled': True
            })
        else:
            stats['content_awareness_enabled'] = False
        
        return stats


# Factory function for easy migration
def create_content_aware_chunker(
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    verbose: bool = False,
    enable_content_awareness: bool = True,
    **kwargs
) -> ContentAwareDocumentChunker:
    """
    Factory function to create a content-aware chunker.
    
    Args:
        chunk_size: Target size for each chunk
        chunk_overlap: Number of characters to overlap between chunks  
        verbose: Enable verbose logging
        enable_content_awareness: Enable content-aware features
        **kwargs: Additional arguments for DocumentChunker
        
    Returns:
        ContentAwareDocumentChunker instance
    """
    return ContentAwareDocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        verbose=verbose,
        enable_content_awareness=enable_content_awareness,
        **kwargs
    )


if __name__ == "__main__":
    """Test the content-aware chunker."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python content_aware_chunker.py <text_content>")
        sys.exit(1)
    
    text = sys.argv[1]
    
    try:
        # Test content-aware chunking
        chunker = create_content_aware_chunker(
            chunk_size=300,
            chunk_overlap=50,
            verbose=True,
            enable_content_awareness=True
        )
        
        chunks = list(chunker.chunk_document(text, "test_file.txt"))
        
        print(f"\nContent-Aware Chunking Results:")
        print(f"Created {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i + 1}:")
            print(f"  ID: {chunk.chunk_id}")
            print(f"  Content Type: {chunk.content_type} (confidence: {chunk.content_confidence:.2f})")
            print(f"  Boundary Quality: {chunk.boundary_quality:.2f}")
            print(f"  Keywords: {chunk.semantic_keywords}")
            print(f"  Length: {chunk.char_count} chars")
            print(f"  Content: {chunk.content[:150]}...")
        
        print(f"\nEnhanced Statistics:")
        stats = chunker.get_enhanced_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)