from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import PyPDF2
import logging
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..models.chat import DocumentChunk
from ..core.config import settings
from ..core.exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)

# Try to import additional PDF libraries
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    logger.warning("pdfplumber not available, falling back to PyPDF2 only")

try:
    import textract
    HAS_TEXTRACT = True
except ImportError:
    HAS_TEXTRACT = False
    logger.warning("textract not available, using PyPDF2 and pdfplumber only")

class DocumentService(ABC):
    """Abstract base class for document processing"""

    @abstractmethod
    async def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        pass

    @abstractmethod
    async def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        pass

    @abstractmethod
    async def create_document_chunks(self, text_chunks: List[str]) -> List[DocumentChunk]:
        """Create document chunk objects with metadata"""
        pass

class PDFDocumentService(DocumentService):
    """Concrete implementation for PDF document processing"""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    async def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file with multiple methods for quality assurance"""
        try:
            logger.info(f"Starting text extraction from: {pdf_path}")

            # Try multiple extraction methods and pick the best one
            extraction_methods = []

            # Method 1: PyPDF2 (existing method)
            try:
                pypdf2_text = self._extract_with_pypdf2(pdf_path)
                if pypdf2_text:
                    extraction_methods.append(("PyPDF2", pypdf2_text))
            except Exception as e:
                logger.warning(f"PyPDF2 extraction failed: {e}")

            # Method 2: pdfplumber (better for complex layouts)
            if HAS_PDFPLUMBER:
                try:
                    pdfplumber_text = self._extract_with_pdfplumber(pdf_path)
                    if pdfplumber_text:
                        extraction_methods.append(("pdfplumber", pdfplumber_text))
                except Exception as e:
                    logger.warning(f"pdfplumber extraction failed: {e}")

            # Method 3: textract (OCR fallback for scanned PDFs)
            if HAS_TEXTRACT:
                try:
                    textract_text = self._extract_with_textract(pdf_path)
                    if textract_text:
                        extraction_methods.append(("textract", textract_text))
                except Exception as e:
                    logger.warning(f"textract extraction failed: {e}")

            if not extraction_methods:
                raise DocumentProcessingError("All PDF extraction methods failed")

            # Choose the best extraction based on content quality
            best_method, best_text = self._choose_best_extraction(extraction_methods)
            logger.info(f"Selected {best_method} for PDF text extraction")

            # Apply content filtering and cleaning
            cleaned_text = self._clean_and_filter_text(best_text)

            if not cleaned_text or len(cleaned_text.strip()) < 100:
                raise DocumentProcessingError("Extracted text is too short or empty after cleaning")

            logger.info(f"Successfully extracted {len(cleaned_text)} characters from PDF")
            return cleaned_text

        except Exception as e:
            raise DocumentProcessingError(f"Failed to extract text from PDF: {e}")

    def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2"""
        text_content = []

        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        page_text_with_metadata = f"[Page {page_num + 1}]\n{page_text}"
                        text_content.append(page_text_with_metadata)
                except Exception as e:
                    logger.debug(f"PyPDF2 error on page {page_num + 1}: {e}")
                    continue

        return "\n\n".join(text_content) if text_content else ""

    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber (better for complex layouts)"""
        text_content = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        page_text_with_metadata = f"[Page {page_num + 1}]\n{page_text}"
                        text_content.append(page_text_with_metadata)
                except Exception as e:
                    logger.debug(f"pdfplumber error on page {page_num + 1}: {e}")
                    continue

        return "\n\n".join(text_content) if text_content else ""

    def _extract_with_textract(self, pdf_path: str) -> str:
        """Extract text using textract (includes OCR for scanned PDFs)"""
        try:
            text = textract.process(pdf_path, method='pdfminer')
            if isinstance(text, bytes):
                text = text.decode('utf-8')

            # Add page numbers (basic implementation)
            lines = text.split('\n')
            text_with_pages = []
            current_page = 1

            for line in lines:
                if line.strip():
                    text_with_pages.append(line)
                # Simple page detection - you might want to improve this
                if len(text_with_pages) > 50 * current_page:  # Rough page detection
                    text_with_pages.insert(-1, f"[Page {current_page + 1}]")
                    current_page += 1

            return "\n".join(text_with_pages)
        except Exception as e:
            logger.debug(f"textract extraction failed: {e}")
            return ""

    def _choose_best_extraction(self, extraction_methods: List[tuple]) -> tuple:
        """Choose the best extraction method based on content quality"""
        best_method = extraction_methods[0][0]
        best_text = extraction_methods[0][1]
        best_score = self._calculate_content_quality(best_text)

        for method, text in extraction_methods[1:]:
            score = self._calculate_content_quality(text)
            logger.debug(f"Content quality score for {method}: {score:.2f}")

            if score > best_score:
                best_score = score
                best_method = method
                best_text = text

        return best_method, best_text

    def _calculate_content_quality(self, text: str) -> float:
        """Calculate content quality score (0-1) based on readability metrics"""
        if not text or len(text.strip()) < 100:
            return 0.0

        # Remove whitespace for calculations
        clean_text = text.strip()

        # Factor 1: ASCII readability (important for English content)
        ascii_ratio = sum(1 for c in clean_text if ord(c) < 128) / len(clean_text)

        # Factor 2: Word density (ratio of alphanumeric characters to total)
        word_chars = sum(1 for c in clean_text if c.isalnum() or c.isspace())
        word_density = word_chars / len(clean_text) if clean_text else 0

        # Factor 3: Line quality (avoid too many short lines)
        lines = [line.strip() for line in clean_text.split('\n') if line.strip()]
        avg_line_length = sum(len(line) for line in lines) / len(lines) if lines else 0
        line_quality = min(avg_line_length / 50, 1.0)  # Normalize to 0-1, 50 chars = ideal

        # Factor 4: Vocabulary diversity (unique words / total words)
        words = clean_text.lower().split()
        unique_words = set(words)
        vocab_diversity = len(unique_words) / len(words) if words else 0

        # Factor 5: Penalize excessive special characters (sign of encoding issues)
        special_char_ratio = sum(1 for c in clean_text if not (c.isalnum() or c.isspace() or c in '.,;:!?-')) / len(clean_text)
        special_penalty = max(0, 1 - (special_char_ratio * 10))  # Penalize heavily

        # Combined score with weights
        score = (
            ascii_ratio * 0.3 +           # 30% weight on ASCII readability
            word_density * 0.25 +         # 25% weight on word density
            line_quality * 0.2 +          # 20% weight on line quality
            vocab_diversity * 0.15 +      # 15% weight on vocabulary diversity
            special_penalty * 0.1         # 10% weight on special character penalty
        )

        return max(0.0, min(1.0, score))

    def _clean_and_filter_text(self, text: str) -> str:
        """Clean and filter extracted text to improve quality"""
        if not text:
            return ""

        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Filter out lines that are likely corrupted
            # Check for excessive special characters or encoding issues
            special_char_count = sum(1 for c in line if not (c.isalnum() or c.isspace() or c in '.,;:!?-()[]'))

            # Skip lines with too many special characters (likely corrupted)
            if len(line) > 0 and special_char_count / len(line) > 0.3:
                continue

            # Skip lines that are just single characters or symbols
            if len(line) <= 3 and not any(c.isalnum() for c in line):
                continue

            # Keep lines that have reasonable content
            if len(line) >= 3 and (any(c.isalpha() for c in line) or any(c.isdigit() for c in line)):
                cleaned_lines.append(line)

        cleaned_text = '\n'.join(cleaned_lines)

        # Final quality check
        if self._calculate_content_quality(cleaned_text) < 0.3:
            logger.warning(f"Text quality is low after cleaning: {self._calculate_content_quality(cleaned_text):.2f}")

        return cleaned_text

    async def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks using LangChain"""
        try:
            chunks = self.text_splitter.split_text(text)

            if not chunks:
                raise DocumentProcessingError("No chunks created from text")

            # Filter out very short chunks
            filtered_chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]

            if not filtered_chunks:
                raise DocumentProcessingError("No valid chunks after filtering")

            return filtered_chunks

        except Exception as e:
            raise DocumentProcessingError(f"Failed to chunk text: {e}")

    async def create_document_chunks(self, text_chunks: List[str], source: str = "andrew-ng-machine-learning-yearning.pdf") -> List[DocumentChunk]:
        """Create document chunk objects with metadata and quality validation"""
        try:
            document_chunks = []

            for i, chunk in enumerate(text_chunks):
                # Calculate content quality for this chunk
                quality_score = self._calculate_content_quality(chunk)

                # Skip very low quality chunks
                if quality_score < 0.2:
                    logger.debug(f"Skipping low-quality chunk {i}: quality={quality_score:.2f}")
                    continue

                # Extract page number if available
                page_info = self._extract_page_info(chunk)

                # Create enhanced metadata
                metadata = {
                    "source": source,
                    "chunk_index": i,
                    "page_number": page_info["page_number"],
                    "chapter": self._extract_chapter_info(chunk),
                    "section": self._extract_section_info(chunk),
                    "character_count": len(chunk),
                    "content_quality": quality_score,
                    "is_readable": quality_score >= 0.5,
                    "extraction_method": "enhanced",
                    "language": self._detect_language(chunk)
                }

                document_chunk = DocumentChunk(
                    id=f"chunk_{source}_{i}_{hash(chunk) % 1000000}",
                    content=chunk.strip(),
                    metadata=metadata
                )

                document_chunks.append(document_chunk)

            logger.info(f"Created {len(document_chunks)} high-quality chunks from {len(text_chunks)} total chunks")
            return document_chunks

        except Exception as e:
            raise DocumentProcessingError(f"Failed to create document chunks: {e}")

    def _detect_language(self, chunk: str) -> str:
        """Simple language detection based on character patterns"""
        if not chunk:
            return "unknown"

        # Check for Vietnamese characters
        vietnamese_chars = set('áàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ')
        vietnamese_count = sum(1 for c in chunk.lower() if c in vietnamese_chars)

        # If more than 2% of characters are Vietnamese-specific, assume Vietnamese
        if vietnamese_count / len(chunk) > 0.02:
            return "vietnamese"

        # Default to English for technical content
        return "english"

    def _extract_page_info(self, chunk: str) -> Dict[str, Any]:
        """Extract page information from chunk"""
        if "[Page " in chunk:
            try:
                page_start = chunk.find("[Page ") + 6
                page_end = chunk.find("]", page_start)
                page_number = int(chunk[page_start:page_end])
                return {"page_number": page_number, "has_page_info": True}
            except:
                return {"page_number": None, "has_page_info": False}
        return {"page_number": None, "has_page_info": False}

    def _extract_chapter_info(self, chunk: str) -> Optional[str]:
        """Extract chapter information from chunk"""
        # Look for patterns like "Chapter X:" or "Part X:"
        import re

        chapter_patterns = [
            r"Chapter\s+\d+[^\n]*",
            r"Part\s+\d+[^\n]*",
            r"\d+\.\s*[A-Z][^\n]*"  # Numbered sections
        ]

        for pattern in chapter_patterns:
            match = re.search(pattern, chunk, re.IGNORECASE)
            if match:
                return match.group().strip()

        return None

    def _extract_section_info(self, chunk: str) -> Optional[str]:
        """Extract section information from chunk"""
        # Look for section headers or important topics
        lines = chunk.split('\n')

        for line in lines[:3]:  # Check first few lines
            line = line.strip()
            if (len(line) < 100 and
                (line.isupper() or
                 line.endswith(':') or
                 any(keyword in line.lower() for keyword in ['introduction', 'conclusion', 'summary', 'example']))):
                return line

        return None