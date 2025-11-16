from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
import re
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..models.chat import DocumentChunk
from ..core.config import settings
from ..core.exceptions import DocumentProcessingError

# Import PyMuPDF (fitz) for superior PDF processing
try:
    import fitz
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    raise ImportError("PyMuPDF (pymupdf) is required but not installed. Please install it with: pip install pymupdf>=1.24.0")

logger = logging.getLogger(__name__)

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
    """Enhanced PDF document processing using PyMuPDF (fitz)"""

    def __init__(self):
        if not HAS_PYMUPDF:
            raise DocumentProcessingError("PyMuPDF is not available")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    async def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF with optimized settings"""
        try:
            logger.info(f"Starting PyMuPDF text extraction from: {pdf_path}")

            # Open the PDF with PyMuPDF
            doc = fitz.open(pdf_path)

            if not doc:
                raise DocumentProcessingError(f"Could not open PDF: {pdf_path}")

            extracted_text_parts = []
            total_pages = len(doc)

            logger.info(f"Processing {total_pages} pages with PyMuPDF")

            # Extract text from each page with optimized settings
            for page_num in range(total_pages):
                try:
                    page = doc[page_num]

                    # Get text with layout preservation
                    # Using flags to get better text extraction
                    text = page.get_text(
                        "text",  # Plain text extraction
                        flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES
                    )

                    if text.strip():
                        extracted_text_parts.append(text)

                        # Log progress every 10 pages
                        if (page_num + 1) % 10 == 0:
                            logger.info(f"Processed {page_num + 1}/{total_pages} pages")

                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                    # Continue with other pages even if one fails
                    continue

            doc.close()

            if not extracted_text_parts:
                raise DocumentProcessingError("No text could be extracted from PDF")

            # Combine all extracted text
            full_text = "\n\n".join(extracted_text_parts)

            # Apply basic text cleaning (simplified to avoid regex errors)
            cleaned_text = self._basic_text_cleaning(full_text)

            logger.info(f"Successfully extracted {len(cleaned_text)} characters from {total_pages} pages")

            return cleaned_text

        except Exception as e:
            logger.error(f"PyMuPDF extraction failed for {pdf_path}: {e}")
            raise DocumentProcessingError(f"Failed to extract text from PDF: {e}")

    def _basic_text_cleaning(self, text: str) -> str:
        """Basic text cleaning to fix common PDF extraction issues"""
        if not text:
            return text

        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple blank lines to double
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single

        # Fix common spacing issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # lowercaseUppercase -> lowercase Uppercase
        text = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', text)  # numberLetter -> number Letter
        text = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', text)  # letterNumber -> letter Number

        # Fix punctuation spacing
        text = re.sub(r'\s*([.,;:!?])\s*', r'\1 ', text)  # Normalize punctuation spacing
        text = re.sub(r'\s+', ' ', text)  # Clean up any remaining multiple spaces
        text = text.strip()  # Remove leading/trailing whitespace

        return text

    async def chunk_text(self, text: str) -> List[str]:
        """Split text into manageable chunks using LangChain"""
        try:
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Split text into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            raise DocumentProcessingError(f"Text chunking failed: {e}")

    async def create_document_chunks(self, text_chunks: List[str]) -> List[DocumentChunk]:
        """Create document chunk objects with enhanced metadata"""
        try:
            document_chunks = []

            for i, chunk_text in enumerate(text_chunks):
                # Skip empty chunks
                if not chunk_text or not chunk_text.strip():
                    continue

                # Create unique ID
                chunk_id = f"chunk_{i}_{hash(chunk_text[:100]) % 10000}"

                # Enhanced metadata
                metadata = {
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                    "extraction_method": "PyMuPDF",
                    "text_quality": self._assess_text_quality(chunk_text)
                }

                # Create DocumentChunk object
                chunk = DocumentChunk(
                    id=chunk_id,
                    content=chunk_text.strip(),
                    metadata=metadata
                )

                document_chunks.append(chunk)

            logger.info(f"Created {len(document_chunks)} document chunks")
            return document_chunks

        except Exception as e:
            raise DocumentProcessingError(f"Document chunk creation failed: {e}")

    def _assess_text_quality(self, text: str) -> float:
        """Simple quality assessment for extracted text"""
        if not text:
            return 0.0

        score = 1.0

        # Penalize extremely short chunks
        if len(text) < 50:
            score *= 0.5

        # Check for reasonable word length distribution
        words = text.split()
        if words:
            avg_word_length = sum(len(word.strip('.,!?;:')) for word in words) / len(words)
            # Very long average words might indicate concatenation issues
            if avg_word_length > 15:
                score *= 0.8
            # Very short average words might indicate fragmentation
            elif avg_word_length < 3:
                score *= 0.9

        # Check for presence of alphanumeric content
        alnum_ratio = sum(1 for c in text if c.isalnum()) / len(text) if text else 0
        if alnum_ratio < 0.3:  # Too many special characters
            score *= 0.7

        return max(0.0, min(1.0, score))

# Factory function for dependency injection
def create_document_service() -> DocumentService:
    """Create and return a document service instance"""
    return PDFDocumentService()