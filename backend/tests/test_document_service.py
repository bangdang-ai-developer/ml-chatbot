import pytest
import asyncio
from unittest.mock import Mock, patch
from pathlib import Path
from src.services.document_service import PDFDocumentService
from src.models.chat import DocumentChunk
from src.core.exceptions import DocumentProcessingError

class TestPDFDocumentService:
    """Test cases for PDFDocumentService"""

    @pytest.fixture
    def document_service(self):
        """Create document service instance"""
        return PDFDocumentService()

    @pytest.fixture
    def sample_text(self):
        """Sample text for testing"""
        return """
        [Page 1]
        Machine Learning Yearning
        by Andrew Ng

        This is a comprehensive guide on machine learning strategy.

        [Page 2]
        Chapter 1: Introduction to Machine Learning Strategy
        Machine learning strategy is crucial for building successful ML products.
        """

    @pytest.mark.asyncio
    async def test_extract_text_from_pdf_success(self, document_service):
        """Test successful PDF text extraction"""
        with patch('PyPDF2.PdfReader') as mock_pdf_reader:
            # Mock PDF pages
            mock_page1 = Mock()
            mock_page1.extract_text.return_value = "Page 1 content"
            mock_page2 = Mock()
            mock_page2.extract_text.return_value = "Page 2 content"

            mock_reader = Mock()
            mock_reader.pages = [mock_page1, mock_page2]
            mock_pdf_reader.return_value = mock_reader

            # Mock file operations
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value = Mock()

                result = await document_service.extract_text_from_pdf("test.pdf")

                assert "Page 1 content" in result
                assert "Page 2 content" in result
                assert "[Page 1]" in result
                assert "[Page 2]" in result

    @pytest.mark.asyncio
    async def test_extract_text_from_pdf_file_not_found(self, document_service):
        """Test PDF extraction with non-existent file"""
        with pytest.raises(DocumentProcessingError):
            await document_service.extract_text_from_pdf("nonexistent.pdf")

    @pytest.mark.asyncio
    async def test_chunk_text_success(self, document_service, sample_text):
        """Test successful text chunking"""
        chunks = await document_service.chunk_text(sample_text)

        assert len(chunks) > 0
        assert all(len(chunk.strip()) > 50 for chunk in chunks)
        # Check that page information is preserved
        assert any("[Page 1]" in chunk for chunk in chunks)
        assert any("[Page 2]" in chunk for chunk in chunks)

    @pytest.mark.asyncio
    async def test_chunk_text_empty_input(self, document_service):
        """Test chunking with empty text"""
        with pytest.raises(DocumentProcessingError):
            await document_service.chunk_text("")

    @pytest.mark.asyncio
    async def test_create_document_chunks_success(self, document_service):
        """Test creating document chunks"""
        text_chunks = [
            "[Page 1] This is the first chunk of text.",
            "[Page 2] This is the second chunk with Chapter 1 info."
        ]

        chunks = await document_service.create_document_chunks(text_chunks)

        assert len(chunks) == 2
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.id.startswith("chunk_") for chunk in chunks)
        assert all(chunk.content.strip() for chunk in chunks)

        # Check metadata
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["source"] == "andrew-ng-machine-learning-yearning.pdf"
            assert "page_number" in chunk.metadata
            assert "character_count" in chunk.metadata

    @pytest.mark.asyncio
    async def test_extract_page_info(self, document_service):
        """Test page information extraction"""
        chunk_with_page = "[Page 5] Some content here"
        chunk_without_page = "Just some content without page info"

        page_info_with = document_service._extract_page_info(chunk_with_page)
        page_info_without = document_service._extract_page_info(chunk_without_page)

        assert page_info_with["page_number"] == 5
        assert page_info_with["has_page_info"] is True
        assert page_info_without["page_number"] is None
        assert page_info_without["has_page_info"] is False

    @pytest.mark.asyncio
    async def test_extract_chapter_info(self, document_service):
        """Test chapter information extraction"""
        chapter_chunks = [
            "Chapter 3: Debugging ML Models",
            "Part 2: Setting up Development and Test Sets",
            "1. Introduction to Error Analysis"
        ]

        non_chapter_chunk = "This is just regular content"

        for chunk in chapter_chunks:
            chapter_info = document_service._extract_chapter_info(chunk)
            assert chapter_info is not None
            assert len(chapter_info) > 0

        non_chapter_info = document_service._extract_chapter_info(non_chapter_chunk)
        assert non_chapter_info is None