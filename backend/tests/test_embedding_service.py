import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.services.embedding_service import GeminiEmbeddingService
from src.models.chat import DocumentChunk
from src.core.exceptions import EmbeddingError

class TestGeminiEmbeddingService:
    """Test cases for GeminiEmbeddingService"""

    @pytest.fixture
    def embedding_service(self):
        """Create embedding service instance with mocked dependencies"""
        with patch('src.services.embedding_service.genai.configure'):
            with patch('src.services.embedding_service.GoogleGenerativeAIEmbeddings'):
                return GeminiEmbeddingService()

    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing"""
        return [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Error analysis helps improve model performance."
        ]

    @pytest.fixture
    def sample_chunks(self):
        """Sample document chunks for testing"""
        return [
            DocumentChunk(
                id="chunk_1",
                content="Machine learning is a subset of artificial intelligence.",
                metadata={"source": "test.pdf", "page_number": 1}
            ),
            DocumentChunk(
                id="chunk_2",
                content="Deep learning uses neural networks with multiple layers.",
                metadata={"source": "test.pdf", "page_number": 2}
            )
        ]

    @pytest.mark.asyncio
    async def test_generate_embeddings_success(self, embedding_service, sample_texts):
        """Test successful embedding generation"""
        # Mock the embedding service
        embedding_service.embeddings.embed_documents = AsyncMock(return_value=[
            [0.1, 0.2, 0.3] * 256,  # Mock 768-dimensional embedding
            [0.4, 0.5, 0.6] * 256,
            [0.7, 0.8, 0.9] * 256
        ])

        embeddings = await embedding_service.generate_embeddings(sample_texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 768 for emb in embeddings)  # Gemini embedding dimension
        embedding_service.embeddings.embed_documents.assert_called_once_with(sample_texts)

    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_input(self, embedding_service):
        """Test embedding generation with empty input"""
        embeddings = await embedding_service.generate_embeddings([])
        assert embeddings == []

    @pytest.mark.asyncio
    async def test_generate_embeddings_api_error(self, embedding_service, sample_texts):
        """Test embedding generation with API error"""
        embedding_service.embeddings.embed_documents = AsyncMock(
            side_effect=Exception("API Error")
        )

        with pytest.raises(EmbeddingError):
            await embedding_service.generate_embeddings(sample_texts)

    @pytest.mark.asyncio
    async def test_generate_query_embedding_success(self, embedding_service):
        """Test successful query embedding generation"""
        query = "What is machine learning?"
        expected_embedding = [0.1, 0.2, 0.3] * 256

        embedding_service.embeddings.embed_query = Mock(return_value=expected_embedding)

        result = await embedding_service.generate_query_embedding(query)

        assert result == expected_embedding
        assert len(result) == 768
        embedding_service.embeddings.embed_query.assert_called_once_with(query)

    @pytest.mark.asyncio
    async def test_generate_query_embedding_empty_query(self, embedding_service):
        """Test query embedding with empty query"""
        with pytest.raises(EmbeddingError):
            await embedding_service.generate_query_embedding("")

        with pytest.raises(EmbeddingError):
            await embedding_service.generate_query_embedding("   ")

    @pytest.mark.asyncio
    async def test_generate_query_embedding_api_error(self, embedding_service):
        """Test query embedding with API error"""
        embedding_service.embeddings.embed_query = Mock(
            side_effect=Exception("API Error")
        )

        with pytest.raises(EmbeddingError):
            await embedding_service.generate_query_embedding("Test query")

    @pytest.mark.asyncio
    async def test_add_embeddings_to_chunks(self, embedding_service, sample_chunks):
        """Test adding embeddings to document chunks"""
        expected_embeddings = [
            [0.1, 0.2, 0.3] * 256,
            [0.4, 0.5, 0.6] * 256
        ]

        # Mock embedding generation
        embedding_service.generate_embeddings = AsyncMock(return_value=expected_embeddings)

        result_chunks = await embedding_service.add_embeddings_to_chunks(sample_chunks)

        assert len(result_chunks) == 2
        for i, chunk in enumerate(result_chunks):
            assert chunk.embedding == expected_embeddings[i]
            assert chunk.content == sample_chunks[i].content
            assert chunk.id == sample_chunks[i].id

        embedding_service.generate_embeddings.assert_called_once_with(
            [chunk.content for chunk in sample_chunks]
        )

    @pytest.mark.asyncio
    async def test_add_embeddings_to_chunks_empty_input(self, embedding_service):
        """Test adding embeddings to empty chunk list"""
        result = await embedding_service.add_embeddings_to_chunks([])
        assert result == []

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_processing(self, embedding_service):
        """Test batch processing for large text lists"""
        # Create a large list of texts
        large_text_list = [f"Text {i}" for i in range(250)]

        # Mock batch responses
        embedding_service.embeddings.embed_documents = AsyncMock(
            side_effect=[
                [[0.1] * 768] * 100,  # First batch
                [[0.2] * 768] * 100,  # Second batch
                [[0.3] * 768] * 50    # Third batch
            ]
        )

        embeddings = await embedding_service.generate_embeddings(large_text_list)

        assert len(embeddings) == 250
        assert embedding_service.embeddings.embed_documents.call_count == 3