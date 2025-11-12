import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from src.main import app
from src.models.chat import ChatRequest, ChatResponse

class TestChatAPI:
    """Test cases for chat API endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    def mock_rag_service(self):
        """Mock RAG service"""
        with patch('src.api.chat.get_rag_service') as mock_get_rag:
            mock_service = Mock()
            mock_get_rag.return_value = mock_service
            yield mock_service

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "ML Chatbot API" in response.json()["message"]

    def test_chat_endpoint_success(self, client, mock_rag_service):
        """Test successful chat request"""
        # Mock RAG service response
        mock_response = ChatResponse(
            message="Machine learning is a subset of AI...",
            session_id="test_session",
            sources=[{
                "chunk_id": "chunk_1",
                "page_number": 1,
                "chapter": "Chapter 1",
                "confidence": 0.85
            }],
            confidence=0.85
        )

        mock_rag_service.process_query = AsyncMock(return_value=mock_response)
        mock_rag_service.validate_query = AsyncMock(return_value=True)

        request_data = {
            "message": "What is machine learning?",
            "session_id": "test_session"
        }

        response = client.post("/api/v1/chat", json=request_data)

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["message"] == "Machine learning is a subset of AI..."
        assert response_data["session_id"] == "test_session"
        assert len(response_data["sources"]) == 1
        assert response_data["confidence"] == 0.85

    def test_chat_endpoint_invalid_query(self, client, mock_rag_service):
        """Test chat endpoint with invalid query"""
        mock_rag_service.validate_query = AsyncMock(return_value=False)

        request_data = {
            "message": "",  # Empty message
            "session_id": "test_session"
        }

        response = client.post("/api/v1/chat", json=request_data)
        assert response.status_code == 400
        assert "Invalid query" in response.json()["detail"]

    def test_chat_endpoint_service_error(self, client, mock_rag_service):
        """Test chat endpoint with service error"""
        mock_rag_service.validate_query = AsyncMock(return_value=True)
        mock_rag_service.process_query = AsyncMock(
            side_effect=Exception("Service error")
        )

        request_data = {
            "message": "What is machine learning?",
            "session_id": "test_session"
        }

        response = client.post("/api/v1/chat", json=request_data)
        assert response.status_code == 500

    def test_index_document_endpoint(self, client, mock_rag_service):
        """Test document indexing endpoint"""
        request_data = {}
        response = client.post("/api/v1/index-document", json=request_data)

        assert response.status_code == 200
        assert "started in background" in response.json()["message"]

    def test_get_indexing_status(self, client):
        """Test indexing status endpoint"""
        response = client.get("/api/v1/indexing-status")
        assert response.status_code == 200

        response_data = response.json()
        assert "status" in response_data
        assert "indexed_chunks" in response_data
        assert "last_updated" in response_data

    def test_get_chatbot_stats(self, client, mock_rag_service):
        """Test chatbot statistics endpoint"""
        mock_stats = {
            "indexed_documents": True,
            "vector_store": "Milvus",
            "embedding_model": "text-embedding-004",
            "similarity_threshold": 0.7
        }

        mock_rag_service.get_document_stats = AsyncMock(return_value=mock_stats)

        response = client.get("/api/v1/stats")
        assert response.status_code == 200

        response_data = response.json()
        assert response_data["indexed_documents"] is True
        assert response_data["vector_store"] == "Milvus"
        assert response_data["embedding_model"] == "text-embedding-004"

    def test_reindex_documents_endpoint(self, client, mock_rag_service):
        """Test document reindexing endpoint"""
        request_data = {}
        response = client.post("/api/v1/reindex", json=request_data)

        assert response.status_code == 200
        assert "started in background" in response.json()["message"]

    def test_chat_request_validation(self, client):
        """Test chat request validation"""
        # Test missing message
        response = client.post("/api/v1/chat", json={"session_id": "test"})
        assert response.status_code == 422  # Validation error

        # Test invalid data types
        response = client.post("/api/v1/chat", json={
            "message": 123,  # Should be string
            "session_id": "test"
        })
        assert response.status_code == 422