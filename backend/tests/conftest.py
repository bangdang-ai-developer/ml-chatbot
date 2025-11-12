import pytest
import asyncio
from unittest.mock import Mock

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing"""
    monkeypatch.setenv("GOOGLE_API_KEY", "test_api_key")
    monkeypatch.setenv("MILVUS_HOST", "localhost")
    monkeypatch.setenv("MILVUS_PORT", "19530")
    monkeypatch.setenv("PDF_PATH", "/test/data/test.pdf")
    monkeypatch.setenv("CHUNK_SIZE", "1000")
    monkeypatch.setenv("CHUNK_OVERLAP", "200")

@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing"""
    return """
    Machine Learning Yearning by Andrew Ng

    Chapter 1: Introduction
    Machine learning is the science of getting computers to act without being explicitly programmed.

    Chapter 2: Setting up Development and Test Sets
    Your development and test sets should come from the same distribution.
    """

@pytest.fixture
def sample_document_chunks():
    """Sample document chunks for testing"""
    return [
        {
            "id": "chunk_1",
            "content": "Machine learning is the science of getting computers to act without being explicitly programmed.",
            "metadata": {
                "source": "test.pdf",
                "page_number": 1,
                "chapter": "Chapter 1"
            }
        },
        {
            "id": "chunk_2",
            "content": "Your development and test sets should come from the same distribution.",
            "metadata": {
                "source": "test.pdf",
                "page_number": 2,
                "chapter": "Chapter 2"
            }
        }
    ]