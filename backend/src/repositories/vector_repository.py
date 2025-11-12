from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
from ..models.chat import DocumentChunk
from ..core.config import settings
from ..core.exceptions import VectorStoreError

class VectorRepository(ABC):
    """Abstract base class for vector storage operations"""

    @abstractmethod
    async def store_embeddings(self, chunks: List[DocumentChunk]) -> bool:
        """Store document embeddings"""
        pass

    @abstractmethod
    async def search_similar(self, query_embedding: List[float], limit: int = 5) -> List[DocumentChunk]:
        """Search for similar documents"""
        pass

    @abstractmethod
    async def delete_collection(self) -> bool:
        """Delete the entire collection"""
        pass

class MilvusRepository(VectorRepository):
    """Milvus implementation of vector repository"""

    def __init__(self):
        self.collection_name = "ml_chatbot_docs"
        self.dimension = 768  # Gemini embedding dimension
        self._connect()
        self._create_collection_if_not_exists()

    def _connect(self):
        """Connect to Milvus server"""
        try:
            connections.connect(
                alias="default",
                host=settings.milvus_host,
                port=settings.milvus_port
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to connect to Milvus: {e}")

    def _create_collection_if_not_exists(self):
        """Create collection if it doesn't exist"""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            return

        # Define schema
        id_field = FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True)
        content_field = FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535)
        metadata_field = FieldSchema(name="metadata", dtype=DataType.JSON)
        embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)

        schema = CollectionSchema(
            fields=[id_field, content_field, metadata_field, embedding_field],
            description="ML Chatbot document embeddings"
        )

        # Create collection
        self.collection = Collection(
            name=self.collection_name,
            schema=schema
        )

        # Create index for embeddings
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }

        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )

    async def store_embeddings(self, chunks: List[DocumentChunk]) -> bool:
        """Store document embeddings in Milvus"""
        try:
            if not chunks:
                return True

            # Prepare data
            ids = [chunk.id for chunk in chunks]
            contents = [chunk.content for chunk in chunks]
            metadata_list = [chunk.metadata for chunk in chunks]
            embeddings = [chunk.embedding for chunk in chunks]

            # Insert data
            entities = [
                ids,
                contents,
                metadata_list,
                embeddings
            ]

            self.collection.insert(entities)
            self.collection.flush()

            return True

        except Exception as e:
            raise VectorStoreError(f"Failed to store embeddings: {e}")

    async def search_similar(self, query_embedding: List[float], limit: int = 5) -> List[DocumentChunk]:
        """Search for similar documents"""
        try:
            # Load collection
            self.collection.load()

            # Search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }

            # Perform search
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                expr=None,
                output_fields=["id", "content", "metadata"]
            )

            # Convert to DocumentChunk objects
            chunks = []
            for hit in results[0]:
                if hit.score >= settings.similarity_threshold:
                    chunk = DocumentChunk(
                        id=hit.entity.get("id"),
                        content=hit.entity.get("content"),
                        metadata=hit.entity.get("metadata", {}),
                        confidence=hit.score
                    )
                    chunks.append(chunk)

            return chunks

        except Exception as e:
            raise VectorStoreError(f"Failed to search similar documents: {e}")

    async def delete_collection(self) -> bool:
        """Delete the entire collection"""
        try:
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
            return True
        except Exception as e:
            raise VectorStoreError(f"Failed to delete collection: {e}")