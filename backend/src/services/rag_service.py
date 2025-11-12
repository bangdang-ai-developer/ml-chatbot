from typing import List, Dict, Any, Optional
import logging
from ..models.chat import DocumentChunk, ChatRequest, ChatResponse
from ..repositories.vector_repository import VectorRepository
from ..services.embedding_service import EmbeddingService
from ..services.ai_service import AIService
from ..core.config import settings
from ..core.exceptions import VectorStoreError, EmbeddingError, AIServiceError

class RAGService:
    """Retrieval-Augmented Generation service for the chatbot"""

    def __init__(
        self,
        vector_repository: VectorRepository,
        embedding_service: EmbeddingService,
        ai_service: AIService
    ):
        self.vector_repository = vector_repository
        self.embedding_service = embedding_service
        self.ai_service = ai_service

    async def process_query(self, request: ChatRequest) -> ChatResponse:
        """Process a chat query using RAG pipeline"""
        try:
            # Step 1: Generate embedding for the user query
            query_embedding = await self.embedding_service.generate_query_embedding(request.message)

            # Step 2: Search for relevant documents
            similar_docs = await self.vector_repository.search_similar(
                query_embedding=query_embedding,
                limit=settings.max_retrieved_docs
            )

            if not similar_docs:
                return ChatResponse(
                    message="I couldn't find relevant information in Andrew Ng's Machine Learning Yearning book to answer your question. Please try asking about machine learning concepts, development strategies, or ML system design.",
                    session_id=request.session_id or "default",
                    sources=[],
                    confidence=0.0
                )

            # Step 3: Extract context from similar documents
            context_texts = [doc.content for doc in similar_docs]
            sources = [
                {
                    "chunk_id": doc.id,
                    "page_number": doc.metadata.get("page_number"),
                    "chapter": doc.metadata.get("chapter"),
                    "section": doc.metadata.get("section"),
                    "confidence": getattr(doc, 'confidence', 0.0)
                }
                for doc in similar_docs
            ]

            # Step 4: Generate AI response using RAG
            ai_response = await self.ai_service.generate_response(
                query=request.message,
                context=context_texts
            )

            # Step 5: Calculate overall confidence
            avg_confidence = sum(
                getattr(doc, 'confidence', 0.0) for doc in similar_docs
            ) / len(similar_docs) if similar_docs else 0.0

            # Step 6: Return response
            return ChatResponse(
                message=ai_response,
                session_id=request.session_id or "default",
                sources=sources,
                confidence=avg_confidence
            )

        except EmbeddingError as e:
            raise AIServiceError(f"Embedding generation failed: {e}")
        except VectorStoreError as e:
            raise AIServiceError(f"Vector search failed: {e}")
        except AIServiceError:
            raise
        except Exception as e:
            raise AIServiceError(f"RAG processing failed: {e}")

    async def index_document(self, chunks: List[DocumentChunk]) -> bool:
        """Index document chunks into the vector store"""
        try:
            # Generate embeddings for chunks
            chunks_with_embeddings = await self.embedding_service.add_embeddings_to_chunks(chunks)

            # Store in vector database
            success = await self.vector_repository.store_embeddings(chunks_with_embeddings)

            return success

        except EmbeddingError as e:
            raise AIServiceError(f"Document indexing failed during embedding: {e}")
        except VectorStoreError as e:
            raise AIServiceError(f"Document indexing failed during storage: {e}")
        except Exception as e:
            raise AIServiceError(f"Document indexing failed: {e}")

    async def reindex_all_documents(self, chunks: List[DocumentChunk]) -> bool:
        """Reindex all documents (clear and reindex)"""
        try:
            # Delete existing collection
            await self.vector_repository.delete_collection()

            # Reindex all chunks
            return await self.index_document(chunks)

        except Exception as e:
            raise AIServiceError(f"Document reindexing failed: {e}")

    async def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about indexed documents"""
        try:
            # This would need to be implemented based on your vector store capabilities
            # For now, return basic info
            return {
                "indexed_documents": True,
                "vector_store": "Milvus",
                "embedding_model": settings.embedding_model,
                "similarity_threshold": settings.similarity_threshold
            }

        except Exception as e:
            raise AIServiceError(f"Failed to get document stats: {e}")

    async def validate_query(self, query: str) -> bool:
        """Validate if query is appropriate for the ML chatbot"""
        if not query or not query.strip():
            return False

        query_lower = query.lower().strip()

        # Check if query is related to machine learning
        ml_keywords = [
            "machine learning", "ml", "ai", "artificial intelligence",
            "neural network", "deep learning", "model", "algorithm",
            "training", "dataset", "data", "prediction", "classification",
            "regression", "optimization", "gradient", "loss", "accuracy",
            "andrew ng", "machine learning yearning"
        ]

        # If query contains ML keywords, it's likely relevant
        for keyword in ml_keywords:
            if keyword in query_lower:
                return True

        # If no ML keywords but query is general, still allow it
        # The AI service will handle if it can't provide relevant info
        return len(query.strip()) > 5

    async def retrieve_documents_for_query(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """
        Retrieve relevant documents for a query - used by smart response router
        Returns dict with context texts and sources
        """
        try:
            # Generate embedding for the user query
            query_embedding = await self.embedding_service.generate_query_embedding(query)

            # Search for relevant documents
            similar_docs = await self.vector_repository.search_similar(
                query_embedding=query_embedding,
                limit=limit
            )

            if not similar_docs:
                return {
                    'context_texts': [],
                    'sources': [],
                    'has_content': False,
                    'quality_score': 0.0
                }

            # Extract context from similar documents
            context_texts = [doc.content for doc in similar_docs]
            sources = [
                {
                    "chunk_id": doc.id,
                    "page_number": doc.metadata.get("page_number"),
                    "chapter": doc.metadata.get("chapter"),
                    "section": doc.metadata.get("section"),
                    "confidence": getattr(doc, 'confidence', 0.0)
                }
                for doc in similar_docs
            ]

            # Calculate quality score based on content
            total_content_length = sum(len(text) for text in context_texts)
            quality_score = min(total_content_length / 1000, 1.0)  # Normalize to 0-1

            return {
                'context_texts': context_texts,
                'sources': sources,
                'has_content': len(context_texts) > 0,
                'quality_score': quality_score
            }

        except EmbeddingError as e:
            logger.error(f"Document retrieval embedding failed: {e}")
            return {
                'context_texts': [],
                'sources': [],
                'has_content': False,
                'quality_score': 0.0
            }
        except VectorStoreError as e:
            logger.error(f"Document retrieval search failed: {e}")
            return {
                'context_texts': [],
                'sources': [],
                'has_content': False,
                'quality_score': 0.0
            }
        except Exception as e:
            logger.error(f"Unexpected error in document retrieval: {e}")
            return {
                'context_texts': [],
                'sources': [],
                'has_content': False,
                'quality_score': 0.0
            }