from typing import List, Dict, Any, Optional
import logging
from ..models.chat import DocumentChunk, ChatRequest, ChatResponse
from ..repositories.vector_repository import VectorRepository
from ..services.embedding_service import EmbeddingService
from ..services.ai_service import AIService
from ..services.translation_service import get_translation_service
from ..core.config import settings
from ..core.exceptions import VectorStoreError, EmbeddingError, AIServiceError

logger = logging.getLogger(__name__)

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
        self.translation_service = get_translation_service()

    async def process_query(self, request: ChatRequest) -> ChatResponse:
        """Process a chat query using RAG pipeline with Vietnamese-English translation"""
        try:
            # Step 1: Process query for retrieval (translate if Vietnamese)
            query_processing = await self.translation_service.process_query_for_retrieval(request.message)
            retrieval_query = query_processing['retrieval_query']

            # Log translation for debugging
            if query_processing['is_translated']:
                logger.info(f"Translated Vietnamese query: '{request.message}' → '{retrieval_query}'")
            else:
                logger.info(f"Using original query: '{request.message}'")

            # Step 2: Generate embedding for the retrieval query (English if translated)
            query_embedding = await self.embedding_service.generate_query_embedding(retrieval_query)

            # Step 3: Search for relevant documents using retrieval query
            similar_docs = await self.vector_repository.search_similar(
                query_embedding=query_embedding,
                limit=settings.max_retrieved_docs
            )

            if not similar_docs:
                # Use original query in the no-results message
                return ChatResponse(
                    message=f"I couldn't find relevant information in the Ian Goodfellow Deep Learning book to answer your question: '{request.message}'. Please try asking about deep learning concepts, neural networks, optimization, or technical challenges.",
                    session_id=request.session_id or "default",
                    sources=[]
                )

            # Step 4: Extract context from similar documents
            context_texts = [doc.content for doc in similar_docs]
            sources = [
                {
                    "chunk_id": doc.id,
                    "page_number": doc.metadata.get("page_number"),
                    "chapter": doc.metadata.get("chapter"),
                    "section": doc.metadata.get("section")
                }
                for doc in similar_docs
            ]

            # Step 5: Generate AI response using ORIGINAL query but retrieved context
            # This ensures response is in the user's language
            ai_response = await self.ai_service.generate_response(
                query=request.message,  # Use original Vietnamese query
                context=context_texts
            )

            # Step 6: Return response without confidence calculations
            return ChatResponse(
                message=ai_response,
                session_id=request.session_id or "default",
                sources=sources
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
                "max_retrieved_docs": settings.max_retrieved_docs
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
            logger.debug(f"[CONTEXT RETRIEVAL] Starting retrieval for query: '{query[:100]}...' (limit: {limit})")

            # Process query for retrieval (translate if Vietnamese)
            query_processing = await self.translation_service.process_query_for_retrieval(query)
            retrieval_query = query_processing['retrieval_query']

            # Log translation for debugging
            if query_processing['is_translated']:
                logger.info(f"[CONTEXT RETRIEVAL] Translated query: '{query}' → '{retrieval_query}'")
            else:
                logger.debug(f"[CONTEXT RETRIEVAL] Using original query (no translation needed): '{retrieval_query}'")

            # Generate embedding for the retrieval query (English if translated)
            logger.debug(f"[CONTEXT RETRIEVAL] Generating embedding for: '{retrieval_query[:50]}...'")
            query_embedding = await self.embedding_service.generate_query_embedding(retrieval_query)
            logger.debug(f"[CONTEXT RETRIEVAL] Embedding generated successfully (dimension: {len(query_embedding)})")

            # Search for relevant documents using retrieval query
            logger.debug(f"[CONTEXT RETRIEVAL] Searching for similar documents with limit={limit}")
            similar_docs = await self.vector_repository.search_similar(
                query_embedding=query_embedding,
                limit=limit
            )

            if not similar_docs:
                logger.warning(f"[CONTEXT RETRIEVAL] No documents found for query: '{query[:50]}...'")
                return {
                    'context_texts': [],
                    'sources': [],
                    'has_content': False,
                    'quality_score': 0.0
                }

            logger.info(f"[CONTEXT RETRIEVAL] Found {len(similar_docs)} documents for query: '{query[:50]}...'")

            # Log document details without confidence scores
            for i, doc in enumerate(similar_docs):
                content_preview = doc.content[:100].replace('\n', ' ')
                logger.debug(f"[CONTEXT RETRIEVAL] Doc {i+1}: preview='{content_preview}...'")

            # Extract context from similar documents
            context_texts = [doc.content for doc in similar_docs]
            sources = [
                {
                    "chunk_id": doc.id,
                    "page_number": doc.metadata.get("page_number"),
                    "chapter": doc.metadata.get("chapter"),
                    "section": doc.metadata.get("section")
                }
                for doc in similar_docs
            ]

            # Log the complete context texts being returned
            logger.info(f"[CONTEXT RETRIEVAL] Complete context texts being returned:")
            for i, context in enumerate(context_texts):
                context_clean = context.replace('\n', ' ').replace('\r', ' ')
                logger.info(f"[CONTEXT RETRIEVAL] Context {i+1}/{len(context_texts)}: {context_clean}")

            # Log content metrics without quality scores
            total_content_length = sum(len(text) for text in context_texts)
            logger.info(f"[CONTEXT RETRIEVAL] Content metrics:")
            logger.debug(f"[CONTEXT RETRIEVAL]   - Total context length: {total_content_length} chars")

            return {
                'context_texts': context_texts,
                'sources': sources,
                'has_content': len(context_texts) > 0
            }

        except EmbeddingError as e:
            logger.error(f"Document retrieval embedding failed: {e}")
            return {
                'context_texts': [],
                'sources': [],
                'has_content': False
            }
        except VectorStoreError as e:
            logger.error(f"Document retrieval search failed: {e}")
            return {
                'context_texts': [],
                'sources': [],
                'has_content': False
            }
        except Exception as e:
            logger.error(f"Unexpected error in document retrieval: {e}")
            return {
                'context_texts': [],
                'sources': [],
                'has_content': False
            }