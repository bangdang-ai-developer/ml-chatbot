from abc import ABC, abstractmethod
from typing import List, Dict, Any
import google.generativeai as genai
from ..models.chat import DocumentChunk
from ..core.config import settings
from ..core.exceptions import EmbeddingError
from .vietnamese_text_service import VietnameseTextService

class EmbeddingService(ABC):
    """Abstract base class for embedding generation"""

    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for list of texts"""
        pass

    @abstractmethod
    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a single query"""
        pass

class GeminiEmbeddingService(EmbeddingService):
    """Gemini-based embedding service implementation using native Google API with Vietnamese language support"""

    def __init__(self):
        try:
            genai.configure(api_key=settings.google_api_key)

            # Initialize Vietnamese text service for preprocessing
            self.vietnamese_service = VietnameseTextService()

            # Use the configured embedding model without adding models/ prefix
            self.model_name = settings.embedding_model
            print(f"DEBUG: Initializing embeddings with model: {self.model_name}")

            # Test model availability using the latest API patterns
            try:
                test_embedding = genai.embed_content(
                    model=f"models/{self.model_name}",
                    content="test",
                    task_type="retrieval_document"
                )
                print(f"DEBUG: Successfully initialized embedding model: {self.model_name}")
            except Exception as model_error:
                print(f"WARNING: Failed to initialize {self.model_name}, trying fallback models: {model_error}")
                # Try fallback models with latest available models
                fallback_models = [
                    "text-embedding-004",
                    "text-multilingual-embedding-002",
                    "text-embedding-004"
                ]
                for fallback_model in fallback_models:
                    try:
                        test_embedding = genai.embed_content(
                            model=f"models/{fallback_model}",
                            content="test",
                            task_type="retrieval_document"
                        )
                        self.model_name = fallback_model
                        print(f"DEBUG: Successfully initialized fallback model: {fallback_model}")
                        break
                    except Exception:
                        continue
                else:
                    raise EmbeddingError("No available embedding models found")

        except Exception as e:
            raise EmbeddingError(f"Failed to initialize Gemini embeddings: {e}")

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text with Vietnamese language support
        """
        try:
            # Detect language
            language, confidence = self.vietnamese_service.detect_language(text)

            # Normalize text based on detected language
            if language == 'vietnamese':
                normalized_text = self.vietnamese_service.normalize_vietnamese_text(text)
            else:
                # Basic English preprocessing
                normalized_text = ' '.join(text.split())  # Normalize whitespace

            return normalized_text.strip()
        except Exception as e:
            # Fallback to basic preprocessing if Vietnamese processing fails
            return ' '.join(text.split())

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for list of texts using native Google API with Vietnamese preprocessing"""
        try:
            if not texts:
                return []

            # Preprocess texts with Vietnamese language support
            preprocessed_texts = [self._preprocess_text(text) for text in texts]

            # Process in batches to avoid rate limits
            batch_size = 100  # Google API limit is typically 100-500 texts per request
            all_embeddings = []

            for i in range(0, len(preprocessed_texts), batch_size):
                batch_texts = preprocessed_texts[i:i + batch_size]

                try:
                    # Generate embeddings using native Google API
                    result = genai.embed_content(
                        model=f"models/{self.model_name}",
                        content=batch_texts,
                        task_type="retrieval_document"
                    )

                    # Extract embeddings from the result
                    batch_embeddings = result['embedding']
                    if isinstance(batch_embeddings[0], list):
                        # Multiple embeddings returned
                        all_embeddings.extend(batch_embeddings)
                    else:
                        # Single embedding returned, replicate for each text
                        embedding = batch_embeddings
                        all_embeddings.extend([embedding] * len(batch_texts))

                except Exception as e:
                    raise EmbeddingError(f"Failed to generate embeddings for batch {i//batch_size}: {e}")

            if len(all_embeddings) != len(texts):
                raise EmbeddingError("Number of embeddings doesn't match number of texts")

            return all_embeddings

        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {e}")

    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a single query using native Google API with Vietnamese preprocessing"""
        try:
            if not query or not query.strip():
                raise EmbeddingError("Query cannot be empty")

            # Preprocess query with Vietnamese language support
            preprocessed_query = self._preprocess_text(query.strip())

            # Generate embedding using native Google API
            result = genai.embed_content(
                model=f"models/{self.model_name}",
                content=preprocessed_query,
                task_type="retrieval_query"
            )

            embedding = result['embedding']

            if not embedding:
                raise EmbeddingError("Failed to generate query embedding")

            return embedding

        except Exception as e:
            raise EmbeddingError(f"Failed to generate query embedding: {e}")

    async def generate_enhanced_query_embedding(self, query: str) -> Dict[str, Any]:
        """
        Generate enhanced query embedding with Vietnamese language analysis
        Returns embedding with additional metadata
        """
        try:
            if not query or not query.strip():
                raise EmbeddingError("Query cannot be empty")

            # Process query using Vietnamese text service
            query_analysis = self.vietnamese_service.process_vietnamese_query(query)

            # Generate embedding for the processed query
            embedding = await self.generate_query_embedding(query_analysis['processed_query'])

            return {
                'embedding': embedding,
                'original_query': query_analysis['original_query'],
                'processed_query': query_analysis['processed_query'],
                'detected_language': query_analysis['detected_language'],
                'language_confidence': query_analysis['language_confidence'],
                'keywords': query_analysis['keywords'],
                'expanded_queries': query_analysis['expanded_queries'],
                'complexity_score': query_analysis['complexity_score'],
                'is_vietnamese': query_analysis['is_vietnamese']
            }

        except Exception as e:
            raise EmbeddingError(f"Failed to generate enhanced query embedding: {e}")

    async def add_embeddings_to_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Add embeddings to document chunks with Vietnamese metadata enhancement"""
        try:
            if not chunks:
                return []

            # Extract content from chunks
            texts = [chunk.content for chunk in chunks]

            # Generate embeddings
            embeddings = await self.generate_embeddings(texts)

            # Add embeddings and Vietnamese metadata to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding

                # Create enhanced Vietnamese metadata
                vietnamese_metadata = self.vietnamese_service.create_vietnamese_metadata(
                    chunk.content,
                    chunk_id=chunk.id,
                    page_number=getattr(chunk, 'page_number', None)
                )

                # Merge Vietnamese metadata with existing metadata
                if chunk.metadata is None:
                    chunk.metadata = {}

                chunk.metadata.update(vietnamese_metadata)

            return chunks

        except Exception as e:
            raise EmbeddingError(f"Failed to add embeddings to chunks: {e}")

    async def create_chunk_with_embedding(
        self,
        content: str,
        chunk_id: str = None,
        page_number: int = None
    ) -> DocumentChunk:
        """
        Create a document chunk with embedding and Vietnamese metadata
        """
        try:
            # Generate embedding for the content
            embedding = await self.generate_query_embedding(content)

            # Create Vietnamese metadata
            vietnamese_metadata = self.vietnamese_service.create_vietnamese_metadata(
                content, chunk_id, page_number
            )

            # Create chunk
            chunk = DocumentChunk(
                id=chunk_id,
                content=content,
                embedding=embedding,
                metadata=vietnamese_metadata
            )

            return chunk

        except Exception as e:
            raise EmbeddingError(f"Failed to create chunk with embedding: {e}")