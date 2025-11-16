from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
from ..models.chat import DocumentChunk
from ..core.config import settings
from ..core.exceptions import VectorStoreError

logger = logging.getLogger(__name__)

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
        self.dimension = settings.embedding_dimension  # Use configurable embedding dimension
        logger.info(f"[MILVUS] Initializing repository with dimension: {self.dimension} (from settings.embedding_dimension)")
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
        """Create collection if it doesn't exist with enhanced error handling"""
        logger.info(f"[MILVUS] Checking if collection '{self.collection_name}' exists...")

        # Only create collection if it doesn't exist - preserve existing data
        if utility.has_collection(self.collection_name):
            logger.info(f"[MILVUS] Collection '{self.collection_name}' already exists, loading existing collection")
            # Try to load existing collection
            try:
                from pymilvus import Collection as MilvusCollection
                self.collection = MilvusCollection(self.collection_name)
                self.collection.load()
                logger.info(f"[MILVUS] Existing collection loaded successfully")

                # Verify collection has the right schema
                stats = self.collection.describe()
                if len(stats.get('fields', [])) == 4:
                    logger.info(f"[MILVUS] Existing collection has correct schema with {len(stats['fields'])} fields")
                    return True
                else:
                    logger.warning(f"[MILVUS] Existing collection has {len(stats.get('fields', []))} fields, expected 4")
                    logger.info(f"[MILVUS] Dropping collection with wrong schema...")
                    utility.drop_collection(self.collection_name)

            except Exception as load_error:
                logger.error(f"[MILVUS] Failed to load existing collection: {load_error}")
                logger.info(f"[MILVUS] Dropping corrupted collection...")
                utility.drop_collection(self.collection_name)

        # Create new collection (either didn't exist or was dropped)
        logger.info(f"[MILVUS] Creating new collection '{self.collection_name}'...")

        # Define schema
        logger.debug(f"[MILVUS] Defining collection schema...")
        id_field = FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True)
        content_field = FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535)
        metadata_field = FieldSchema(name="metadata", dtype=DataType.JSON)
        embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)

        schema = CollectionSchema(
            fields=[id_field, content_field, metadata_field, embedding_field],
            description="ML Chatbot document embeddings"
        )
        logger.info(f"[MILVUS] Collection schema created successfully")

        # Create collection
        logger.info(f"[MILVUS] Creating collection '{self.collection_name}'...")
        try:
            # Create the collection without any local imports
            from pymilvus import Collection as MilvusCollection
            self.collection = MilvusCollection(
                name=self.collection_name,
                schema=schema
            )
            logger.info(f"[MILVUS] Collection '{self.collection_name}' created successfully")
        except Exception as create_error:
            logger.error(f"[MILVUS] Failed to create collection: {create_error}")
            raise VectorStoreError(f"Failed to create Milvus collection: {create_error}")

        # Create index for embeddings
        logger.info(f"[MILVUS] Creating vector index...")
        try:
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }

            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            logger.info(f"[MILVUS] Vector index created successfully")
        except Exception as index_error:
            logger.error(f"[MILVUS] Failed to create index: {index_error}")
            raise VectorStoreError(f"Failed to create vector index: {index_error}")

        # Verify collection is ready
        logger.info(f"[MILVUS] Verifying collection state...")
        try:
            self.collection.load()
            logger.info(f"[MILVUS] Collection loaded successfully")

            # Get collection stats
            stats = self.collection.describe()
            logger.info(f"[MILVUS] Collection ready: {len(stats['fields'])} fields, dimension={self.dimension}")

            return True
        except Exception as load_error:
            logger.error(f"[MILVUS] Failed to load collection: {load_error}")
            raise VectorStoreError(f"Failed to load collection: {load_error}")

    async def store_embeddings(self, chunks: List[DocumentChunk]) -> bool:
        """Store document embeddings in Milvus with enhanced verification"""
        logger.info(f"[MILVUS] ===== store_embeddings method called with {len(chunks)} chunks =====")
        try:
            if not chunks:
                logger.info("[MILVUS] No chunks to store")
                return True

            logger.info(f"[MILVUS] Preparing to store {len(chunks)} document chunks...")

            # Refresh collection reference to ensure it's current
            logger.info(f"[MILVUS] Refreshing collection reference...")
            try:
                from pymilvus import Collection as MilvusCollection
                self.collection = MilvusCollection(self.collection_name)
                self.collection.load()
                logger.info(f"[MILVUS] Collection reference refreshed successfully")
            except Exception as refresh_error:
                logger.error(f"[MILVUS] Failed to refresh collection reference: {refresh_error}")
                # Try to recreate collection
                logger.info(f"[MILVUS] Attempting to recreate collection...")
                self._create_collection_if_not_exists()

            # Verify collection is ready
            if not self.collection:
                logger.error("[MILVUS] Collection not initialized")
                raise VectorStoreError("Collection not initialized")

            # Prepare data
            ids = [chunk.id for chunk in chunks]
            contents = [chunk.content for chunk in chunks]
            metadata_list = [chunk.metadata for chunk in chunks]
            embeddings = [chunk.embedding for chunk in chunks]

            # Verify data integrity
            logger.info(f"[MILVUS] Data preparation complete:")
            logger.info(f"[MILVUS]   - IDs: {len(ids)}")
            logger.info(f"[MILVUS]   - Contents: {len(contents)}")
            logger.info(f"[MILVUS]   - Metadata: {len(metadata_list)}")
            logger.info(f"[MILVUS]   - Embeddings: {len(embeddings)}")
            logger.info(f"[MILVUS]   - Embedding dimension: {len(embeddings[0]) if embeddings else 'N/A'}")

            # Validate embedding dimensions
            if embeddings and len(embeddings[0]) != self.dimension:
                logger.error(f"[MILVUS] Embedding dimension mismatch: expected {self.dimension}, got {len(embeddings[0])}")
                raise VectorStoreError(f"Embedding dimension mismatch: expected {self.dimension}, got {len(embeddings[0])}")

            # Insert data with retry logic
            logger.info(f"[MILVUS] Inserting data into collection...")
            entities = [
                ids,
                contents,
                metadata_list,
                embeddings
            ]

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    insert_result = self.collection.insert(entities)
                    logger.info(f"[MILVUS] Data insertion completed on attempt {attempt + 1}, insert result: {insert_result}")
                    break
                except Exception as insert_error:
                    if attempt == max_retries - 1:
                        raise insert_error
                    logger.warning(f"[MILVUS] Insert attempt {attempt + 1} failed, retrying... Error: {insert_error}")
                    import time
                    time.sleep(1)

            # Flush to ensure data is persisted
            logger.info(f"[MILVUS] Flushing data to persistent storage...")
            self.collection.flush()

            # Verify insertion was successful
            logger.info(f"[MILVUS] Verifying insertion...")
            collection_stats = self.collection.describe()
            logger.info(f"[MILVUS] Collection statistics after insertion: {collection_stats}")

            logger.info(f"[MILVUS] Successfully stored {len(chunks)} document chunks")
            return True

        except Exception as e:
            logger.error(f"[MILVUS] Failed to store embeddings: {e}")
            logger.error(f"[MILVUS] Error details: {type(e).__name__}: {str(e)}")
            raise VectorStoreError(f"Failed to store embeddings: {e}")

    async def search_similar(self, query_embedding: List[float], limit: int = 5) -> List[DocumentChunk]:
        """Search for similar documents with enhanced error handling and verification"""
        try:
            logger.info(f"[VECTOR SEARCH] ===== Starting similarity search with limit={limit} =====")
            logger.debug(f"[VECTOR SEARCH] Query embedding dimension: {len(query_embedding)}")
            logger.debug(f"[VECTOR SEARCH] Query embedding sample: {query_embedding[:3] if query_embedding else 'None'}")

            # Refresh collection reference to ensure it's current
            logger.info(f"[VECTOR SEARCH] Refreshing collection reference...")
            try:
                from pymilvus import Collection as MilvusCollection
                self.collection = MilvusCollection(self.collection_name)
                self.collection.load()
                logger.info(f"[VECTOR SEARCH] Collection reference refreshed successfully")
            except Exception as refresh_error:
                logger.error(f"[VECTOR SEARCH] Failed to refresh collection reference: {refresh_error}")
                raise VectorStoreError(f"Failed to refresh collection for search: {refresh_error}")

            # Verify collection is properly loaded and has data
            logger.debug(f"[VECTOR SEARCH] Verifying collection state...")
            if not self.collection:
                logger.error(f"[VECTOR SEARCH] Collection not initialized")
                raise VectorStoreError("Collection not initialized for search")

            # Check collection statistics
            try:
                num_entities = self.collection.num_entities
                stats = self.collection.describe()
                logger.info(f"[VECTOR SEARCH] Collection stats: {len(stats['fields'])} fields, dimension={self.dimension}")
                logger.info(f"[VECTOR SEARCH] Collection entities: {num_entities}")

                if num_entities == 0:
                    logger.warning(f"[VECTOR SEARCH] Collection is empty, no documents to search")
                    return []

            except Exception as stats_error:
                logger.error(f"[VECTOR SEARCH] Failed to get collection stats: {stats_error}")
                # Continue anyway, as this might not be critical for search

            # Verify index state and ensure index exists
            logger.debug(f"[VECTOR SEARCH] Verifying index state...")
            self._ensure_index_exists()

            # Get index info to match search parameters
            try:
                index_info = self.collection.index()
                if index_info:
                    logger.info(f"[VECTOR SEARCH] Using index: {index_info}")
                else:
                    logger.warning(f"[VECTOR SEARCH] No index info available")
            except Exception as index_error:
                logger.warning(f"[VECTOR SEARCH] Failed to get index info: {index_error}")

            # Brief pause to ensure index is ready for search
            import time
            time.sleep(0.5)

            # Search parameters matching index configuration
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }

            logger.info(f"[VECTOR SEARCH] Search parameters: {search_params}")
            logger.debug(f"[VECTOR SEARCH] Executing vector search...")

            # Perform search with enhanced error handling
            try:
                results = self.collection.search(
                    data=[query_embedding],
                    anns_field="embedding",
                    param=search_params,
                    limit=limit,
                    expr=None,
                    output_fields=["id", "content", "metadata"]
                )
                logger.info(f"[VECTOR SEARCH] Search completed successfully")
            except Exception as search_error:
                logger.error(f"[VECTOR SEARCH] Search execution failed: {search_error}")
                logger.error(f"[VECTOR SEARCH] Search parameters used: {search_params}")
                logger.error(f"[VECTOR SEARCH] Query embedding length: {len(query_embedding)}")
                raise VectorStoreError(f"Vector search failed: {search_error}")

            total_results = len(results[0]) if results else 0
            logger.info(f"[VECTOR SEARCH] Search completed. Total results found: {total_results}")

            # Convert to DocumentChunk objects - all results included, sorted by relevance score
            chunks = []

            for i, hit in enumerate(results[0]):
                chunk = DocumentChunk(
                    id=hit.entity.get("id"),
                    content=hit.entity.get("content"),
                    metadata=hit.entity.get("metadata", {})
                )
                chunks.append(chunk)

                # Log details for all retrieved results
                content_clean = chunk.content.replace('\n', ' ').replace('\r', ' ')
                logger.debug(f"[VECTOR SEARCH] Result {i+1}: score={hit.score:.3f}, id={chunk.id}")
                logger.debug(f"[VECTOR SEARCH] Content preview: {content_clean[:200]}...")

            logger.info(f"[VECTOR SEARCH] Results summary:")
            logger.info(f"[VECTOR SEARCH]   - Total results returned: {len(chunks)}")
            logger.info(f"[VECTOR SEARCH]   - Results sorted by relevance score")

            return chunks

        except Exception as e:
            logger.error(f"[VECTOR SEARCH] Search failed: {e}")
            raise VectorStoreError(f"Failed to search similar documents: {e}")

    def _ensure_index_exists(self):
        """Ensure vector index exists for the collection"""
        try:
            logger.debug(f"[MILVUS] Checking if index exists...")
            index_info = self.collection.index()

            if not index_info:
                logger.info(f"[MILVUS] No index found, creating vector index...")
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                }

                self.collection.create_index(
                    field_name="embedding",
                    index_params=index_params
                )
                logger.info(f"[MILVUS] Vector index created successfully")

                # Wait a moment for index to be created
                import time
                time.sleep(1.0)

            else:
                logger.debug(f"[MILVUS] Index exists: {index_info}")

        except Exception as e:
            logger.error(f"[MILVUS] Failed to ensure index exists: {e}")
            # Continue anyway, as index creation might not be strictly necessary
            # if it was already created during collection setup

    async def delete_collection(self) -> bool:
        """Delete the entire collection"""
        try:
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
            return True
        except Exception as e:
            raise VectorStoreError(f"Failed to delete collection: {e}")