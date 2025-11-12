"""
Hybrid Search Service for Vietnamese RAG
Combines semantic search with keyword-based search optimized for Vietnamese
"""

from typing import List, Dict, Any, Optional, Tuple
import math
import re
from collections import Counter, defaultdict
import logging
from ..repositories.vector_repository import MilvusRepository
from ..services.embedding_service import GeminiEmbeddingService
from ..services.vietnamese_text_service import VietnameseTextService
from ..models.chat import DocumentChunk
from ..core.exceptions import ChatbotException

logger = logging.getLogger(__name__)

class HybridSearchService:
    """
    Hybrid search combining semantic and keyword search for Vietnamese RAG
    """

    def __init__(self, vector_repository: MilvusRepository = None):
        self.vector_repo = vector_repository or MilvusRepository()
        self.embedding_service = GeminiEmbeddingService()
        self.vietnamese_service = VietnameseTextService()

        # Search configuration
        self.semantic_weight = 0.7  # Weight for semantic search
        self.keyword_weight = 0.3   # Weight for keyword search

        # Vietnamese-specific search parameters
        self.min_keyword_length = 2  # Minimum length for Vietnamese keywords
        self.max_keywords = 10       # Maximum keywords to extract

        # BM25 parameters for keyword search
        self.k1 = 1.2  # Controls term frequency saturation
        self.b = 0.75  # Controls document length normalization

        # Document statistics for BM25 (will be calculated dynamically)
        self.doc_stats = None

    async def search(
        self,
        query: str,
        top_k: int = 10,
        search_type: str = "hybrid",
        query_analysis: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword search
        """
        try:
            if search_type == "semantic_only":
                return await self._semantic_search_only(query, top_k)
            elif search_type == "keyword_only":
                return await self._keyword_search_only(query, top_k)
            else:
                return await self._hybrid_search(query, top_k, query_analysis)

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            raise ChatbotException(f"Hybrid search failed: {e}")

    async def _semantic_search_only(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Pure semantic search using embeddings"""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_query_embedding(query)

            # Perform vector search
            results = await self.vector_repo.similarity_search(
                query_vector=query_embedding,
                limit=top_k
            )

            return self._format_search_results(results, "semantic")

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    async def _keyword_search_only(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Pure keyword search using BM25"""
        try:
            # Get all documents (or use text search if available)
            all_docs = await self.vector_repo.get_all_documents(limit=1000)

            # Calculate BM25 scores
            bm25_results = self._calculate_bm25_scores(query, all_docs)

            # Get top results
            top_results = sorted(bm25_results, key=lambda x: x['score'], reverse=True)[:top_k]

            return top_results

        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []

    async def _hybrid_search(
        self,
        query: str,
        top_k: int,
        query_analysis: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Combine semantic and keyword search results
        """
        try:
            # Prepare query analysis if not provided
            if not query_analysis:
                from ..services.query_expansion_service import QueryExpansionService
                query_service = QueryExpansionService()
                query_analysis = query_service.process_query_comprehensive(query)

            # Extract search parameters from query analysis
            is_vietnamese = query_analysis['language_analysis']['is_vietnamese']
            keywords = query_analysis['entities']
            complexity = query_analysis['complexity']['complexity_score']

            # Adjust search weights based on query characteristics
            semantic_weight, keyword_weight = self._adjust_search_weights(
                is_vietnamese, complexity, len(keywords)
            )

            # Perform both searches
            semantic_results = await self._semantic_search_only(query, top_k * 2)
            keyword_results = await self._keyword_search_only(query, top_k * 2)

            # Combine and rank results
            combined_results = self._combine_search_results(
                semantic_results,
                keyword_results,
                semantic_weight,
                keyword_weight
            )

            # Apply Vietnamese-specific re-ranking
            if is_vietnamese:
                combined_results = self._vietnamese_rerank(combined_results, query)

            # Return top_k results
            return combined_results[:top_k]

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fallback to semantic search only
            return await self._semantic_search_only(query, top_k)

    def _adjust_search_weights(
        self,
        is_vietnamese: bool,
        complexity: float,
        keyword_count: int
    ) -> Tuple[float, float]:
        """
        Dynamically adjust search weights based on query characteristics
        """
        semantic_weight = self.semantic_weight
        keyword_weight = self.keyword_weight

        # Vietnamese queries might benefit more from keyword matching
        if is_vietnamese:
            keyword_weight += 0.1
            semantic_weight -= 0.1

        # Complex queries benefit more from semantic understanding
        if complexity > 0.7:
            semantic_weight += 0.15
            keyword_weight -= 0.15

        # Queries with many specific keywords benefit from keyword search
        if keyword_count > 3:
            keyword_weight += 0.1
            semantic_weight -= 0.1

        # Ensure weights sum to 1.0
        total = semantic_weight + keyword_weight
        return semantic_weight / total, keyword_weight / total

    def _calculate_bm25_scores(
        self,
        query: str,
        documents: List[DocumentChunk]
    ) -> List[Dict[str, Any]]:
        """
        Calculate BM25 scores for keyword search
        """
        try:
            # Preprocess query and extract keywords
            processed_query = self.vietnamese_service.normalize_vietnamese_text(query)
            query_keywords = self.vietnamese_service.tokenize_vietnamese_words(processed_query)
            query_keywords = [kw.lower() for kw in query_keywords if len(kw) >= self.min_keyword_length]

            if not query_keywords:
                return []

            # Calculate document statistics
            doc_lengths = [len(doc.content.split()) for doc in documents]
            avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0

            # Calculate term frequencies and document frequencies
            doc_freqs = defaultdict(int)
            term_doc_maps = defaultdict(list)

            for i, doc in enumerate(documents):
                doc_content = self.vietnamese_service.normalize_vietnamese_text(doc.content)
                doc_keywords = self.vietnamese_service.tokenize_vietnamese_words(doc_content)
                doc_keywords = [kw.lower() for kw in doc_keywords]

                doc_term_freq = Counter(doc_keywords)

                for term in set(doc_keywords):
                    doc_freqs[term] += 1
                    term_doc_maps[term].append((i, doc_term_freq.get(term, 0), len(doc_keywords)))

            # Calculate BM25 scores
            results = []
            total_docs = len(documents)

            for i, doc in enumerate(documents):
                score = 0.0
                doc_content = self.vietnamese_service.normalize_vietnamese_text(doc.content)
                doc_keywords = self.vietnamese_service.tokenize_vietnamese_words(doc_content)
                doc_keywords = [kw.lower() for kw in doc_keywords]
                doc_term_freq = Counter(doc_keywords)
                doc_length = len(doc_keywords)

                for term in query_keywords:
                    if term in doc_term_freq:
                        # Term frequency in current document
                        tf = doc_term_freq[term]

                        # Document frequency
                        df = doc_freqs.get(term, 0)

                        if df > 0:
                            # IDF calculation
                            idf = math.log((total_docs - df + 0.5) / (df + 0.5))

                            # BM25 formula
                            term_score = idf * (tf * (self.k1 + 1)) / (
                                tf + self.k1 * (1 - self.b + self.b * (doc_length / avg_doc_length))
                            )
                            score += term_score

                if score > 0:
                    results.append({
                        'document': doc,
                        'score': score,
                        'search_type': 'keyword',
                        'matched_terms': [term for term in query_keywords if term in doc_term_freq]
                    })

            return results

        except Exception as e:
            logger.error(f"Error calculating BM25 scores: {e}")
            return []

    def _combine_search_results(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        semantic_weight: float,
        keyword_weight: float
    ) -> List[Dict[str, Any]]:
        """
        Combine and normalize scores from semantic and keyword search
        """
        try:
            # Create document ID to score mapping
            doc_scores = defaultdict(lambda: {'semantic': 0.0, 'keyword': 0.0, 'doc': None})

            # Process semantic results
            if semantic_results:
                max_semantic_score = max(result['score'] for result in semantic_results)
                for result in semantic_results:
                    doc_id = id(result['document'])
                    normalized_score = result['score'] / max_semantic_score if max_semantic_score > 0 else 0
                    doc_scores[doc_id]['semantic'] = normalized_score
                    doc_scores[doc_id]['doc'] = result['document']

            # Process keyword results
            if keyword_results:
                max_keyword_score = max(result['score'] for result in keyword_results)
                for result in keyword_results:
                    doc_id = id(result['document'])
                    normalized_score = result['score'] / max_keyword_score if max_keyword_score > 0 else 0
                    doc_scores[doc_id]['keyword'] = normalized_score
                    if doc_scores[doc_id]['doc'] is None:
                        doc_scores[doc_id]['doc'] = result['document']

            # Calculate combined scores
            combined_results = []
            for doc_id, scores in doc_scores.items():
                if scores['doc'] is not None:
                    combined_score = (
                        scores['semantic'] * semantic_weight +
                        scores['keyword'] * keyword_weight
                    )

                    combined_results.append({
                        'document': scores['doc'],
                        'score': combined_score,
                        'semantic_score': scores['semantic'],
                        'keyword_score': scores['keyword'],
                        'search_type': 'hybrid'
                    })

            # Sort by combined score
            return sorted(combined_results, key=lambda x: x['score'], reverse=True)

        except Exception as e:
            logger.error(f"Error combining search results: {e}")
            return semantic_results  # Fallback to semantic results

    def _vietnamese_rerank(
        self,
        results: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Apply Vietnamese-specific re-ranking
        """
        try:
            # Extract Vietnamese keywords from query
            query_keywords = self.vietnamese_service.extract_vietnamese_keywords(query)

            # Rerank based on Vietnamese term matching
            for result in results:
                doc_content = result['document'].content.lower()
                bonus_score = 0.0

                # Bonus for exact Vietnamese keyword matches
                for keyword in query_keywords:
                    if keyword in doc_content:
                        bonus_score += 0.1

                # Bonus for Vietnamese language content
                language, confidence = self.vietnamese_service.detect_language(doc_content)
                if language == 'vietnamese':
                    bonus_score += confidence * 0.05

                # Bonus for technical term density
                tech_terms = len([kw for kw in query_keywords if kw in doc_content])
                bonus_score += tech_terms * 0.05

                result['score'] += bonus_score

            # Re-sort after reranking
            return sorted(results, key=lambda x: x['score'], reverse=True)

        except Exception as e:
            logger.error(f"Error in Vietnamese reranking: {e}")
            return results

    def _format_search_results(
        self,
        results: List[Any],
        search_type: str
    ) -> List[Dict[str, Any]]:
        """
        Format search results consistently
        """
        formatted_results = []
        for result in results:
            if isinstance(result, dict) and 'document' in result:
                # Already formatted
                formatted_results.append(result)
            else:
                # Format raw document
                formatted_results.append({
                    'document': result,
                    'score': getattr(result, 'score', 1.0),
                    'search_type': search_type
                })
        return formatted_results

    async def search_with_expansion(
        self,
        query: str,
        query_analysis: Dict[str, Any],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search with query expansion
        """
        try:
            # Get expanded queries
            expanded_queries = query_analysis['expansions']['all_queries']

            all_results = []

            # Search with original query (higher weight)
            main_results = await self.search(
                query,
                top_k=top_k * 2,
                query_analysis=query_analysis
            )
            for result in main_results:
                result['score'] *= 1.0  # Original query gets full weight
            all_results.extend(main_results)

            # Search with expanded queries (lower weight)
            for expanded_query in expanded_queries[1:3]:  # Limit to top 2 expansions
                expanded_results = await self.search(
                    expanded_query,
                    top_k=top_k,
                    query_analysis=query_analysis
                )
                for result in expanded_results:
                    result['score'] *= 0.5  # Expanded queries get half weight
                    result['expanded_from'] = expanded_query
                all_results.extend(expanded_results)

            # Remove duplicates and combine scores
            unique_results = {}
            for result in all_results:
                doc_id = id(result['document'])
                if doc_id in unique_results:
                    unique_results[doc_id]['score'] += result['score']
                else:
                    unique_results[doc_id] = result

            # Sort and return top results
            final_results = sorted(
                unique_results.values(),
                key=lambda x: x['score'],
                reverse=True
            )

            return final_results[:top_k]

        except Exception as e:
            logger.error(f"Error in expanded search: {e}")
            return await self.search(query, top_k)

    async def get_search_statistics(self) -> Dict[str, Any]:
        """
        Get search performance statistics
        """
        try:
            # This would typically connect to monitoring/logging systems
            # For now, return basic configuration info
            return {
                'semantic_weight': self.semantic_weight,
                'keyword_weight': self.keyword_weight,
                'bm25_k1': self.k1,
                'bm25_b': self.b,
                'min_keyword_length': self.min_keyword_length,
                'max_keywords': self.max_keywords,
                'supported_languages': ['vietnamese', 'english'],
                'search_types': ['semantic_only', 'keyword_only', 'hybrid', 'expanded']
            }
        except Exception as e:
            logger.error(f"Error getting search statistics: {e}")
            return {}