"""
Cross-Encoder Reranking Service for Vietnamese RAG
Advanced reranking optimized for Vietnamese language processing
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging
from ..models.chat import DocumentChunk
from ..services.vietnamese_text_service import VietnameseTextService
from ..services.embedding_service import GeminiEmbeddingService
from ..core.exceptions import ChatbotException

logger = logging.getLogger(__name__)

class VietnameseRerankingService:
    """
    Cross-encoder reranking service optimized for Vietnamese RAG systems
    """

    def __init__(self, vietnamese_service: VietnameseTextService = None):
        self.vietnamese_service = vietnamese_service or VietnameseTextService()
        self.embedding_service = GeminiEmbeddingService()

        # Reranking configuration
        self.min_score_threshold = 0.1  # Minimum score to keep document
        self.max_results = 10           # Maximum results to return
        self.diversity_boost = 0.2      # Boost for diverse content

        # Vietnamese-specific weighting
        self.vietnamese_term_weight = 1.5   # Boost for Vietnamese keyword matches
        self.technical_term_weight = 1.3   # Boost for technical ML terms
        self.content_type_weights = {
            'definition': 1.2,
            'example': 1.1,
            'technical': 1.3,
            'practical': 1.0
        }

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        query_analysis: Dict[str, Any] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using multiple scoring strategies optimized for Vietnamese
        """
        try:
            if not documents:
                return []

            # Prepare query analysis if not provided
            if not query_analysis:
                from ..services.query_expansion_service import QueryExpansionService
                query_service = QueryExpansionService()
                query_analysis = query_service.process_query_comprehensive(query)

            # Calculate relevance scores using multiple strategies
            reranked_docs = []
            for doc_data in documents:
                doc = doc_data['document'] if 'document' in doc_data else doc_data
                base_score = doc_data.get('score', 0.0)

                # Calculate enhanced relevance score
                enhanced_score = await self._calculate_enhanced_score(
                    query, doc, query_analysis, base_score
                )

                reranked_docs.append({
                    'document': doc,
                    'original_score': base_score,
                    'enhanced_score': enhanced_score,
                    'score_breakdown': self._get_score_breakdown(
                        query, doc, query_analysis, base_score
                    )
                })

            # Apply diversity penalty
            reranked_docs = self._apply_diversity_boost(reranked_docs)

            # Sort by enhanced score
            reranked_docs.sort(key=lambda x: x['enhanced_score'], reverse=True)

            # Filter and return top results
            filtered_docs = [
                doc for doc in reranked_docs
                if doc['enhanced_score'] >= self.min_score_threshold
            ]

            return filtered_docs[:top_k]

        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            # Fallback to original order
            return documents[:top_k]

    async def _calculate_enhanced_score(
        self,
        query: str,
        document: DocumentChunk,
        query_analysis: Dict[str, Any],
        base_score: float
    ) -> float:
        """
        Calculate enhanced relevance score using multiple factors
        """
        try:
            # Start with base score
            enhanced_score = base_score

            # 1. Semantic similarity score (using embeddings)
            semantic_score = await self._calculate_semantic_similarity(query, document.content)
            enhanced_score += semantic_score * 0.4

            # 2. Vietnamese keyword matching score
            vietnamese_score = self._calculate_vietnamese_keyword_score(
                query, document.content, query_analysis
            )
            enhanced_score += vietnamese_score * 0.3

            # 3. Content type relevance score
            content_score = self._calculate_content_type_score(document, query_analysis)
            enhanced_score += content_score * 0.2

            # 4. Technical term density score
            technical_score = self._calculate_technical_density_score(document, query_analysis)
            enhanced_score += technical_score * 0.1

            return enhanced_score

        except Exception as e:
            logger.error(f"Error calculating enhanced score: {e}")
            return base_score

    async def _calculate_semantic_similarity(self, query: str, doc_content: str) -> float:
        """
        Calculate semantic similarity using embeddings
        """
        try:
            # Get embeddings for query and document
            query_embedding = await self.embedding_service.generate_query_embedding(query)
            doc_embedding = await self.embedding_service.generate_query_embedding(doc_content)

            # Calculate cosine similarity
            query_vec = np.array(query_embedding)
            doc_vec = np.array(doc_embedding)

            if np.linalg.norm(query_vec) == 0 or np.linalg.norm(doc_vec) == 0:
                return 0.0

            similarity = np.dot(query_vec, doc_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
            )

            return max(0.0, similarity)  # Ensure non-negative

        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0

    def _calculate_vietnamese_keyword_score(
        self,
        query: str,
        doc_content: str,
        query_analysis: Dict[str, Any]
    ) -> float:
        """
        Calculate keyword matching score optimized for Vietnamese
        """
        try:
            # Normalize Vietnamese text
            query_normalized = self.vietnamese_service.normalize_vietnamese_text(query)
            doc_normalized = self.vietnamese_service.normalize_vietnamese_text(doc_content)

            # Extract keywords from query
            query_keywords = self.vietnamese_service.tokenize_vietnamese_words(query_normalized)
            doc_keywords = self.vietnamese_service.tokenize_vietnamese_words(doc_normalized)

            # Calculate keyword overlap
            query_keyword_set = set(kw.lower() for kw in query_keywords)
            doc_keyword_set = set(kw.lower() for kw in doc_keywords)

            if not query_keyword_set:
                return 0.0

            overlap = len(query_keyword_set & doc_keyword_set)
            precision = overlap / len(query_keyword_set)

            # Boost for Vietnamese technical terms
            vietnamese_terms = query_analysis.get('entities', [])
            vietnamese_boost = sum(
                self.vietnamese_term_weight for term in vietnamese_terms
                if term.lower() in doc_normalized.lower()
            ) / max(len(vietnamese_terms), 1)

            return precision + (vietnamese_boost * 0.5)

        except Exception as e:
            logger.error(f"Error calculating Vietnamese keyword score: {e}")
            return 0.0

    def _calculate_content_type_score(
        self,
        document: DocumentChunk,
        query_analysis: Dict[str, Any]
    ) -> float:
        """
        Calculate score based on content type relevance
        """
        try:
            # Get content type from metadata
            metadata = document.metadata or {}
            content_type = metadata.get('content_type', 'general')

            # Get query intent
            query_intent = query_analysis.get('intent_analysis', {}).get('primary_intent', 'general')

            # Content type and intent matching
            intent_type_mapping = {
                'definition': ['definition'],
                'examples': ['example', 'practical'],
                'how_to': ['practical', 'technical'],
                'comparison': ['technical'],
                'advantages_disadvantages': ['technical', 'practical']
            }

            preferred_types = intent_type_mapping.get(query_intent, [])
            base_weight = self.content_type_weights.get(content_type, 1.0)

            # Boost if content type matches intent preference
            intent_boost = 1.2 if content_type in preferred_types else 1.0

            # Vietnamese content boost
            is_vietnamese = metadata.get('is_vietnamese', False)
            query_is_vietnamese = query_analysis.get('language_analysis', {}).get('is_vietnamese', False)

            language_boost = 1.3 if is_vietnamese and query_is_vietnamese else 1.0

            return base_weight * intent_boost * language_boost - 1.0  # Normalize to 0-based

        except Exception as e:
            logger.error(f"Error calculating content type score: {e}")
            return 0.0

    def _calculate_technical_density_score(
        self,
        document: DocumentChunk,
        query_analysis: Dict[str, Any]
    ) -> float:
        """
        Calculate score based on technical term density
        """
        try:
            doc_content = document.content.lower()

            # Vietnamese ML technical terms
            technical_terms = [
                'học máy', 'trí tuệ nhân tạo', 'mạng nơ-ron', 'học sâu', 'thuật toán',
                'mô hình', 'dữ liệu', 'huấn luyện', 'phân loại', 'hồi quy', 'tối ưu',
                'machine learning', 'ai', 'neural network', 'deep learning', 'algorithm',
                'model', 'data', 'training', 'classification', 'regression', 'optimization'
            ]

            # Count technical terms
            tech_term_count = sum(1 for term in technical_terms if term in doc_content)
            doc_word_count = len(document.content.split())

            if doc_word_count == 0:
                return 0.0

            # Calculate density
            density = tech_term_count / doc_word_count

            # Boost based on query complexity
            complexity = query_analysis.get('complexity', {}).get('complexity_score', 0.5)
            complexity_boost = 1.0 + complexity

            return min(density * 10 * complexity_boost, 1.0)  # Cap at 1.0

        except Exception as e:
            logger.error(f"Error calculating technical density score: {e}")
            return 0.0

    def _apply_diversity_boost(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply diversity boost to ensure varied content
        """
        try:
            if len(documents) <= 1:
                return documents

            # Sort by original enhanced score
            documents.sort(key=lambda x: x['enhanced_score'], reverse=True)

            # Apply diversity penalty for similar content
            for i, doc_i in enumerate(documents):
                for j, doc_j in enumerate(documents[i+1:], i+1):
                    similarity = self._calculate_content_similarity(
                        doc_i['document'].content,
                        doc_j['document'].content
                    )

                    if similarity > 0.8:  # High similarity threshold
                        # Apply penalty to lower-ranked document
                        penalty = (similarity - 0.8) * self.diversity_boost
                        documents[j]['enhanced_score'] *= (1 - penalty)

            # Re-sort after diversity adjustments
            documents.sort(key=lambda x: x['enhanced_score'], reverse=True)

            return documents

        except Exception as e:
            logger.error(f"Error applying diversity boost: {e}")
            return documents

    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """
        Calculate content similarity for diversity checking
        """
        try:
            # Simple word overlap similarity for diversity checking
            words1 = set(content1.lower().split())
            words2 = set(content2.lower().split())

            if not words1 or not words2:
                return 0.0

            intersection = len(words1 & words2)
            union = len(words1 | words2)

            return intersection / union if union > 0 else 0.0

        except Exception as e:
            logger.error(f"Error calculating content similarity: {e}")
            return 0.0

    def _get_score_breakdown(
        self,
        query: str,
        document: DocumentChunk,
        query_analysis: Dict[str, Any],
        base_score: float
    ) -> Dict[str, float]:
        """
        Get detailed score breakdown for debugging
        """
        try:
            breakdown = {
                'base_score': base_score,
                'semantic_similarity': 0.0,
                'vietnamese_keyword_match': 0.0,
                'content_type_relevance': 0.0,
                'technical_density': 0.0
            }

            # Calculate individual components
            content_lower = document.content.lower()
            query_lower = query.lower()

            # Vietnamese keyword matching
            query_keywords = self.vietnamese_service.tokenize_vietnamese_words(query)
            doc_keywords = self.vietnamese_service.tokenize_vietnamese_words(document.content)
            overlap = len(set(kw.lower() for kw in query_keywords) &
                          set(kw.lower() for kw in doc_keywords))
            breakdown['vietnamese_keyword_match'] = overlap / max(len(query_keywords), 1)

            # Technical density
            technical_terms = ['học máy', 'ai', 'mạng nơ-ron', 'thuật toán', 'mô hình']
            tech_count = sum(1 for term in technical_terms if term in content_lower)
            breakdown['technical_density'] = min(tech_count / 10, 1.0)

            # Content type
            metadata = document.metadata or {}
            content_type = metadata.get('content_type', 'general')
            breakdown['content_type_relevance'] = self.content_type_weights.get(content_type, 1.0) - 1.0

            return breakdown

        except Exception as e:
            logger.error(f"Error getting score breakdown: {e}")
            return {'error': str(e)}

    async def adaptive_rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        query_analysis: Dict[str, Any],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Adaptive reranking based on query complexity and language
        """
        try:
            # Get query characteristics
            is_vietnamese = query_analysis.get('language_analysis', {}).get('is_vietnamese', False)
            complexity = query_analysis.get('complexity', {}).get('complexity_score', 0.5)
            intent = query_analysis.get('intent_analysis', {}).get('primary_intent', 'general')

            # Adjust reranking strategy based on characteristics
            if is_vietnamese and complexity > 0.7:
                # Complex Vietnamese query - emphasize keyword matching
                self.vietnamese_term_weight = 2.0
                return await self.rerank(query, documents, query_analysis, top_k)
            elif complexity < 0.3:
                # Simple query - emphasize semantic similarity
                return await self.rerank(query, documents, query_analysis, top_k)
            else:
                # Standard approach
                return await self.rerank(query, documents, query_analysis, top_k)

        except Exception as e:
            logger.error(f"Error in adaptive reranking: {e}")
            return await self.rerank(query, documents, query_analysis, top_k)

    def get_reranking_stats(self) -> Dict[str, Any]:
        """
        Get reranking configuration and statistics
        """
        return {
            'min_score_threshold': self.min_score_threshold,
            'max_results': self.max_results,
            'diversity_boost': self.diversity_boost,
            'vietnamese_term_weight': self.vietnamese_term_weight,
            'technical_term_weight': self.technical_term_weight,
            'content_type_weights': self.content_type_weights,
            'supported_languages': ['vietnamese', 'english'],
            'reranking_strategies': ['standard', 'adaptive', 'diversity_focused']
        }