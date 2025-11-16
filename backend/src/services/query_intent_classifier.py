"""
Intelligent Query Intent Classifier
Determines whether to use RAG, general AI knowledge, or hybrid approach
"""

from typing import Dict, Any, List, Tuple
import re
import logging
from dataclasses import dataclass
from enum import Enum

from .vietnamese_text_service import VietnameseTextService

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Query intent types for intelligent routing"""
    DOCUMENT_SPECIFIC = "document_specific"  # Needs RAG from indexed documents
    GENERAL_KNOWLEDGE = "general_knowledge"   # Can use general AI knowledge
    HYBRID = "hybrid"                        # Benefits from both RAG and general knowledge
    CONVERSATIONAL = "conversational"        # General chat, no specific knowledge needed

@dataclass
class IntentAnalysis:
    """Result of intent classification"""
    intent: QueryIntent
    reasoning: str
    specific_entities: List[str]
    question_type: str
    complexity: str

class QueryIntentClassifier:
    """
    Intelligent classifier that determines the best response strategy
    """

    def __init__(self):
        self.vietnamese_service = VietnameseTextService()

        # ML/AI specific terms that likely have document coverage - expanded
        self.document_specific_terms = {
            'machine learning yearning', 'andrew ng', 'bias and variance', 'end-to-end deep learning',
            'supervised learning', 'unsupervised learning', 'neural network', 'deep learning', 'backpropagation',
            'gradient descent', 'overfitting', 'underfitting', 'training set', 'dev set', 'test set',
            'cross-validation', 'regularization', 'hyperparameter', 'learning rate', 'activation function',
            'loss function', 'optimization', 'feature engineering', 'data preprocessing',
            'book', 'chapter', 'page', 'section', 'according to the text', 'based on the context',
            # Additional ML terms
            'backpropagation', 'gradient descent', 'neural', 'network', 'deep', 'learning',
            'machine learning', 'artificial intelligence', 'ai', 'model', 'algorithm',
            'training', 'dataset', 'epoch', 'batch', 'weight', 'bias', 'layer',
            'convolutional', 'recurrent', 'lstm', 'rnn', 'cnn', 'transformer',
            'attention', 'embedding', 'vector', 'matrix', 'tensor', 'pytorch',
            'tensorflow', 'keras', 'goodfellow', 'bengio', 'lecun',
            # Mathematical terms for ML/DL
            'eigendecomposition', 'eigenvalue', 'eigenvector', 'eigenvalues', 'eigenvectors',
            'singular value decomposition', 'svd', 'matrix decomposition', 'matrix factorization',
            'principal component analysis', 'pca', 'covariance matrix', 'linear algebra',
            'singular values', 'orthogonal', 'matrix multiplication', 'transpose', 'inverse',
            'determinant', 'trace', 'norm', 'vector space', 'linear transformation',
            'quadratic form', 'positive definite', 'symmetric matrix', 'diagonalization',
            # Vietnamese ML terms that should use RAG
            'phép toán tích chập', 'convolution', 'tích chập', 'mạng nơ-ron tích chập', 'cnn',
            'truyền ngược', 'backpropagation', 'lan truyền ngược', 'gradient descent',
            'học sâu', 'deep learning', 'mạng nơ-ron', 'neural network', 'trí tuệ nhân tạo',
            'artificial intelligence', 'học máy', 'machine learning', 'hàm mất mát',
            'loss function', 'hàm kích hoạt', 'activation function', 'tối ưu hóa',
            'optimization', 'siêu tham số', 'hyperparameter', 'tốc độ học',
            'learning rate', 'kiểm tra chéo', 'cross-validation', 'quá khớp',
            'overfitting', 'dưới khớp', 'underfitting', 'tập huấn', 'training',
            'bộ dữ liệu', 'dataset', 'trọng số', 'weights', 'thiên kiến',
            'bias', 'lớp', 'layer', 'goodfellow', 'ian goodfellow',
            # Vietnamese mathematical terms
            'phân rã riêng', 'giá trị riêng', 'vector riêng', 'giá trị riêng vector riêng',
            'phân rã giá trị riêng lẻ', 'phân rã ma trận', 'phân nhân ma trận',
            'phân tích thành phần chính', 'pca', 'ma trận hiệp phương sai',
            'đại số tuyến tính', 'không gian vector', 'biến đổi tuyến tính',
            'ma trận đối xứng', 'ma trận xác định dương', 'chéo hóa',
            'định thức', 'vết', 'chuẩn', 'nhiễu', 'ma trận nghịch đảo'
        }

        # General ML/AI concepts that can use general knowledge
        self.general_knowledge_terms = {
            'what is', 'define', 'explain', 'introduction', 'overview', 'basics',
            'history', 'applications', 'examples', 'comparison', 'difference',
            'advantages', 'disadvantages', 'pros', 'cons', 'when to use', 'how to choose'
        }

        # Conversational patterns
        self.conversational_patterns = [
            r'^(hello|hi|chào|xin chào)',
            r'^(thank you|thanks|cảm ơn)',
            r'^(how are you|bạn khỏe không)',
            r'^(goodbye|bye|tạm biệt)',
            r'^(help|giúp đỡ)',
            r'^(what can you do|bạn có thể làm gì)'
        ]

    def classify_intent(self, query: str, query_analysis: Dict[str, Any]) -> IntentAnalysis:
        """
        Classify query intent to determine optimal response strategy
        """
        try:
            normalized_query = query.lower().strip()

            # Check for conversational patterns first
            conversational_score = self._check_conversational_intent(normalized_query)
            if conversational_score > 0.7:
                return IntentAnalysis(
                    intent=QueryIntent.CONVERSATIONAL,
                    reasoning="Query matches conversational patterns",
                    specific_entities=[],
                    question_type="conversational",
                    complexity="low"
                )

            # Analyze for document-specific content
            document_score, doc_entities = self._analyze_document_specific(normalized_query)

            # Analyze for general knowledge suitability
            general_score = self._analyze_general_knowledge_suitability(normalized_query)

            # Get complexity from existing analysis
            complexity = query_analysis.get('complexity', {}).get('complexity_score', 0.5)
            if complexity > 0.7:
                complexity_level = "high"
            elif complexity < 0.3:
                complexity_level = "low"
            else:
                complexity_level = "medium"

            # Enhanced decision logic without confidence calculations
            if document_score > 0.2:
                # Clear document-specific case - very sensitive threshold
                intent = QueryIntent.DOCUMENT_SPECIFIC
                reasoning = f"Query matches document-specific content ({document_score:.2f})"
            elif general_score > 0.15:
                # Clear general knowledge case - very sensitive threshold
                intent = QueryIntent.GENERAL_KNOWLEDGE
                reasoning = f"Query is well-suited for general knowledge ({general_score:.2f})"
            elif document_score > 0.1 and general_score > 0.1:
                # Strong hybrid case - both RAG and general knowledge beneficial
                intent = QueryIntent.HYBRID
                reasoning = f"Query benefits from both document content ({document_score:.2f}) and general knowledge ({general_score:.2f})"
            else:
                # Only default to hybrid for truly ambiguous cases
                intent = QueryIntent.HYBRID
                reasoning = "Query is ambiguous, using hybrid approach"

            # Determine question type
            question_type = self._determine_question_type(normalized_query)

            return IntentAnalysis(
                intent=intent,
                reasoning=reasoning,
                specific_entities=doc_entities,
                question_type=question_type,
                complexity=complexity_level
            )

        except Exception as e:
            logger.error(f"Error classifying query intent: {e}")
            # Default to hybrid on error
            return IntentAnalysis(
                intent=QueryIntent.HYBRID,
                reasoning="Classification failed, using hybrid fallback",
                specific_entities=[],
                question_type="unknown",
                complexity="medium"
            )

    def _check_conversational_intent(self, query: str) -> float:
        """Check if query is conversational"""
        score = 0.0
        for pattern in self.conversational_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                score = max(score, 0.8)
        return score

    def _analyze_document_specific(self, query: str) -> Tuple[float, List[str]]:
        """Analyze if query requires document-specific information"""
        score = 0.0
        found_entities = []

        # Check for document-specific terms
        for term in self.document_specific_terms:
            if term in query:
                score += 0.3
                found_entities.append(term)

        # Check for direct references to the book/content
        book_references = ['machine learning yearning', 'andrew ng', 'the book', 'the text',
                          'the context', 'according to', 'based on', 'in the chapter', 'page']
        for ref in book_references:
            if ref in query:
                score += 0.4
                found_entities.append(ref)

        # Check for specific page/section references
        if re.search(r'page \d+|chapter \d+|section \d+', query):
            score += 0.5
            found_entities.append("page_reference")

        # Vietnamese specific indicators
        vietnamese_refs = ['trang', 'chương', 'phần', 'mục', 'sách', 'tài liệu', 'ngữ cảnh']
        for ref in vietnamese_refs:
            if ref in query:
                score += 0.4
                found_entities.append(ref)

        return min(score, 1.0), found_entities

    def _analyze_general_knowledge_suitability(self, query: str) -> float:
        """Analyze if query can be answered with general AI knowledge"""
        score = 0.0

        # Check for general knowledge patterns
        for term in self.general_knowledge_terms:
            if term in query:
                score += 0.2

        # Fundamental ML concepts that are well-known
        fundamental_concepts = [
            'what is', 'define', 'explain', 'introduction', 'how does', 'why is',
            'advantages', 'disadvantages', 'comparison', 'difference', 'when to use'
        ]

        for concept in fundamental_concepts:
            if concept in query:
                score += 0.3

        # Vietnamese equivalents
        vietnamese_general = [
            'là gì', 'định nghĩa', 'giải thích', 'giới thiệu', 'hoạt động như thế nào',
            'tại sao', 'ưu điểm', 'nhược điểm', 'so sánh', 'khác biệt', 'khi nào'
        ]

        for term in vietnamese_general:
            if term in query:
                score += 0.3

        return min(score, 1.0)

    def _determine_question_type(self, query: str) -> str:
        """Determine the type of question being asked"""
        if any(word in query for word in ['what', 'what is', 'là gì', 'định nghĩa']):
            return "definition"
        elif any(word in query for word in ['how', 'how to', 'làm thế nào', 'cách']):
            return "how_to"
        elif any(word in query for word in ['why', 'why is', 'tại sao']):
            return "explanation"
        elif any(word in query for word in ['compare', 'difference', 'khác biệt', 'so sánh']):
            return "comparison"
        elif any(word in query for word in ['advantages', 'disadvantages', 'pros', 'cons', 'ưu điểm', 'nhược điểm']):
            return "evaluation"
        elif any(word in query for word in ['examples', 'example', 'ví dụ']):
            return "examples"
        else:
            return "general"

    def should_use_rag(self, intent_analysis: IntentAnalysis, context_available: bool = True) -> bool:
        """
        Determine if RAG should be used based on intent and context availability
        """
        if not context_available:
            return False

        return intent_analysis.intent in [
            QueryIntent.DOCUMENT_SPECIFIC,
            QueryIntent.HYBRID
        ]

    def should_use_general_knowledge(self, intent_analysis: IntentAnalysis) -> bool:
        """
        Determine if general knowledge should be used
        """
        return intent_analysis.intent in [
            QueryIntent.GENERAL_KNOWLEDGE,
            QueryIntent.HYBRID
        ]

    def get_response_strategy(self, intent_analysis: IntentAnalysis, context_quality: float = 0.0) -> Dict[str, Any]:
        """
        Get recommended response strategy based on intent analysis
        """
        strategy = {
            'use_rag': False,
            'use_general_knowledge': False,
            'primary_source': 'none',
            'fallback_enabled': True,
            'citation_required': False
        }

        if intent_analysis.intent == QueryIntent.DOCUMENT_SPECIFIC:
            strategy['use_rag'] = True
            strategy['primary_source'] = 'documents'
            strategy['citation_required'] = True
        elif intent_analysis.intent == QueryIntent.GENERAL_KNOWLEDGE:
            strategy['use_general_knowledge'] = True
            strategy['primary_source'] = 'ai_knowledge'
        elif intent_analysis.intent == QueryIntent.HYBRID:
            # Always use both RAG and general knowledge for hybrid queries
            strategy['use_rag'] = True
            strategy['use_general_knowledge'] = True
            strategy['primary_source'] = 'hybrid'
            strategy['citation_required'] = True
        elif intent_analysis.intent == QueryIntent.CONVERSATIONAL:
            strategy['use_general_knowledge'] = True
            strategy['primary_source'] = 'conversational'
            strategy['citation_required'] = False

        return strategy