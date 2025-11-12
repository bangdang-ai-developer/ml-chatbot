"""
Query Understanding and Expansion Service for Vietnamese
Advanced RAG query processing with semantic expansion and intent detection
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import logging
from .vietnamese_text_service import VietnameseTextService
from ..core.exceptions import ChatbotException

logger = logging.getLogger(__name__)

class QueryExpansionService:
    """
    Advanced query understanding and expansion service for Vietnamese RAG systems
    """

    def __init__(self, vietnamese_service: VietnameseTextService = None):
        self.vietnamese_service = vietnamese_service or VietnameseTextService()

        # Vietnamese ML/AI concept hierarchies
        self.concept_hierarchy = {
            'học máy': {
                'synonyms': ['machine learning', 'học máy có giám sát', 'học không giám sát'],
                'related_concepts': ['thuật toán', 'mô hình', 'dữ liệu', 'huấn luyện', 'đánh giá'],
                'subtopics': ['phân loại', 'hồi quy', 'cụm', 'giảm chiều dữ liệu'],
                'examples': ['decision tree', 'random forest', 'svm', 'neural networks']
            },
            'mạng nơ-ron': {
                'synonyms': ['neural network', 'mạng thần kinh', 'deep learning', 'học sâu'],
                'related_concepts': ['tầng', 'trọng số', 'hàm kích hoạt', 'lan truyền ngược'],
                'subtopics': ['cnn', 'rnn', 'lstm', 'transformer', 'attention'],
                'examples': ['tensorflow', 'pytorch', 'keras', 'caffe']
            },
            'trí tuệ nhân tạo': {
                'synonyms': ['artificial intelligence', 'AI', 'thông minh nhân tạo'],
                'related_concepts': ['tư duy', 'học tập', 'lý luận', 'giải quyết vấn đề'],
                'subtopics': ['machine learning', 'natural language processing', 'computer vision', 'robotics'],
                'examples': ['chatbot', 'hiểu ngôn ngữ tự nhiên', 'xử lý ảnh', 'game AI']
            },
            'dữ liệu': {
                'synonyms': ['data', 'dataset', 'bộ dữ liệu', 'cơ sở dữ liệu'],
                'related_concepts': ['thu thập', 'làm sạch', 'phân tích', 'trực quan hóa'],
                'subtopics': ['big data', 'data mining', 'data preprocessing', 'feature engineering'],
                'examples': ['csv', 'json', 'database', 'data warehouse']
            },
            'thuật toán': {
                'synonyms': ['algorithm', 'giải thuật', 'phương pháp'],
                'related_concepts': ['tối ưu', 'độ phức tạp', 'hiệu suất', 'cải tiến'],
                'subtopics': ['sorting', 'searching', 'optimization', 'machine learning algorithms'],
                'examples': ['gradient descent', 'backpropagation', 'k-means', 'pca']
            }
        }

        # Vietnamese question patterns and intents
        self.question_patterns = {
            'definition': [
                r'^.*\s+là\s+gì\s*$',  # ... là gì?
                r'^.*\s+nghĩa\s+là\s+.*$',  # ... nghĩa là...
                r'^.*\s+định\s+nghĩa\s+.*$',  # ... định nghĩa...
                r'^giải\s+thích\s+về\s+.*$',  # giải thích về...
            ],
            'comparison': [
                r'^.*\s+và\s+.*\s+khác\s+nhau\s+như\s+thế\s+nào\s*$',  # ... và ... khác nhau như thế nào?
                r'^so\s+sánh\s+.*\s+với\s+.*$',  # so sánh ... với ...
                r'^.*\s+tốt\s+hơn\s+.*\s+không\s*$',  # ... tốt hơn ... không?
            ],
            'how_to': [
                r'^làm\s+thế\s+nào\s+để\s+.*$',  # làm thế nào để ...
                r'^cách\s+.*$',  # cách ...
                r'^hướng\s+dẫn\s+.*$',  # hướng dẫn ...
                r'^các\s+bước\s+.*$',  # các bước ...
            ],
            'why': [
                r'^tại\s+sao\s+.*$',  # tại sao ...
                r'^vì\s+sao\s+.*$',  # vì sao ...
                r'^lý\s+do\s+.*$',  # lý do ...
                r'^nguyên\s+nhân\s+.*$',  # nguyên nhân ...
            ],
            'examples': [
                r'^ví\s+dụ\s+về\s+.*$',  # ví dụ về ...
                r'^.*\s+ví\s+dụ\s+.*$',  # ... ví dụ ...
                r'^cho\s+một\s+ví\s+dụ\s+.*$',  # cho một ví dụ ...
                r'^một\s+số\s+ví\s+dụ\s+.*$',  # một số ví dụ ...
            ],
            'advantages_disadvantages': [
                r'^ưu\s+điểm\s+của\s+.*$',  # ưu điểm của ...
                r'^nhược\s+điểm\s+của\s+.*$',  # nhược điểm của ...
                r'^.*\s+có\slợi\s+ích\s+gì\s*$',  # ... có lợi ích gì?
                r'^.*\s+có\shạn\s+chế\s+gì\s*$',  # ... có hạn chế gì?
            ]
        }

        # Common Vietnamese query transformations
        self.query_transformations = {
            'formal_to_informal': {
                'thuật toán': 'cách làm',
                'tối ưu hóa': 'cải tiến',
                'đánh giá': 'kiểm tra',
                'triển khai': 'áp dụng',
                'phân tích': 'xem xét'
            },
            'technical_to_simple': {
                'mạng nơ-ron': 'mạng học tập',
                'học sâu': 'học nâng cao',
                'học không giám sát': 'tự tìm hiểu',
                'lan truyền ngược': 'cập nhật ngược'
            }
        }

    def detect_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Detect the intent and type of Vietnamese query
        """
        try:
            query_lower = query.lower().strip()
            detected_intents = []

            # Check against question patterns
            for intent, patterns in self.question_patterns.items():
                for pattern in patterns:
                    if re.match(pattern, query_lower):
                        detected_intents.append(intent)
                        break

            # Determine primary intent
            primary_intent = detected_intents[0] if detected_intents else 'general'

            # Extract key entities
            entities = self._extract_entities(query)

            return {
                'primary_intent': primary_intent,
                'all_intents': detected_intents,
                'entities': entities,
                'is_question': bool(detected_intents),
                'question_type': primary_intent if detected_intents else None
            }

        except Exception as e:
            logger.error(f"Error detecting query intent: {e}")
            return {
                'primary_intent': 'general',
                'all_intents': [],
                'entities': [],
                'is_question': False,
                'question_type': None
            }

    def _extract_entities(self, query: str) -> List[str]:
        """
        Extract ML/AI related entities from Vietnamese query
        """
        entities = []
        query_lower = query.lower()

        # Check for concepts in hierarchy
        for concept in self.concept_hierarchy.keys():
            if concept in query_lower:
                entities.append(concept)

        # Check for subtopics
        for concept_data in self.concept_hierarchy.values():
            for subtopic in concept_data['subtopics']:
                if subtopic in query_lower:
                    entities.append(subtopic)

        return list(set(entities))  # Remove duplicates

    def expand_query_semantically(self, query: str, max_expansions: int = 5) -> List[str]:
        """
        Expand Vietnamese query with semantically related terms
        """
        try:
            expansions = [query]  # Include original query
            query_lower = query.lower()

            # Process entities found in query
            entities = self._extract_entities(query)

            for entity in entities:
                if entity in self.concept_hierarchy:
                    concept_data = self.concept_hierarchy[entity]

                    # Add synonyms
                    for synonym in concept_data['synonyms']:
                        expanded_query = query_lower.replace(entity, synonym)
                        if expanded_query not in [e.lower() for e in expansions]:
                            expansions.append(expanded_query.title())

                    # Add related concepts
                    for related in concept_data['related_concepts']:
                        related_query = f"{query} {related}"
                        if related_query not in expansions:
                            expansions.append(related_query)

                    if len(expansions) >= max_expansions:
                        break

            return expansions[:max_expansions]

        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return [query]

    def generate_hyphen_queries(self, query: str) -> List[str]:
        """
        Generate queries with different Vietnamese term variations
        """
        try:
            variations = [query]

            # Apply transformations
            for transform_type, mappings in self.query_transformations.items():
                for formal, informal in mappings.items():
                    if formal in query:
                        variation = query.replace(formal, informal)
                        variations.append(variation)

            # Generate acronym variations (common in Vietnamese)
            acronym_variations = self._generate_acronym_variations(query)
            variations.extend(acronym_variations)

            return list(set(variations))  # Remove duplicates

        except Exception as e:
            logger.error(f"Error generating query variations: {e}")
            return [query]

    def _generate_acronym_variations(self, query: str) -> List[str]:
        """
        Generate variations using common Vietnamese ML acronyms
        """
        acronyms = {
            'trí tuệ nhân tạo': 'AI',
            'học máy': 'ML',
            'học sâu': 'DL',
            'xử lý ngôn ngữ tự nhiên': 'NLP',
            'trực quan hóa dữ liệu': 'DV',
            'khoa học dữ liệu': 'DS'
        }

        variations = []
        query_lower = query.lower()

        for full_form, acronym in acronyms.items():
            if full_form in query_lower:
                # Replace full form with acronym
                variation = query_lower.replace(full_form, acronym)
                variations.append(variation.upper())

                # Add both acronym and full form
                both_form = query_lower.replace(full_form, f"{full_form} ({acronym})")
                variations.append(both_form.title())

        return variations

    def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """
        Analyze Vietnamese query complexity for adaptive retrieval
        """
        try:
            # Basic metrics
            word_count = len(query.split())
            char_count = len(query)
            entity_count = len(self._extract_entities(query))

            # Vietnamese-specific complexity factors
            has_technical_terms = any(term in query.lower() for term in
                                    ['thuật toán', 'mô hình', 'tối ưu', 'phân tích', 'đánh giá'])
            has_mathematical_content = any(char in query for char in ['=', '+', '-', '*', '/', '∑'])
            has_english_terms = bool(re.search(r'[a-zA-Z]{3,}', query))

            # Calculate complexity score
            complexity_score = min(1.0, (
                word_count * 0.1 +           # Length factor
                entity_count * 0.2 +         # Entity density
                (1 if has_technical_terms else 0) * 0.3 +  # Technical content
                (1 if has_mathematical_content else 0) * 0.2 +  # Mathematical content
                (1 if has_english_terms else 0) * 0.1  # Mixed language
            ))

            # Determine retrieval strategy
            if complexity_score < 0.3:
                strategy = 'simple_semantic'
            elif complexity_score < 0.6:
                strategy = 'hybrid_search'
            else:
                strategy = 'advanced_reranking'

            return {
                'complexity_score': complexity_score,
                'word_count': word_count,
                'char_count': char_count,
                'entity_count': entity_count,
                'has_technical_terms': has_technical_terms,
                'has_mathematical_content': has_mathematical_content,
                'has_english_terms': has_english_terms,
                'recommended_strategy': strategy
            }

        except Exception as e:
            logger.error(f"Error analyzing query complexity: {e}")
            return {
                'complexity_score': 0.5,
                'recommended_strategy': 'hybrid_search'
            }

    def process_query_comprehensive(self, query: str) -> Dict[str, Any]:
        """
        Comprehensive Vietnamese query processing for RAG
        """
        try:
            # Detect intent
            intent_analysis = self.detect_query_intent(query)

            # Language detection and preprocessing
            vietnamese_analysis = self.vietnamese_service.process_vietnamese_query(query)

            # Semantic expansion
            semantic_expansions = self.expand_query_semantically(query)

            # Query variations
            query_variations = self.generate_hyphen_queries(query)

            # Complexity analysis
            complexity_analysis = self.analyze_query_complexity(query)

            # Combine all analyses
            comprehensive_analysis = {
                'original_query': query,
                'processed_query': vietnamese_analysis['processed_query'],
                'intent_analysis': intent_analysis,
                'language_analysis': {
                    'detected_language': vietnamese_analysis['detected_language'],
                    'language_confidence': vietnamese_analysis['language_confidence'],
                    'is_vietnamese': vietnamese_analysis['is_vietnamese']
                },
                'expansions': {
                    'semantic': semantic_expansions,
                    'variations': query_variations,
                    'all_queries': list(set(semantic_expansions + query_variations))
                },
                'entities': vietnamese_analysis['keywords'],
                'complexity': complexity_analysis,
                'retrieval_strategy': complexity_analysis['recommended_strategy']
            }

            return comprehensive_analysis

        except Exception as e:
            logger.error(f"Error in comprehensive query processing: {e}")
            raise ChatbotException(f"Query processing failed: {e}")

    def get_contextual_hints(self, query_analysis: Dict[str, Any]) -> List[str]:
        """
        Generate contextual hints for better Vietnamese response generation
        """
        hints = []

        try:
            # Language hints
            if query_analysis['language_analysis']['is_vietnamese']:
                hints.append("Respond in Vietnamese language")
                hints.append("Use Vietnamese technical terminology where appropriate")
            else:
                hints.append("Respond in English with Vietnamese examples where helpful")

            # Intent-based hints
            intent = query_analysis['intent_analysis']['primary_intent']
            if intent == 'definition':
                hints.append("Provide clear, concise definitions")
                hints.append("Include Vietnamese explanations of technical terms")
            elif intent == 'examples':
                hints.append("Provide practical, Vietnamese-context examples")
                hints.append("Use Vietnamese ML scenarios when possible")
            elif intent == 'how_to':
                hints.append("Provide step-by-step instructions")
                hints.append("Use Vietnamese instructional language")
            elif intent == 'comparison':
                hints.append("Create comparative tables with Vietnamese labels")
                hints.append("Highlight key differences in Vietnamese terms")

            # Complexity-based hints
            complexity = query_analysis['complexity']['complexity_score']
            if complexity > 0.7:
                hints.append("This is a complex technical query - provide detailed explanations")
                hints.append("Include both theoretical and practical aspects")
            elif complexity < 0.3:
                hints.append("This is a simple query - provide clear, direct answers")

            # Entity-based hints
            entities = query_analysis['entities']
            if 'mạng nơ-ron' in entities or 'deep learning' in entities:
                hints.append("Include recent deep learning developments")
            if 'trí tuệ nhân tạo' in entities or 'AI' in entities:
                hints.append("Mention current AI trends and applications")

            return hints

        except Exception as e:
            logger.error(f"Error generating contextual hints: {e}")
            return ["Provide helpful, accurate response"]