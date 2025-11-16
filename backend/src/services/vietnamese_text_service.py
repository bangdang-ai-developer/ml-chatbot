"""
Vietnamese Text Processing Service
Advanced text processing for Vietnamese language in ML/AI context
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from ..core.exceptions import ChatbotException

logger = logging.getLogger(__name__)

class VietnameseTextService:
    """
    Service for processing Vietnamese text in ML/AI context
    """

    def __init__(self):
        # Vietnamese technical terms and their English equivalents
        self.technical_terms = {
            'học máy': 'machine learning',
            'học sâu': 'deep learning',
            'trí tuệ nhân tạo': 'artificial intelligence',
            'mạng nơ-ron': 'neural network',
            'tích chập': 'convolution',
            'củng cố': 'reinforcement',
            'giám sát': 'supervised',
            'không giám sát': 'unsupervised',
            'truyền ngược': 'backpropagation',
            'tối ưu': 'optimization',
            'hàm mất mát': 'loss function',
            'độ chính xác': 'accuracy',
            'kiểm chứng chéo': 'cross validation',
            'quá trình học': 'training process',
            'dữ liệu huấn luyện': 'training data',
            'dữ liệu kiểm tra': 'test data',
            'dữ liệu xác thực': 'validation data'
        }

        # Vietnamese stopwords for filtering
        self.stopwords = {
            'và', 'là', 'của', 'trong', 'cho', 'với', 'để', 'từ', 'bởi',
            'mà', 'nên', 'thì', 'khi', 'nếu', 'nhưng', 'tuy nhiên', 'vì',
            'có', 'không', 'được', 'làm', 'đi', 'này', 'kia', 'đó', 'một',
            'hai', 'ba', 'nhiều', 'mấy', 'mọi', 'các', 'những', 'cả'
        }

    def detect_vietnamese(self, text: str) -> bool:
        """
        Detect if text is Vietnamese
        """
        try:
            # Check for Vietnamese characters
            vietnamese_chars = set('áàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ')
            text_chars = set(text.lower())

            # Count Vietnamese characters
            viet_count = len(text_chars & vietnamese_chars)
            total_chars = len([c for c in text if c.isalpha()])

            # Consider Vietnamese if >10% characters are Vietnamese
            if total_chars > 0:
                return (viet_count / total_chars) > 0.1

            return False

        except Exception as e:
            logger.error(f"Error detecting Vietnamese: {e}")
            return False

    def normalize_vietnamese_text(self, text: str) -> str:
        """
        Normalize Vietnamese text for processing
        """
        try:
            if not text:
                return ""

            # Convert to lowercase
            text = text.lower().strip()

            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)

            # Normalize some common Vietnamese text patterns
            text = re.sub(r'(?<=[a-z])\s*-\s*', '-', text)  # Fix word breaks

            return text

        except Exception as e:
            logger.error(f"Error normalizing Vietnamese text: {e}")
            return text

    def extract_technical_terms(self, text: str) -> List[str]:
        """
        Extract Vietnamese technical terms from text
        """
        try:
            found_terms = []
            normalized_text = self.normalize_vietnamese_text(text)

            for viet_term, eng_term in self.technical_terms.items():
                if viet_term in normalized_text:
                    found_terms.append({
                        'vietnamese': viet_term,
                        'english': eng_term,
                        'context': self._get_context(normalized_text, viet_term)
                    })

            return found_terms

        except Exception as e:
            logger.error(f"Error extracting technical terms: {e}")
            return []

    def _get_context(self, text: str, term: str, window: int = 20) -> str:
        """
        Get context around a technical term
        """
        try:
            index = text.find(term)
            if index == -1:
                return ""

            start = max(0, index - window)
            end = min(len(text), index + len(term) + window)

            return text[start:end].strip()

        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return ""

    def translate_technical_terms(self, text: str) -> str:
        """
        Replace Vietnamese technical terms with English equivalents
        """
        try:
            translated_text = text

            # Sort terms by length (longest first) to avoid partial matches
            sorted_terms = sorted(self.technical_terms.items(),
                               key=lambda x: len(x[0]), reverse=True)

            for viet_term, eng_term in sorted_terms:
                # Replace with word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(viet_term) + r'\b'
                translated_text = re.sub(pattern, eng_term, translated_text, flags=re.IGNORECASE)

            return translated_text

        except Exception as e:
            logger.error(f"Error translating technical terms: {e}")
            return text

    def tokenize_vietnamese(self, text: str) -> List[str]:
        """
        Simple Vietnamese tokenization
        """
        try:
            # Basic Vietnamese word segmentation
            # This is a simplified version - in production, you'd use a proper tokenizer like underthesea
            text = self.normalize_vietnamese_text(text)

            # Remove punctuation and split
            words = re.findall(r'\b\w+\b', text)

            # Filter stopwords
            words = [word for word in words if word not in self.stopwords and len(word) > 1]

            return words

        except Exception as e:
            logger.error(f"Error tokenizing Vietnamese: {e}")
            return []

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two Vietnamese texts
        """
        try:
            words1 = set(self.tokenize_vietnamese(text1))
            words2 = set(self.tokenize_vietnamese(text2))

            if not words1 or not words2:
                return 0.0

            # Jaccard similarity
            intersection = len(words1 & words2)
            union = len(words1 | words2)

            return intersection / union if union > 0 else 0.0

        except Exception as e:
            logger.error(f"Error calculating text similarity: {e}")
            return 0.0

    def extract_key_phrases(self, text: str) -> List[str]:
        """
        Extract key phrases from Vietnamese text
        """
        try:
            words = self.tokenize_vietnamese(text)

            # Look for common ML/AI technical patterns
            key_phrases = []

            # Check for technical terms
            technical_terms_found = self.extract_technical_terms(text)
            key_phrases.extend([term['vietnamese'] for term in technical_terms_found])

            # Look for question patterns
            question_patterns = [
                'cách hoạt động', 'làm thế nào', 'giải thích', 'định nghĩa',
                'so sánh', 'khác biệt', 'ưu điểm', 'nhược điểm', 'ví dụ'
            ]

            for pattern in question_patterns:
                if pattern in text.lower():
                    key_phrases.append(pattern)

            return list(set(key_phrases))  # Remove duplicates

        except Exception as e:
            logger.error(f"Error extracting key phrases: {e}")
            return []

    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze intent of Vietnamese query
        """
        try:
            normalized_query = self.normalize_vietnamese_text(query)
            key_phrases = self.extract_key_phrases(query)
            technical_terms = self.extract_technical_terms(query)

            # Determine intent based on patterns
            intent = 'general'
            confidence = 0.5

            if any(phrase in normalized_query for phrase in ['định nghĩa', 'là gì', 'giải thích']):
                intent = 'definition'
                confidence = 0.8
            elif any(phrase in normalized_query for phrase in ['so sánh', 'khác biệt', 'so với']):
                intent = 'comparison'
                confidence = 0.8
            elif any(phrase in normalized_query for phrase in ['cách', 'làm thế nào', 'hướng dẫn']):
                intent = 'how_to'
                confidence = 0.7
            elif any(phrase in normalized_query for phrase in ['ví dụ', 'minh họa']):
                intent = 'examples'
                confidence = 0.7
            elif any(phrase in normalized_query for phrase in ['ưu điểm', 'nhược điểm', 'tốt', 'kém']):
                intent = 'advantages_disadvantages'
                confidence = 0.6

            return {
                'intent': intent,
                'confidence': confidence,
                'key_phrases': key_phrases,
                'technical_terms': technical_terms,
                'is_technical': len(technical_terms) > 0,
                'complexity': self._assess_complexity(normalized_query, technical_terms)
            }

        except Exception as e:
            logger.error(f"Error analyzing query intent: {e}")
            return {
                'intent': 'general',
                'confidence': 0.0,
                'key_phrases': [],
                'technical_terms': [],
                'is_technical': False,
                'complexity': 0.5
            }

    def _assess_complexity(self, text: str, technical_terms: List[Dict]) -> float:
        """
        Assess complexity of Vietnamese text
        """
        try:
            complexity = 0.3  # Base complexity

            # Add complexity for technical terms
            complexity += len(technical_terms) * 0.1

            # Add complexity for long sentences
            sentences = text.split('.')
            if sentences:
                avg_sentence_length = sum(len(s.strip().split()) for s in sentences) / len(sentences)
                if avg_sentence_length > 15:
                    complexity += 0.2

            # Add complexity for specific complex indicators
            complexity_indicators = ['tương lai', 'nghiên cứu', 'phát triển', 'nâng cao', 'tối ưu']
            if any(indicator in text for indicator in complexity_indicators):
                complexity += 0.1

            return min(complexity, 1.0)  # Cap at 1.0

        except Exception as e:
            logger.error(f"Error assessing complexity: {e}")
            return 0.5

    def generate_response_template(self, intent: str, is_technical: bool = False) -> str:
        """
        Generate appropriate response template for Vietnamese
        """
        templates = {
            'definition': "À, để mình giải thích [concept] một cách đơn giản nhé...",
            'comparison': "Ôi, đây là câu hỏi hay lắm! Mình phân tích giúp bạn nhé...",
            'how_to': "Được chứ! Mình hướng dẫn bạn từng bước nhé...",
            'examples': "À có nhiều ví dụ thú vị lắm! Mình kể bạn nghe nhé...",
            'advantages_disadvantages': "Đây là câu hỏi thực tế! Mình phân tích kỹ giúp bạn...",
            'general': "Mình hiểu câu hỏi của bạn rồi. Để mình chia sẻ nhé..."
        }

        base_template = templates.get(intent, templates['general'])

        if is_technical:
            base_template += "\n\nĐây là chủ đề kỹ thuật nên mình sẽ giải thích chi tiết hơn nhé."

        return base_template

    def create_vietnamese_metadata(self, content: str, chunk_id: str = None, page_number: int = None) -> Dict[str, Any]:
        """
        Create Vietnamese metadata for document chunks
        """
        try:
            # Process text for RAG
            processed_text = self.process_text_for_rag(content)

            # Create enhanced metadata
            metadata = {
                'chunk_id': chunk_id,
                'page_number': page_number,
                'is_vietnamese_content': processed_text['is_vietnamese'],
                'language_detected': 'vietnamese' if processed_text['is_vietnamese'] else 'english',
                'normalized_text': processed_text['normalized_text'],
                'key_phrases': processed_text['key_phrases'],
                'technical_terms_found': processed_text['technical_terms'],
                'intent_analysis': processed_text['intent_analysis'],
                'translated_technical_terms': processed_text['translated_text'],
                'token_count': len(processed_text['tokens']),
                'complexity_score': processed_text['intent_analysis']['complexity'],
                'is_technical_content': processed_text['intent_analysis']['is_technical'],
                'content_quality': self._assess_content_quality(content, processed_text)
            }

            logger.debug(f"Created Vietnamese metadata for chunk {chunk_id}: language={metadata['language_detected']}, technical={metadata['is_technical_content']}")
            return metadata

        except Exception as e:
            logger.error(f"Error creating Vietnamese metadata: {e}")
            # Return basic metadata on error
            return {
                'chunk_id': chunk_id,
                'page_number': page_number,
                'is_vietnamese_content': False,
                'language_detected': 'english',
                'error': str(e)
            }

    def _assess_content_quality(self, content: str, processed_text: Dict) -> float:
        """
        Assess quality of content for RAG purposes
        """
        try:
            quality = 0.5  # Base quality

            # Add quality for substantial content
            if len(content) > 100:
                quality += 0.1
            if len(content) > 500:
                quality += 0.1

            # Add quality for technical content
            if processed_text['intent_analysis']['is_technical']:
                quality += 0.2

            # Add quality for clear structure
            sentences = content.split('.')
            if 3 <= len(sentences) <= 10:
                quality += 0.1

            return min(quality, 1.0)

        except Exception as e:
            logger.error(f"Error assessing content quality: {e}")
            return 0.5

    def process_text_for_rag(self, text: str) -> Dict[str, Any]:
        """
        Process Vietnamese text for RAG system
        """
        try:
            return {
                'original_text': text,
                'is_vietnamese': self.detect_vietnamese(text),
                'normalized_text': self.normalize_vietnamese_text(text),
                'tokens': self.tokenize_vietnamese(text),
                'technical_terms': self.extract_technical_terms(text),
                'translated_text': self.translate_technical_terms(text),
                'intent_analysis': self.analyze_query_intent(text),
                'key_phrases': self.extract_key_phrases(text)
            }
        except Exception as e:
            logger.error(f"Error processing text for RAG: {e}")
            raise ChatbotException(f"Text processing failed: {e}")