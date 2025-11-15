"""
Vietnamese Text Processing Service
Optimized for RAG systems with Vietnamese language support
"""

import re
from typing import List, Dict, Any, Optional, Tuple
import unicodedata
from ..core.exceptions import DocumentProcessingError


class VietnameseTextService:
    """Service for processing Vietnamese text with RAG-optimized transformations"""

    def __init__(self):
        # Vietnamese-specific patterns
        self.vietnamese_chars = set('áàảãấầậấậẫâấầâẽêễễẽễìỉĩîĩìĩỉĩõóỏõóỏõôôôôôơờơờờờôôơờùúúũũũũũưứỳýýỵựưứ')

        # Vietnamese stop words and common terms
        self.vietnamese_stop_words = {
            'và', 'là', 'của', 'cho', 'cái', 'các', 'với', 'như', 'nhưng', 'thì', 'nên',
            'mà', 'có', 'không', 'được', 'trong', 'tại', 'cần', 'sẽ', 'đã', 'bạn', 'mình',
            'chúng tôi', 'bạn', 'anh', 'chị', 'em', 'tôi', 'người', 'điều', 'hơn', 'nhiều',
            'để', 'trên', 'này', 'ngày', 'giờ', 'giờ', 'năm', 'năm', 'nào', 'nào'
        }

        # Vietnamese ML-related keywords
        self.ml_keywords_vn = [
            'học máy', 'trí tuệ nhân tạo', 'mạng nơ-ron', 'học sâu', 'mô hình', 'thuật toán',
            'dữ liệu', 'bộ dữ liệu', 'đào tạo', 'huấn luyện', 'đánh giá', 'chính xác',
            'tối ưu', 'đạo dốc', 'mất mát', 'kiến trình', 'gradient', 'loss', 'hồi quy',
            'andrew ng', 'học máy năm cũ', 'network', 'network sâu', 'neural network',
            'cây quyết định', 'phân loại', 'hồi quy tuyến tính', 'deep learning',
            'machine learning', 'ai', 'trí tuệ nhân tạo', 'thông minh nhân tạo'
        ]

        # Vietnamese sentence end patterns
        self.sentence_endings = r'[.!?]+(?:\s|$)'

        # Vietnamese word segmentation patterns
        self.vietnamese_patterns = [
            # Common Vietnamese syllable patterns
            r'[ảâầấậẫấầâấẩẩẫãêễễễìỉĩỉìĩỉĩõóỏõóỏõôôôôôơờơờờôôôôùúúũũũũũưứỳýýỵựưứ]',
            # Technical terms and numbers
            r'[a-zA-Z0-9]+',
            # Vietnamese special characters and punctuation
            r'[^\w\s]',
            # Single characters
            r'\w',
        ]

    def normalize_vietnamese_text(self, text: str) -> str:
        """
        Normalize Vietnamese text with proper diacritics and spacing
        """
        try:
            # Normalize Unicode characters
            text = unicodedata.normalize('NFC', text)

            # Fix common Vietnamese spacing issues
            text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
            text = text.strip()  # Remove leading/trailing spaces

            # Fix common diacritic combinations
            text = text.replace('òa', 'à').replace('ó', 'ò').replace('ú', 'ù')
            text = text.replace('ỳ', 'ỳ').replace('ý', 'y')

            return text
        except Exception as e:
            raise DocumentProcessingError(f"Vietnamese text normalization failed: {e}")

    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect if text is Vietnamese and return language with confidence score
        Returns: (language, confidence_score)
        """
        try:
            text_lower = text.lower()

            # Count Vietnamese characters
            vietnamese_char_count = sum(1 for char in text if char in self.vietnamese_chars)
            total_chars = len(text.replace(' ', ''))

            if total_chars == 0:
                return ('unknown', 0.0)

            vietnamese_ratio = vietnamese_char_count / total_chars

            # Check for Vietnamese stop words
            stop_word_matches = sum(1 for word in self.vietnamese_stop_words if word in text_lower)

            # Check for Vietnamese ML keywords
            ml_keyword_matches = sum(1 for keyword in self.ml_keywords_vn if keyword in text_lower)

            # Calculate confidence scores
            char_confidence = min(vietnamese_ratio * 2, 1.0)  # Max 1.0
            stop_confidence = min(stop_word_matches / 10, 1.0)  # Assume max 10 relevant stop words
            ml_confidence = min(ml_keyword_matches / 5, 1.0)  # Assume 5 relevant keywords

            # Weighted average
            overall_confidence = (char_confidence * 0.4 + stop_confidence * 0.3 + ml_confidence * 0.3)

            if overall_confidence > 0.3:
                return ('vietnamese', overall_confidence)
            else:
                return ('english', overall_confidence)

        except Exception as e:
            return ('unknown', 0.0)

    def tokenize_vietnamese_words(self, text: str) -> List[str]:
        """
        Tokenize Vietnamese text into words, preserving compound words and technical terms
        """
        try:
            # Basic splitting by whitespace first
            words = text.split()

            # Filter out empty strings and very short tokens
            words = [word.strip() for word in words if word.strip() and len(word.strip()) > 0]

            # Remove Vietnamese stop words
            filtered_words = [word for word in words if word.lower() not in self.vietnamese_stop_words]

            return filtered_words

        except Exception as e:
            raise DocumentProcessingError(f"Vietnamese tokenization failed: {e}")

    def extract_vietnamese_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract Vietnamese ML/AI related keywords from text
        """
        try:
            text_lower = text.lower()
            found_keywords = []

            for keyword in self.ml_keywords_vn:
                if keyword in text_lower and keyword not in found_keywords:
                    found_keywords.append(keyword)
                    if len(found_keywords) >= max_keywords:
                        break

            return found_keywords

        except Exception as e:
            raise DocumentProcessingError(f"Keyword extraction failed: {e}")

    def expand_vietnamese_query(self, query: str) -> List[str]:
        """
        Expand Vietnamese query with synonyms and related terms
        """
        try:
            language, confidence = self.detect_language(query)

            if language != 'vietnamese':
                return [query]

            # Vietnamese synonyms and expansions
            expansions = {
                'học máy': ['học machine learning', 'học trí tuệ nhân tạo', 'trí tuệ máy tính'],
                'ai': ['trí tuệ nhân tạo', 'thông minh nhân tạo', 'ai nhân tạo'],
                'mạng nơ-ron': ['neural network', 'mạng nơ ron', 'mạng thần kinh'],
                'huấn luyện': ['train', 'training model', 'đào tạo mô hình'],
                'dự đoán': ['prediction', 'dự báo', 'dự báo kết quả'],
                'phân loại': ['classification', 'phân loại dữ liệu', 'nhận diện'],
                'tối ưu': ['optimization', 'tối ưu hóa', 'cải thiện hiệu suất'],
                'hồi quy': ['overfitting', 'hồi quy tuyến tính', 'quá khớp'],
                'mô hình': ['model', 'mô hình máy học', 'hệ thống học']
            }

            expanded_queries = [query]

            for base_term, synonyms in expansions.items():
                if base_term in query.lower():
                    expanded_queries.extend(synonyms)

            # Remove duplicates while preserving order
            seen = set()
            unique_expansions = []
            for q in expanded_queries:
                if q not in seen:
                    seen.add(q)
                    unique_expansions.append(q)

            return unique_expansions

        except Exception as e:
            raise DocumentProcessingError(f"Query expansion failed: {e}")

    def process_vietnamese_query(self, query: str) -> Dict[str, Any]:
        """
        Process Vietnamese query for RAG system
        Returns: Dict with processed query, detected language, keywords, etc.
        """
        try:
            # Normalize text
            normalized_query = self.normalize_vietnamese_text(query)

            # Detect language
            language, confidence = self.detect_language(normalized_query)

            # Tokenize
            tokens = self.tokenize_vietnamese_words(normalized_query)

            # Extract keywords
            keywords = self.extract_vietnamese_keywords(normalized_query)

            # Expand query
            expanded_queries = self.expand_vietnamese_query(normalized_query)

            # Calculate query complexity
            complexity_score = min(
                len(normalized_query) / 100,  # Length factor (normalized)
                len(keywords),  # Keyword factor
                1.0  # Maximum
            )

            return {
                'original_query': query,
                'processed_query': normalized_query,
                'detected_language': language,
                'language_confidence': confidence,
                'tokens': tokens,
                'keywords': keywords,
                'expanded_queries': expanded_queries,
                'complexity_score': complexity_score,
                'is_vietnamese': language == 'vietnamese'
            }

        except Exception as e:
            raise DocumentProcessingError(f"Query processing failed: {e}")

    def analyze_vietnamese_content(self, text: str) -> Dict[str, Any]:
        """
        Analyze Vietnamese content for metadata and classification
        """
        try:
            # Basic statistics
            char_count = len(text)
            word_count = len(text.split())

            # Vietnamese content analysis
            language, confidence = self.detect_language(text)

            # Extract Vietnamese keywords
            keywords = self.extract_vietnamese_keywords(text)

            # Estimate difficulty level based on vocabulary complexity
            technical_terms = len([kw for kw in keywords if len(kw.split()) > 1])
            difficulty_level = 'beginner'
            if technical_terms > 3:
                difficulty_level = 'intermediate'
            if technical_terms > 6:
                difficulty_level = 'advanced'

            # Content type detection
            content_type = 'general'
            if any(term in text.lower() for term in ['mã hình', 'thuật toán', 'dữ liệu', 'huấn luyện']):
                content_type = 'technical'
            elif any(term in text.lower() for term in ['ví dụ', 'ví dụ', 'thực tế', 'ứng dụng']):
                content_type = 'practical'

            return {
                'char_count': char_count,
                'word_count': word_count,
                'detected_language': language,
                'language_confidence': confidence,
                'keywords': keywords,
                'technical_term_count': technical_terms,
                'difficulty_level': difficulty_level,
                'content_type': content_type,
                'is_vietnamese': language == 'vietnamese'
            }

        except Exception as e:
            raise DocumentProcessingError(f"Content analysis failed: {e}")

    def create_vietnamese_metadata(self, text: str, chunk_id: str = None, page_number: int = None) -> Dict[str, Any]:
        """
        Create Vietnamese-specific metadata for document chunks
        """
        try:
            analysis = self.analyze_vietnamese_content(text)

            metadata = {
                'chunk_id': chunk_id,
                'page_number': page_number,
                'char_count': analysis['char_count'],
                'word_count': analysis['word_count'],
                'language': analysis['detected_language'],
                'language_confidence': analysis['language_confidence'],
                'keywords': analysis['keywords'],
                'technical_terms_count': analysis['technical_term_count'],
                'difficulty_level': analysis['difficulty_level'],
                'content_type': analysis['content_type'],
                'is_vietnamese': analysis['is_vietnamese'],
                'processing_timestamp': None  # Will be set by caller
            }

            return metadata

        except Exception as e:
            raise DocumentProcessingError(f"Metadata creation failed: {e}")