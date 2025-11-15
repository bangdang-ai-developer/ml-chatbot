"""
Dynamic Translation Service using Gemini AI
Provides high-quality contextual translations for Vietnamese RAG system
"""

from typing import Dict, List, Any, Optional, Tuple
import asyncio
import logging
from datetime import datetime
from collections import OrderedDict
import re
from tenacity import retry, stop_after_attempt, wait_exponential

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..core.config import settings
from ..core.exceptions import ChatbotException

logger = logging.getLogger(__name__)

class LRUCache:
    """Simple LRU cache implementation for translation caching"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()

    def get(self, key: str) -> Optional[str]:
        """Get item from cache"""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: str) -> None:
        """Put item in cache"""
        if key in self.cache:
            # Update existing
            self.cache.move_to_end(key)
            self.cache[key] = value
        else:
            # Add new
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
            self.cache[key] = value

    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()

    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)

class TranslationService:
    """
    Dynamic translation service using Gemini AI with contextual understanding
    and caching optimization for Vietnamese RAG systems
    """

    def __init__(self):
        """Initialize translation service with Gemini AI and caching"""
        try:
            # Configure Google AI
            genai.configure(api_key=settings.google_api_key)

            # Initialize translation-optimized LLM
            self.translation_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.1,  # Low temperature for consistent translations
                max_tokens=1024,  # Reasonable limit for translations
                timeout=15.0,     # Shorter timeout for translations
                google_api_key=settings.google_api_key
            )

            # Initialize fallback translation LLM
            self.fallback_llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.2,
                max_tokens=512,
                timeout=10.0,
                google_api_key=settings.google_api_key
            )

            # Create specialized translation prompts
            self._create_translation_prompts()

            # Initialize output parser
            self.output_parser = StrOutputParser()

            # Initialize caching
            self.translation_cache = LRUCache(max_size=1000)
            self.batch_translation_cache = LRUCache(max_size=500)

            # Initialize fallback libraries
            self._initialize_fallback_translators()

            # Translation statistics
            self.stats = {
                'total_translations': 0,
                'cache_hits': 0,
                'gemini_translations': 0,
                'fallback_translations': 0,
                'batch_translations': 0
            }

            logger.info("Translation service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize translation service: {e}")
            raise ChatbotException(f"Failed to initialize translation service: {e}")

    def _create_translation_prompts(self):
        """Create specialized prompts for different translation scenarios"""

        # Single term translation prompt
        self.term_translation_prompt = ChatPromptTemplate.from_template("""
You are a professional translator specializing in Machine Learning and Artificial Intelligence terminology.

Translate the following English term to Vietnamese:
**Term:** {text}

Guidelines:
1. Provide the most accurate and commonly used Vietnamese technical term
2. If multiple translations exist, choose the one most widely accepted in Vietnamese ML/AI community
3. Maintain technical precision and context appropriateness
4. Return ONLY the Vietnamese translation, no explanations

**Vietnamese Translation:**""")

        # Technical text translation prompt
        self.technical_translation_prompt = ChatPromptTemplate.from_template("""
You are a professional translator specializing in Machine Learning and Artificial Intelligence content.

Translate the following English text to Vietnamese:
**Text:** {text}

Guidelines:
1. Preserve technical accuracy and meaning
2. Use appropriate Vietnamese ML/AI terminology
3. Maintain natural Vietnamese grammar and flow
4. Keep technical terms in English if no standard Vietnamese equivalent exists
5. Preserve formatting and structure
6. Return ONLY the Vietnamese translation

**Vietnamese Translation:**""")

        # Batch translation prompt
        self.batch_translation_prompt = ChatPromptTemplate.from_template("""
You are a professional translator specializing in Machine Learning and Artificial Intelligence.

Translate the following English terms to Vietnamese, one per line:
**Terms:**
{text}

Guidelines:
1. Provide accurate Vietnamese technical translation for each term
2. If no standard Vietnamese equivalent exists, keep the English term
3. Return translations in the same order, one per line
4. Include ONLY the translations, no numbering or explanations

**Vietnamese Translations:**""")

    def _initialize_fallback_translators(self):
        """Initialize fallback translation libraries"""
        try:
            # Try to import translation libraries
            from googletrans import Translator as GoogleTranslator
            self.google_translator = GoogleTranslator()
            logger.info("Google Translator fallback initialized")
        except ImportError:
            logger.warning("Google Translator not available, using only Gemini AI")
            self.google_translator = None

        try:
            from deep_translator import GoogleTranslator as DeepGoogleTranslator
            self.deep_translator = DeepGoogleTranslator()
            logger.info("Deep Translator fallback initialized")
        except ImportError:
            logger.warning("Deep Translator not available")
            self.deep_translator = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8)
    )
    async def translate_term(self, term: str, to_vietnamese: bool = True) -> str:
        """
        Translate a single technical term with caching and fallback
        """
        if not term or not term.strip():
            return term

        term = term.strip().lower()

        # Create cache key
        cache_key = f"{term}_{'vi' if to_vietnamese else 'en'}"

        # Check cache first
        cached_result = self.translation_cache.get(cache_key)
        if cached_result:
            self.stats['cache_hits'] += 1
            return cached_result

        try:
            # Detect source language if needed
            if to_vietnamese and not self._is_english(term):
                # Term is already Vietnamese or mixed
                result = term
            elif not to_vietnamese and self._is_english(term):
                # Term is already English
                result = term
            else:
                # Perform translation with Gemini AI
                result = await self._translate_with_gemini(term, to_vietnamese, is_term=True)

                # Validate translation quality
                if not self._validate_translation(term, result, to_vietnamese):
                    logger.warning(f"Gemini translation quality issue for '{term}', trying fallback")
                    result = await self._translate_with_fallback(term, to_vietnamese)

            # Cache the result
            self.translation_cache.put(cache_key, result)
            self.stats['total_translations'] += 1
            self.stats['gemini_translations'] += 1

            return result

        except Exception as e:
            logger.error(f"Failed to translate term '{term}': {e}")
            # Return original term as ultimate fallback
            return term

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=5)
    )
    async def translate_text(self, text: str, to_vietnamese: bool = True) -> str:
        """
        Translate longer technical text with caching
        """
        if not text or not text.strip():
            return text

        text = text.strip()

        # Create cache key (use first 100 chars for long text)
        text_key = text[:100] if len(text) > 100 else text
        cache_key = f"{hash(text_key)}_{'vi' if to_vietnamese else 'en'}"

        # Check cache first
        cached_result = self.translation_cache.get(cache_key)
        if cached_result:
            self.stats['cache_hits'] += 1
            return cached_result

        try:
            # Perform translation
            result = await self._translate_with_gemini(text, to_vietnamese, is_term=False)

            # Cache the result
            self.translation_cache.put(cache_key, result)
            self.stats['total_translations'] += 1
            self.stats['gemini_translations'] += 1

            return result

        except Exception as e:
            logger.error(f"Failed to translate text: {e}")
            return text

    async def translate_batch(self, terms: List[str], to_vietnamese: bool = True) -> List[str]:
        """
        Translate multiple terms efficiently with batch processing
        """
        if not terms:
            return []

        results = []
        uncached_terms = []
        uncached_indices = []

        # Check cache for each term
        for i, term in enumerate(terms):
            if not term or not term.strip():
                results.append(term)
                continue

            term_key = term.strip().lower()
            cache_key = f"{term_key}_{'vi' if to_vietnamese else 'en'}"

            cached_result = self.translation_cache.get(cache_key)
            if cached_result:
                results.append(cached_result)
                self.stats['cache_hits'] += 1
            else:
                results.append(None)  # Placeholder
                uncached_terms.append(term_key)
                uncached_indices.append(i)

        # Process uncached terms in batches
        if uncached_terms:
            try:
                batch_results = await self._batch_translate_with_gemini(
                    uncached_terms, to_vietnamese
                )

                # Update results and cache
                for i, (term, translation) in enumerate(zip(uncached_terms, batch_results)):
                    original_index = uncached_indices[i]
                    results[original_index] = translation

                    # Cache the result
                    cache_key = f"{term}_{'vi' if to_vietnamese else 'en'}"
                    self.translation_cache.put(cache_key, translation)

                self.stats['batch_translations'] += 1
                self.stats['total_translations'] += len(uncached_terms)

            except Exception as e:
                logger.error(f"Batch translation failed: {e}")
                # Fall back to individual translations
                for i, term in enumerate(uncached_terms):
                    original_index = uncached_indices[i]
                    try:
                        translation = await self.translate_term(term, to_vietnamese)
                        results[original_index] = translation
                    except Exception as term_error:
                        logger.error(f"Failed to translate term '{term}': {term_error}")
                        results[original_index] = term

        return results

    async def _translate_with_gemini(self, text: str, to_vietnamese: bool, is_term: bool = False) -> str:
        """Translate using Gemini AI with specialized prompts"""
        try:
            # Choose appropriate prompt
            if is_term:
                prompt = self.term_translation_prompt
                target_lang = "Vietnamese" if to_vietnamese else "English"
            else:
                prompt = self.technical_translation_prompt
                target_lang = "Vietnamese" if to_vietnamese else "English"

            # Create chain
            chain = (
                prompt
                | self.translation_llm
                | self.output_parser
            )

            # Prepare input
            chain_input = {"text": text}

            # Generate translation with timeout
            translation = await asyncio.wait_for(
                chain.ainvoke(chain_input),
                timeout=15.0
            )

            # Clean up translation
            translation = translation.strip()

            # Post-process translation
            translation = self._post_process_translation(text, translation, to_vietnamese)

            return translation

        except asyncio.TimeoutError:
            logger.error(f"Gemini translation timed out for '{text[:50]}...'")
            raise ChatbotException(f"Translation timeout for '{text[:50]}...'")
        except Exception as e:
            logger.error(f"Gemini translation failed: {e}")
            raise

    async def _batch_translate_with_gemini(self, terms: List[str], to_vietnamese: bool) -> List[str]:
        """Batch translate using Gemini AI for efficiency"""
        try:
            # Join terms with newlines
            terms_text = "\n".join(terms)

            # Create chain
            chain = (
                self.batch_translation_prompt
                | self.translation_llm
                | self.output_parser
            )

            # Prepare input
            chain_input = {"text": terms_text}

            # Generate translations with timeout
            translations_text = await asyncio.wait_for(
                chain.ainvoke(chain_input),
                timeout=20.0
            )

            # Split results back into list
            translations = translations_text.strip().split('\n')

            # Ensure we have the right number of results
            if len(translations) != len(terms):
                logger.warning(f"Batch translation returned {len(translations)} results for {len(terms)} terms")
                # Pad or truncate as needed
                while len(translations) < len(terms):
                    translations.append(terms[len(translations)])  # Use original as fallback
                translations = translations[:len(terms)]

            # Clean up each translation
            cleaned_translations = []
            for translation in translations:
                cleaned = translation.strip()
                # Remove any numbering that might have been added
                cleaned = re.sub(r'^\d+\.\s*', '', cleaned)
                cleaned_translations.append(cleaned)

            return cleaned_translations

        except asyncio.TimeoutError:
            logger.error("Batch translation timed out")
            raise ChatbotException("Batch translation timeout")
        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            raise

    async def _translate_with_fallback(self, term: str, to_vietnamese: bool) -> str:
        """Fallback translation using alternative libraries"""
        self.stats['fallback_translations'] += 1

        # Try Google Translator
        if self.google_translator:
            try:
                if to_vietnamese:
                    result = self.google_translator.translate(term, src='en', dest='vi').text
                else:
                    result = self.google_translator.translate(term, src='vi', dest='en').text

                if result and result != term:
                    return result.strip()
            except Exception as e:
                logger.warning(f"Google Translator fallback failed: {e}")

        # Try Deep Translator
        if self.deep_translator:
            try:
                if to_vietnamese:
                    result = self.deep_translator.translate(term, source='en', target='vi')
                else:
                    result = self.deep_translator.translate(term, source='vi', target='en')

                if result and result != term:
                    return result.strip()
            except Exception as e:
                logger.warning(f"Deep Translator fallback failed: {e}")

        # Ultimate fallback - return original
        return term

    def _is_english(self, text: str) -> bool:
        """Simple English language detection"""
        try:
            # Check for Vietnamese characters
            vietnamese_chars = set('áàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ')
            text_lower = text.lower()

            # If contains Vietnamese characters, it's not pure English
            if any(char in vietnamese_chars for char in text_lower):
                return False

            # Simple heuristics
            english_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
            words = text_lower.split()

            if len(words) == 0:
                return True

            english_word_count = sum(1 for word in words if word in english_words)

            # If more than 30% of words are common English words, treat as English
            return english_word_count / len(words) > 0.3

        except Exception:
            return True  # Default to English

    def _validate_translation(self, original: str, translation: str, to_vietnamese: bool) -> bool:
        """Validate translation quality"""
        if not translation or not translation.strip():
            return False

        # Check if translation is identical to original (possible failure)
        if translation.strip().lower() == original.strip().lower():
            return False

        # Check for reasonable length
        if to_vietnamese:
            # Vietnamese translations are typically similar or slightly longer
            if len(translation) < len(original) * 0.3 or len(translation) > len(original) * 3:
                return False
        else:
            # English translations
            if len(translation) < len(original) * 0.3 or len(translation) > len(original) * 3:
                return False

        # Check for translation error indicators
        error_indicators = ['error', 'failed', 'unable', 'cannot', 'sorry', 'translation', 'translate']
        translation_lower = translation.lower()

        if any(indicator in translation_lower for indicator in error_indicators):
            return False

        return True

    def _post_process_translation(self, original: str, translation: str, to_vietnamese: bool) -> str:
        """Post-process translation for quality and consistency"""
        try:
            # Remove common artifacts
            translation = translation.strip()

            # Remove quotes if entire translation is quoted
            if translation.startswith('"') and translation.endswith('"'):
                translation = translation[1:-1].strip()

            if translation.startswith("'") and translation.endswith("'"):
                translation = translation[1:-1].strip()

            # Ensure technical term consistency for Vietnamese
            if to_vietnamese:
                # Common Vietnamese ML term normalizations
                term_mappings = {
                    'học máy': 'học máy',
                    'trí tuệ nhân tạo': 'trí tuệ nhân tạo',
                    'mạng neural': 'mạng nơ-ron',
                    'học sâu': 'học sâu',
                    'machine learning': 'học máy',
                    'ai': 'trí tuệ nhân tạo',
                    'neural network': 'mạng nơ-ron',
                    'deep learning': 'học sâu'
                }

                for en_term, vi_term in term_mappings.items():
                    if en_term.lower() in translation.lower():
                        translation = re.sub(
                            r'\b' + re.escape(en_term) + r'\b',
                            vi_term,
                            translation,
                            flags=re.IGNORECASE
                        )

            return translation

        except Exception as e:
            logger.warning(f"Translation post-processing failed: {e}")
            return translation

    def get_translation_stats(self) -> Dict[str, Any]:
        """Get translation service statistics"""
        return {
            **self.stats,
            'cache_size': self.translation_cache.size(),
            'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['total_translations'], 1) * 100,
            'gemini_success_rate': self.stats['gemini_translations'] / max(self.stats['total_translations'], 1) * 100,
            'fallback_usage_rate': self.stats['fallback_translations'] / max(self.stats['total_translations'], 1) * 100
        }

    def clear_cache(self) -> None:
        """Clear all translation caches"""
        self.translation_cache.clear()
        self.batch_translation_cache.clear()
        logger.info("Translation caches cleared")

    def preload_common_terms(self, terms: List[str], translations: List[str], to_vietnamese: bool = True) -> None:
        """Preload common term translations into cache"""
        try:
            if len(terms) != len(translations):
                logger.warning("Terms and translations lists must have same length")
                return

            for term, translation in zip(terms, translations):
                if term and translation:
                    cache_key = f"{term.strip().lower()}_{'vi' if to_vietnamese else 'en'}"
                    self.translation_cache.put(cache_key, translation.strip())

            logger.info(f"Preloaded {len(terms)} common translations")

        except Exception as e:
            logger.error(f"Failed to preload common terms: {e}")

# Global translation service instance
translation_service = None

def get_translation_service() -> TranslationService:
    """Get or create global translation service instance"""
    global translation_service
    if translation_service is None:
        translation_service = TranslationService()
    return translation_service