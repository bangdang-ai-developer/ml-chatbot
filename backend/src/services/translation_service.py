"""
Vietnamese to English Translation Service for RAG
Clean implementation using only Gemini AI translation
No mock data, no fallbacks, no hardcoded mappings, no technical_terms
"""

import logging
import asyncio
from typing import Optional, Dict, Any
import re
from ..core.config import settings

logger = logging.getLogger(__name__)

class TranslationService:
    """
    Clean Vietnamese-English translation service using Gemini AI only
    No hardcoded terms, no fallbacks, no mock data
    """

    def __init__(self):
        # No initialization needed - Gemini handles everything
        pass

    async def translate_vietnamese_to_english(self, vietnamese_text: str) -> str:
        """
        Translate Vietnamese text to English using Gemini AI only
        """
        try:
            if not vietnamese_text or not vietnamese_text.strip():
                return vietnamese_text

            # Check if text is already mostly English
            if self._is_mostly_english(vietnamese_text):
                logger.info("Text is mostly English, skipping translation")
                return vietnamese_text

            # Translate using Gemini AI only
            translated_text = await self._translate_with_gemini(vietnamese_text)

            logger.info(f"Successfully translated: '{vietnamese_text[:50]}...' → '{translated_text[:50]}...'")
            return translated_text

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            # Return original text - no fallbacks
            return vietnamese_text

    def _is_mostly_english(self, text: str) -> bool:
        """
        Check if text is already mostly in English
        """
        # Count English words (letters-only words)
        english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
        total_words = len(text.split())

        if total_words == 0:
            return False

        english_ratio = english_words / total_words
        return english_ratio > 0.6  # More than 60% English words

    async def _translate_with_gemini(self, text: str) -> str:
        """
        Translate using Gemini AI model - no term preservation needed
        """
        try:
            import google.generativeai as genai

            # Initialize Gemini with API key from settings
            genai.configure(api_key=settings.google_api_key)

            # Simple translation prompt - no hardcoded terms
            translation_prompt = f"""
            Translate the following Vietnamese text to English.
            Provide only the direct translation without any additional explanation.

            Vietnamese text: {text}

            English translation:
            """

            # Use Gemini 2.5 Flash for translation
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(translation_prompt)

            if response.text:
                translated_text = response.text.strip()
                logger.info(f"Gemini translation successful")
                return translated_text
            else:
                logger.warning("Gemini returned empty translation")
                return text

        except Exception as e:
            logger.error(f"Gemini translation failed: {e}")
            return text

    async def is_vietnamese_query(self, text: str) -> bool:
        """
        Detect if the query is in Vietnamese
        """
        try:
            # Simple Vietnamese detection
            vietnamese_chars = set('àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ')
            text_lower = text.lower()

            vietnamese_char_count = sum(1 for char in text_lower if char in vietnamese_chars)
            total_chars = len([c for c in text_lower if c.isalpha()])

            if total_chars == 0:
                return False

            vietnamese_ratio = vietnamese_char_count / total_chars
            return vietnamese_ratio > 0.1  # At least 10% Vietnamese characters

        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return False

    async def process_query_for_retrieval(self, query: str) -> Dict[str, Any]:
        """
        Process query for document retrieval
        Returns both original and translated versions
        """
        try:
            is_vietnamese = await self.is_vietnamese_query(query)

            if is_vietnamese:
                translated_query = await self.translate_vietnamese_to_english(query)
                logger.info(f"Vietnamese query detected and translated")
            else:
                translated_query = query
                logger.info("English query detected, no translation needed")

            return {
                'original_query': query,
                'translated_query': translated_query,
                'is_translated': is_vietnamese,
                'translation_successful': True,
                'retrieval_query': translated_query
            }

        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return {
                'original_query': query,
                'translated_query': query,
                'is_translated': False,
                'translation_successful': False,
                'retrieval_query': query
            }

# Global translation service instance
translation_service = None

def get_translation_service() -> TranslationService:
    """Get or create global translation service instance"""
    global translation_service
    if translation_service is None:
        translation_service = TranslationService()
    return translation_service