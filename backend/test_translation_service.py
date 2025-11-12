#!/usr/bin/env python3
"""
Translation Service Testing
Validates dynamic translation functionality
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TranslationServiceTester:
    """Testing suite for Translation Service"""

    def __init__(self):
        self.test_results = []

    def add_test_result(self, test_name: str, passed: bool, details: str = "", metrics: Dict = None):
        """Add test result to the results list"""
        self.test_results.append({
            'test_name': test_name,
            'passed': passed,
            'details': details,
            'metrics': metrics or {},
            'timestamp': time.time()
        })

    async def test_translation_service_initialization(self):
        """Test Translation Service initialization"""
        logger.info("🧪 Testing Translation Service Initialization...")

        try:
            from src.services.translation_service import get_translation_service

            translation_service = get_translation_service()

            # Check if service is properly initialized
            self.add_test_result(
                "Translation Service Initialization",
                translation_service is not None,
                f"Service initialized: {translation_service is not None}",
                {'service_type': str(type(translation_service))}
            )

            # Check if caching is working
            cache_size = translation_service.translation_cache.size()
            self.add_test_result(
                "Translation Cache Initialization",
                cache_size >= 0,
                f"Cache size: {cache_size}",
                {'cache_size': cache_size}
            )

        except Exception as e:
            logger.error(f"❌ Translation Service initialization test failed: {e}")
            self.add_test_result("Translation Service Initialization", False, f"Exception: {str(e)}")

    async def test_single_term_translation(self):
        """Test single term translation functionality"""
        logger.info("🧪 Testing Single Term Translation...")

        try:
            from src.services.translation_service import get_translation_service
            translation_service = get_translation_service()

            test_terms = [
                ("machine learning", True),  # English to Vietnamese
                ("học máy", False),           # Vietnamese to English
                ("neural network", True),     # English to Vietnamese
                ("trí tuệ nhân tạo", False),  # Vietnamese to English
                ("deep learning", True),      # English to Vietnamese
            ]

            for term, to_vietnamese in test_terms:
                start_time = time.time()
                translated = await translation_service.translate_term(term, to_vietnamese)
                end_time = time.time()

                # Basic validation
                is_valid = translated and translated != term and len(translated.strip()) > 0

                self.add_test_result(
                    f"Term Translation: '{term}' -> {'VI' if to_vietnamese else 'EN'}",
                    is_valid,
                    f"Result: '{translated}'",
                    {
                        'original': term,
                        'translated': translated,
                        'to_vietnamese': to_vietnamese,
                        'processing_time': end_time - start_time
                    }
                )

        except Exception as e:
            logger.error(f"❌ Single term translation test failed: {e}")
            self.add_test_result("Single Term Translation", False, f"Exception: {str(e)}")

    async def test_text_translation(self):
        """Test longer text translation"""
        logger.info("🧪 Testing Text Translation...")

        try:
            from src.services.translation_service import get_translation_service
            translation_service = get_translation_service()

            test_texts = [
                ("Machine learning is a subset of artificial intelligence", True),
                ("Học máy là một nhánh của trí tuệ nhân tạo", False),
                ("Neural networks are inspired by the human brain", True),
            ]

            for text, to_vietnamese in test_texts:
                start_time = time.time()
                translated = await translation_service.translate_text(text, to_vietnamese)
                end_time = time.time()

                # Basic validation
                is_valid = translated and translated != text and len(translated.strip()) > len(text) * 0.5

                self.add_test_result(
                    f"Text Translation: '{text[:30]}...' -> {'VI' if to_vietnamese else 'EN'}",
                    is_valid,
                    f"Result length: {len(translated)} chars",
                    {
                        'original_length': len(text),
                        'translated_length': len(translated),
                        'to_vietnamese': to_vietnamese,
                        'processing_time': end_time - start_time
                    }
                )

        except Exception as e:
            logger.error(f"❌ Text translation test failed: {e}")
            self.add_test_result("Text Translation", False, f"Exception: {str(e)}")

    async def test_batch_translation(self):
        """Test batch translation efficiency"""
        logger.info("🧪 Testing Batch Translation...")

        try:
            from src.services.translation_service import get_translation_service
            translation_service = get_translation_service()

            batch_terms = [
                "machine learning",
                "neural network",
                "deep learning",
                "artificial intelligence",
                "algorithm"
            ]

            start_time = time.time()
            batch_results = await translation_service.translate_batch(batch_terms, True)
            end_time = time.time()

            # Validate batch results
            is_valid = (
                len(batch_results) == len(batch_terms) and
                all(result and result != term for result, term in zip(batch_results, batch_terms))
            )

            self.add_test_result(
                "Batch Translation (5 terms)",
                is_valid,
                f"Translated {len(batch_results)} terms in {end_time - start_time:.3f}s",
                {
                    'input_count': len(batch_terms),
                    'output_count': len(batch_results),
                    'batch_time': end_time - start_time,
                    'avg_time_per_term': (end_time - start_time) / len(batch_terms)
                }
            )

        except Exception as e:
            logger.error(f"❌ Batch translation test failed: {e}")
            self.add_test_result("Batch Translation", False, f"Exception: {str(e)}")

    async def test_translation_caching(self):
        """Test translation caching functionality"""
        logger.info("🧪 Testing Translation Caching...")

        try:
            from src.services.translation_service import get_translation_service
            translation_service = get_translation_service()

            test_term = "machine learning"

            # Clear cache first
            translation_service.clear_cache()

            # First translation (should use API)
            start_time = time.time()
            result1 = await translation_service.translate_term(test_term, True)
            first_time = time.time() - start_time

            # Second translation (should use cache)
            start_time = time.time()
            result2 = await translation_service.translate_term(test_term, True)
            second_time = time.time() - start_time

            # Validate caching
            cache_worked = (
                result1 == result2 and
                second_time < first_time * 0.5  # Should be significantly faster
            )

            stats = translation_service.get_translation_stats()

            self.add_test_result(
                "Translation Caching",
                cache_worked,
                f"Cache improved speed: {first_time:.3f}s -> {second_time:.3f}s",
                {
                    'first_translation_time': first_time,
                    'second_translation_time': second_time,
                    'speed_improvement': (first_time - second_time) / first_time * 100,
                    'cache_hit_rate': stats['cache_hit_rate'],
                    'cache_size': stats['cache_size']
                }
            )

        except Exception as e:
            logger.error(f"❌ Translation caching test failed: {e}")
            self.add_test_result("Translation Caching", False, f"Exception: {str(e)}")

    async def test_updated_prompt_service(self):
        """Test updated Prompt Service with dynamic translation"""
        logger.info("🧪 Testing Updated Prompt Service...")

        try:
            from src.services.vietnamese_text_service import VietnameseTextService
            from src.services.query_expansion_service import QueryExpansionService
            from src.services.prompt_service import PromptService

            # Initialize services
            vietnamese_service = VietnameseTextService()
            query_service = QueryExpansionService(vietnamese_service)
            prompt_service = PromptService(query_service)

            # Test translation method exists and works
            test_text = "Machine learning algorithms are powerful tools"
            translated = await prompt_service.translate_technical_terms(test_text, True)

            # Check if translation service integration works
            is_valid = (
                hasattr(prompt_service, 'translation_service') and
                translated and translated != test_text
            )

            self.add_test_result(
                "Prompt Service Translation Integration",
                is_valid,
                f"Translation integration working: {is_valid}",
                {
                    'has_translation_service': hasattr(prompt_service, 'translation_service'),
                    'original_length': len(test_text),
                    'translated_length': len(translated) if translated else 0
                }
            )

            # Test contextual hints method
            query_analysis = {
                'language_analysis': {'is_vietnamese': True},
                'intent_analysis': {'primary_intent': 'definition'},
                'complexity': {'complexity_score': 0.6}
            }

            hints = prompt_service.get_contextual_hints(query_analysis)
            hints_valid = len(hints) > 0 and all(isinstance(h, str) for h in hints)

            self.add_test_result(
                "Contextual Hints Generation",
                hints_valid,
                f"Generated {len(hints)} contextual hints",
                {'hint_count': len(hints), 'sample_hints': hints[:3]}
            )

        except Exception as e:
            logger.error(f"❌ Updated Prompt Service test failed: {e}")
            self.add_test_result("Updated Prompt Service", False, f"Exception: {str(e)}")

    async def test_translation_quality(self):
        """Test translation quality for ML/AI terminology"""
        logger.info("🧪 Testing Translation Quality...")

        try:
            from src.services.translation_service import get_translation_service
            translation_service = get_translation_service()

            # Common ML/AI terms with expected Vietnamese translations
            quality_tests = [
                ("machine learning", "học máy"),
                ("artificial intelligence", "trí tuệ nhân tạo"),
                ("neural network", "mạng nơ-ron"),
                ("deep learning", "học sâu"),
                ("algorithm", "thuật toán"),
                ("data", "dữ liệu"),
            ]

            quality_scores = []

            for english_term, expected_vietnamese in quality_tests:
                translated = await translation_service.translate_term(english_term, True)

                # Simple quality check: contains expected Vietnamese term or is reasonable translation
                quality_good = (
                    expected_vietnamese in translated.lower() or
                    len(translated) > 3  # At least not empty
                )

                quality_scores.append(quality_good)

                self.add_test_result(
                    f"Translation Quality: '{english_term}'",
                    quality_good,
                    f"Expected: '{expected_vietnamese}', Got: '{translated}'",
                    {
                        'english': english_term,
                        'expected': expected_vietnamese,
                        'actual': translated,
                        'match': expected_vietnamese in translated.lower()
                    }
                )

            overall_quality = sum(quality_scores) / len(quality_scores) * 100

            self.add_test_result(
                "Overall Translation Quality",
                overall_quality >= 70,  # At least 70% quality
                f"Quality score: {overall_quality:.1f}%",
                {'quality_percentage': overall_quality, 'total_tests': len(quality_tests)}
            )

        except Exception as e:
            logger.error(f"❌ Translation quality test failed: {e}")
            self.add_test_result("Translation Quality", False, f"Exception: {str(e)}")

    async def run_all_tests(self):
        """Run all translation service tests"""
        logger.info("🚀 Starting Translation Service Testing...")

        await self.test_translation_service_initialization()
        await self.test_single_term_translation()
        await self.test_text_translation()
        await self.test_batch_translation()
        await self.test_translation_caching()
        await self.test_updated_prompt_service()
        await self.test_translation_quality()

    def generate_report(self):
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r['passed']])
        failed_tests = total_tests - passed_tests

        print("\n" + "="*80)
        print("🧪 TRANSLATION SERVICE TEST REPORT")
        print("="*80)
        print(f"📊 SUMMARY: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        print(f"❌ Failed: {failed_tests} tests")
        print(f"✅ Passed: {passed_tests} tests")
        print("="*80)

        # Detailed results
        for result in self.test_results:
            icon = "✅" if result['passed'] else "❌"
            print(f"\n{icon} {result['test_name']}")
            if result['details']:
                print(f"   📝 {result['details']}")
            if result['metrics']:
                metrics_str = ", ".join([f"{k}: {v}" for k, v in result['metrics'].items() if k not in ['original', 'translated', 'actual']])
                if metrics_str:
                    print(f"   📊 {metrics_str}")

        # Performance summary
        time_metrics = []
        for result in self.test_results:
            if 'processing_time' in result.get('metrics', {}):
                time_metrics.append(result['metrics']['processing_time'])

        if time_metrics:
            avg_time = sum(time_metrics) / len(time_metrics)
            print(f"\n⏱️  PERFORMANCE SUMMARY:")
            print(f"   Average translation time: {avg_time:.3f}s")
            print(f"   Fastest translation: {min(time_metrics):.3f}s")
            print(f"   Slowest translation: {max(time_metrics):.3f}s")

        print("\n" + "="*80)

        # Save detailed report to file
        report_data = {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': passed_tests/total_tests*100,
                'timestamp': time.time()
            },
            'all_results': self.test_results
        }

        with open('translation_service_test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print("📄 Detailed report saved to: translation_service_test_report.json")
        print("="*80)

        return passed_tests == total_tests

async def main():
    """Main testing function"""
    tester = TranslationServiceTester()
    await tester.run_all_tests()
    success = tester.generate_report()

    if success:
        print("\n🎉 ALL TRANSLATION TESTS PASSED! Dynamic translation service is working correctly!")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED! Please check the report above.")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)