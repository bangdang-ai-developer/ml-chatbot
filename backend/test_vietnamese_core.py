#!/usr/bin/env python3
"""
Core Vietnamese RAG Services Testing
Tests the fundamental Vietnamese language processing components
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

class VietnameseCoreTester:
    """Testing suite for core Vietnamese RAG services"""

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

    async def test_vietnamese_text_service(self):
        """Test Vietnamese Text Processing Service"""
        logger.info("🧪 Testing Vietnamese Text Service...")

        try:
            # Import here to avoid issues with missing dependencies
            from src.services.vietnamese_text_service import VietnameseTextService
            vietnamese_service = VietnameseTextService()

            test_cases = [
                "Học máy là gì?",
                "Giải thích về mạng nơ-ron",
                "AI và Machine Learning khác nhau như thế nào?",
                "Implement gradient descent trong deep learning",
                "Các phương pháp tối ưu neural networks"
            ]

            for i, text in enumerate(test_cases):
                # Test language detection
                language, confidence = vietnamese_service.detect_language(text)
                is_vietnamese = language == 'vietnamese' and confidence > 0.3

                self.add_test_result(
                    f"Language Detection {i+1}",
                    is_vietnamese,
                    f"Detected: {language} (confidence: {confidence:.2f})",
                    {'language': language, 'confidence': confidence}
                )

                # Test text normalization
                normalized = vietnamese_service.normalize_vietnamese_text(text)
                self.add_test_result(
                    f"Text Normalization {i+1}",
                    len(normalized) > 0 and isinstance(normalized, str),
                    f"Normalized: '{normalized[:30]}...'",
                    {'original_length': len(text), 'normalized_length': len(normalized)}
                )

                # Test tokenization
                tokens = vietnamese_service.tokenize_vietnamese_words(text)
                self.add_test_result(
                    f"Tokenization {i+1}",
                    len(tokens) > 0,
                    f"Tokenized into {len(tokens)} tokens",
                    {'token_count': len(tokens), 'sample_tokens': tokens[:3]}
                )

                # Test keyword extraction
                keywords = vietnamese_service.extract_vietnamese_keywords(text)
                self.add_test_result(
                    f"Keyword Extraction {i+1}",
                    isinstance(keywords, list),
                    f"Extracted {len(keywords)} keywords",
                    {'keyword_count': len(keywords), 'keywords': keywords}
                )

            # Test query processing
            test_query = "Học máy là gì và nó khác AI như thế nào?"
            processed = vietnamese_service.process_vietnamese_query(test_query)

            self.add_test_result(
                "Query Processing",
                'original_query' in processed and 'processed_query' in processed,
                f"Processed query with {len(processed.get('keywords', []))} keywords",
                {
                    'detected_language': processed.get('detected_language'),
                    'keyword_count': len(processed.get('keywords', [])),
                    'complexity_score': processed.get('complexity_score', 0)
                }
            )

            # Test metadata creation
            metadata = vietnamese_service.create_vietnamese_metadata(text, "test_chunk", 1)
            self.add_test_result(
                "Metadata Creation",
                'char_count' in metadata and 'is_vietnamese' in metadata,
                f"Created metadata for {metadata.get('char_count', 0)} characters",
                metadata
            )

        except Exception as e:
            logger.error(f"❌ Vietnamese Text Service test failed: {e}")
            self.add_test_result("Vietnamese Text Service", False, f"Exception: {str(e)}")

    async def test_query_expansion_service(self):
        """Test Query Understanding and Expansion Service"""
        logger.info("🧪 Testing Query Expansion Service...")

        try:
            from src.services.vietnamese_text_service import VietnameseTextService
            from src.services.query_expansion_service import QueryExpansionService

            vietnamese_service = VietnameseTextService()
            query_service = QueryExpansionService(vietnamese_service)

            test_queries = [
                "Học máy là gì?",
                "So sánh neural network và decision tree",
                "Làm thế nào để train model machine learning?",
                "Các hyperparameter trong deep learning"
            ]

            for i, query in enumerate(test_queries):
                # Test intent detection
                intent_analysis = query_service.detect_query_intent(query)
                self.add_test_result(
                    f"Intent Detection {i+1}",
                    'primary_intent' in intent_analysis,
                    f"Detected intent: {intent_analysis.get('primary_intent', 'N/A')}",
                    {
                        'primary_intent': intent_analysis.get('primary_intent'),
                        'is_question': intent_analysis.get('is_question'),
                        'entity_count': len(intent_analysis.get('entities', []))
                    }
                )

                # Test query expansion
                expansions = query_service.expand_query_semantically(query)
                self.add_test_result(
                    f"Query Expansion {i+1}",
                    len(expansions) > 0,
                    f"Generated {len(expansions)} expansions",
                    {
                        'expansion_count': len(expansions),
                        'sample_expansions': expansions[:2]
                    }
                )

                # Test query variations
                variations = query_service.generate_hyphen_queries(query)
                self.add_test_result(
                    f"Query Variations {i+1}",
                    len(variations) > 0,
                    f"Generated {len(variations)} variations",
                    {'variation_count': len(variations)}
                )

                # Test complexity analysis
                complexity = query_service.analyze_query_complexity(query)
                self.add_test_result(
                    f"Complexity Analysis {i+1}",
                    'complexity_score' in complexity,
                    f"Complexity: {complexity.get('complexity_score', 0):.2f}",
                    complexity
                )

            # Test comprehensive query processing
            test_query = "Giải thích về mạng nơ-ron và ứng dụng trong học sâu"
            comprehensive = query_service.process_query_comprehensive(test_query)

            self.add_test_result(
                "Comprehensive Processing",
                'original_query' in comprehensive and 'expansions' in comprehensive,
                f"Processed with {len(comprehensive.get('expansions', {}).get('all_queries', []))} total queries",
                {
                    'detected_language': comprehensive.get('language_analysis', {}).get('detected_language'),
                    'primary_intent': comprehensive.get('intent_analysis', {}).get('primary_intent'),
                    'complexity_score': comprehensive.get('complexity', {}).get('complexity_score', 0),
                    'total_expansions': len(comprehensive.get('expansions', {}).get('all_queries', []))
                }
            )

        except Exception as e:
            logger.error(f"❌ Query Expansion Service test failed: {e}")
            self.add_test_result("Query Expansion Service", False, f"Exception: {str(e)}")

    async def test_prompt_service(self):
        """Test Vietnamese Prompt Engineering Service"""
        logger.info("🧪 Testing Prompt Service...")

        try:
            from src.services.vietnamese_text_service import VietnameseTextService
            from src.services.query_expansion_service import QueryExpansionService
            from src.services.prompt_service import PromptService

            vietnamese_service = VietnameseTextService()
            query_service = QueryExpansionService(vietnamese_service)
            prompt_service = PromptService(query_service)

            test_queries = [
                {"query": "Học máy là gì?", "expected_lang": "vietnamese"},
                {"query": "What is machine learning?", "expected_lang": "english"},
                {"query": "So sánh các thuật toán classification", "expected_lang": "vietnamese"}
            ]

            for i, test_case in enumerate(test_queries):
                query = test_case['query']
                query_analysis = query_service.process_query_comprehensive(query)

                # Test system prompt generation
                system_prompt = prompt_service.create_vietnamese_system_prompt(query_analysis)
                self.add_test_result(
                    f"System Prompt {i+1}",
                    len(system_prompt) > 100,  # Should be substantial
                    f"Generated {len(system_prompt)} character prompt",
                    {'prompt_length': len(system_prompt)}
                )

                # Test response style configuration
                style_config = prompt_service.get_response_style_config(query_analysis)
                self.add_test_result(
                    f"Response Style {i+1}",
                    'is_vietnamese' in style_config,
                    f"Style: {style_config.get('tone', 'N/A')}",
                    style_config
                )

                # Test technical term translation
                translated = prompt_service.translate_technical_terms(query, True)
                self.add_test_result(
                    f"Translation {i+1}",
                    isinstance(translated, str),
                    f"Translation result length: {len(translated)}",
                    {'original_length': len(query), 'translated_length': len(translated)}
                )

        except Exception as e:
            logger.error(f"❌ Prompt Service test failed: {e}")
            self.add_test_result("Prompt Service", False, f"Exception: {str(e)}")

    async def test_service_integration(self):
        """Test integration between Vietnamese services"""
        logger.info("🧪 Testing Service Integration...")

        try:
            from src.services.vietnamese_text_service import VietnameseTextService
            from src.services.query_expansion_service import QueryExpansionService
            from src.services.prompt_service import PromptService

            # Initialize services
            vietnamese_service = VietnameseTextService()
            query_service = QueryExpansionService(vietnamese_service)
            prompt_service = PromptService(query_service)

            test_queries = [
                "Học máy là gì?",
                "Giải thích về mạng nơ-ron",
                "So sánh supervised và unsupervised learning"
            ]

            for i, query in enumerate(test_queries):
                start_time = time.time()

                # Step 1: Vietnamese text processing
                processed_query = vietnamese_service.process_vietnamese_query(query)

                # Step 2: Query understanding and expansion
                comprehensive_analysis = query_service.process_query_comprehensive(query)

                # Step 3: Prompt generation
                system_prompt = prompt_service.create_vietnamese_system_prompt(comprehensive_analysis)

                # Step 4: Get contextual hints
                hints = prompt_service.get_contextual_hints(comprehensive_analysis)

                end_time = time.time()

                # Verify integration worked
                integration_success = (
                    processed_query and
                    comprehensive_analysis and
                    system_prompt and
                    hints
                )

                self.add_test_result(
                    f"Service Integration {i+1}",
                    integration_success,
                    f"Integrated processing in {(end_time - start_time):.3f}s",
                    {
                        'processing_time': end_time - start_time,
                        'has_processed_query': bool(processed_query),
                        'has_analysis': bool(comprehensive_analysis),
                        'has_prompt': bool(system_prompt),
                        'has_hints': bool(hints),
                        'detected_language': comprehensive_analysis.get('language_analysis', {}).get('detected_language'),
                        'intent': comprehensive_analysis.get('intent_analysis', {}).get('primary_intent'),
                        'hint_count': len(hints)
                    }
                )

        except Exception as e:
            logger.error(f"❌ Service Integration test failed: {e}")
            self.add_test_result("Service Integration", False, f"Exception: {str(e)}")

    async def run_all_tests(self):
        """Run all tests"""
        logger.info("🚀 Starting Vietnamese Core Services Testing...")

        await self.test_vietnamese_text_service()
        await self.test_query_expansion_service()
        await self.test_prompt_service()
        await self.test_service_integration()

    def generate_report(self):
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r['passed']])
        failed_tests = total_tests - passed_tests

        print("\n" + "="*80)
        print("🧪 VIETNAMESE CORE RAG SERVICES TEST REPORT")
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
                metrics_str = ", ".join([f"{k}: {v}" for k, v in result['metrics'].items() if k != 'keywords'])
                print(f"   📊 {metrics_str}")

        # Performance summary
        time_metrics = []
        for result in self.test_results:
            if 'processing_time' in result.get('metrics', {}):
                time_metrics.append(result['metrics']['processing_time'])

        if time_metrics:
            avg_time = sum(time_metrics) / len(time_metrics)
            print(f"\n⏱️  PERFORMANCE SUMMARY:")
            print(f"   Average processing time: {avg_time:.3f}s")
            print(f"   Fastest time: {min(time_metrics):.3f}s")
            print(f"   Slowest time: {max(time_metrics):.3f}s")

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

        with open('vietnamese_core_test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print("📄 Detailed report saved to: vietnamese_core_test_report.json")
        print("="*80)

        return passed_tests == total_tests

async def main():
    """Main testing function"""
    tester = VietnameseCoreTester()
    await tester.run_all_tests()
    success = tester.generate_report()

    if success:
        print("\n🎉 ALL CORE TESTS PASSED! Vietnamese RAG Services are working correctly!")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED! Please check the report above.")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)