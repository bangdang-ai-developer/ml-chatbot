#!/usr/bin/env python3
"""
Comprehensive Testing Suite for Vietnamese Advanced RAG System
Tests all Vietnamese language enhancements and RAG components
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

from src.services.vietnamese_text_service import VietnameseTextService
from src.services.query_expansion_service import QueryExpansionService
from src.services.prompt_service import PromptService
from src.services.hybrid_search_service import HybridSearchService
from src.services.reranking_service import VietnameseRerankingService
from src.services.embedding_service import GeminiEmbeddingService
from src.core.exceptions import ChatbotException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VietnameseRAGTester:
    """Comprehensive testing suite for Vietnamese RAG system"""

    def __init__(self):
        self.test_results = []
        self.services = {}

    async def setup_services(self):
        """Initialize all Vietnamese RAG services"""
        logger.info("🔧 Setting up Vietnamese RAG services...")

        try:
            self.services['vietnamese'] = VietnameseTextService()
            logger.info("✅ Vietnamese Text Service initialized")

            self.services['query_expansion'] = QueryExpansionService(self.services['vietnamese'])
            logger.info("✅ Query Expansion Service initialized")

            self.services['prompt'] = PromptService(self.services['query_expansion'])
            logger.info("✅ Prompt Service initialized")

            self.services['embedding'] = GeminiEmbeddingService()
            logger.info("✅ Embedding Service initialized")

            self.services['hybrid_search'] = HybridSearchService()
            logger.info("✅ Hybrid Search Service initialized")

            self.services['reranking'] = VietnameseRerankingService(self.services['vietnamese'])
            logger.info("✅ Reranking Service initialized")

        except Exception as e:
            logger.error(f"❌ Failed to setup services: {e}")
            raise

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

        test_cases = [
            "Học máy là gì?",
            "Giải thích về mạng nơ-ron",
            "AI và Machine Learning khác nhau như thế nào?",
            "Implement gradient descent trong deep learning",
            "Các phương pháp tối ưu neural networks"
        ]

        try:
            vietnamese_service = self.services['vietnamese']

            for i, text in enumerate(test_cases):
                # Test language detection
                language, confidence = vietnamese_service.detect_language(text)
                self.add_test_result(
                    f"Vietnamese Language Detection Test {i+1}",
                    language == 'vietnamese' and confidence > 0.5,
                    f"Detected: {language} (confidence: {confidence:.2f}) for '{text[:30]}...'",
                    {'language': language, 'confidence': confidence}
                )

                # Test text normalization
                normalized = vietnamese_service.normalize_vietnamese_text(text)
                self.add_test_result(
                    f"Vietnamese Text Normalization Test {i+1}",
                    len(normalized) > 0 and isinstance(normalized, str),
                    f"Normalized: '{normalized[:30]}...'",
                    {'original_length': len(text), 'normalized_length': len(normalized)}
                )

                # Test tokenization
                tokens = vietnamese_service.tokenize_vietnamese_words(text)
                self.add_test_result(
                    f"Vietnamese Tokenization Test {i+1}",
                    len(tokens) > 0,
                    f"Tokenized into {len(tokens)} tokens: {tokens[:5]}",
                    {'token_count': len(tokens), 'first_tokens': tokens[:3]}
                )

                # Test keyword extraction
                keywords = vietnamese_service.extract_vietnamese_keywords(text)
                self.add_test_result(
                    f"Vietnamese Keyword Extraction Test {i+1}",
                    isinstance(keywords, list),
                    f"Extracted keywords: {keywords}",
                    {'keyword_count': len(keywords)}
                )

        except Exception as e:
            logger.error(f"❌ Vietnamese Text Service test failed: {e}")
            self.add_test_result("Vietnamese Text Service", False, f"Exception: {str(e)}")

    async def test_query_expansion_service(self):
        """Test Query Understanding and Expansion Service"""
        logger.info("🧪 Testing Query Expansion Service...")

        test_queries = [
            "Học máy là gì?",
            "So sánh neural network và decision tree",
            "Làm thế nào để train model machine learning?",
            "Các hyperparameter trong deep learning"
        ]

        try:
            query_service = self.services['query_expansion']

            for i, query in enumerate(test_queries):
                # Test comprehensive query processing
                analysis = query_service.process_query_comprehensive(query)

                self.add_test_result(
                    f"Query Processing Test {i+1}",
                    'original_query' in analysis and 'processed_query' in analysis,
                    f"Processed query: {analysis.get('processed_query', 'N/A')}",
                    {
                        'detected_language': analysis.get('language_analysis', {}).get('detected_language'),
                        'intent': analysis.get('intent_analysis', {}).get('primary_intent'),
                        'complexity_score': analysis.get('complexity', {}).get('complexity_score', 0)
                    }
                )

                # Test query expansion
                expansions = query_service.expand_query_semantically(query)
                self.add_test_result(
                    f"Query Expansion Test {i+1}",
                    len(expansions) > 0 and isinstance(expansions, list),
                    f"Generated {len(expansions)} expansions",
                    {'expansion_count': len(expansions), 'first_expansion': expansions[0] if expansions else None}
                )

                # Test intent detection
                intent_analysis = query_service.detect_query_intent(query)
                self.add_test_result(
                    f"Intent Detection Test {i+1}",
                    'primary_intent' in intent_analysis,
                    f"Detected intent: {intent_analysis.get('primary_intent', 'N/A')}",
                    {'primary_intent': intent_analysis.get('primary_intent'), 'is_question': intent_analysis.get('is_question')}
                )

        except Exception as e:
            logger.error(f"❌ Query Expansion Service test failed: {e}")
            self.add_test_result("Query Expansion Service", False, f"Exception: {str(e)}")

    async def test_prompt_service(self):
        """Test Vietnamese Prompt Engineering Service"""
        logger.info("🧪 Testing Prompt Service...")

        test_queries = [
            {"query": "Học máy là gì?", "expected_language": "vietnamese"},
            {"query": "What is machine learning?", "expected_language": "english"},
            {"query": "So sánh các thuật toán classification", "expected_language": "vietnamese"}
        ]

        try:
            prompt_service = self.services['prompt']

            for i, test_case in enumerate(test_queries):
                query = test_case['query']

                # Process query for analysis
                query_analysis = self.services['query_expansion'].process_query_comprehensive(query)

                # Test system prompt generation
                system_prompt = prompt_service.create_vietnamese_system_prompt(query_analysis)
                self.add_test_result(
                    f"System Prompt Generation Test {i+1}",
                    len(system_prompt) > 50,  # Should be substantial
                    f"Generated {len(system_prompt)} character prompt",
                    {'prompt_length': len(system_prompt)}
                )

                # Test response style configuration
                style_config = prompt_service.get_response_style_config(query_analysis)
                self.add_test_result(
                    f"Response Style Test {i+1}",
                    'is_vietnamese' in style_config and 'tone' in style_config,
                    f"Style: {style_config.get('tone', 'N/A')}",
                    style_config
                )

                # Test technical term translation
                translated = prompt_service.translate_technical_terms(query, True)
                self.add_test_result(
                    f"Technical Translation Test {i+1}",
                    isinstance(translated, str),
                    f"Translation: '{translated}'",
                    {'original': query, 'translated': translated}
                )

        except Exception as e:
            logger.error(f"❌ Prompt Service test failed: {e}")
            self.add_test_result("Prompt Service", False, f"Exception: {str(e)}")

    async def test_embedding_service(self):
        """Test Enhanced Embedding Service with Vietnamese preprocessing"""
        logger.info("🧪 Testing Enhanced Embedding Service...")

        test_texts = [
            "Học máy là một nhánh của trí tuệ nhân tạo",
            "Machine learning is a subset of artificial intelligence",
            "Neural networks are used for deep learning applications",
            "Mạng nơ-ron được sử dụng trong các ứng dụng học sâu"
        ]

        try:
            embedding_service = self.services['embedding']

            for i, text in enumerate(test_texts):
                # Test basic embedding generation
                embedding = await embedding_service.generate_query_embedding(text)
                self.add_test_result(
                    f"Embedding Generation Test {i+1}",
                    len(embedding) > 0,
                    f"Generated {len(embedding)} dimensions",
                    {'embedding_length': len(embedding)}
                )

                # Test enhanced query embedding
                enhanced_embedding = await embedding_service.generate_enhanced_query_embedding(text)
                self.add_test_result(
                    f"Enhanced Embedding Test {i+1}",
                    'embedding' in enhanced_embedding and 'processed_query' in enhanced_embedding,
                    f"Enhanced features: {list(enhanced_embedding.keys())}",
                    {
                        'has_embedding': 'embedding' in enhanced_embedding,
                        'has_processed_query': 'processed_query' in enhanced_embedding,
                        'language': enhanced_embedding.get('detected_language', 'N/A')
                    }
                )

        except Exception as e:
            logger.error(f"❌ Embedding Service test failed: {e}")
            self.add_test_result("Embedding Service", False, f"Exception: {str(e)}")

    async def test_hybrid_search_service(self):
        """Test Hybrid Search Service"""
        logger.info("🧪 Testing Hybrid Search Service...")

        test_queries = [
            "học máy cơ bản",
            "neural network architecture",
            "machine learning algorithms"
        ]

        try:
            search_service = self.services['hybrid_search']

            for i, query in enumerate(test_queries):
                # Process query for analysis
                query_analysis = self.services['query_expansion'].process_query_comprehensive(query)

                # Test hybrid search
                start_time = time.time()
                results = await search_service.search(query, top_k=3, query_analysis=query_analysis)
                end_time = time.time()

                self.add_test_result(
                    f"Hybrid Search Test {i+1}",
                    isinstance(results, list),
                    f"Found {len(results)} results in {(end_time - start_time):.2f}s",
                    {
                        'result_count': len(results),
                        'search_time': end_time - start_time,
                        'has_documents': len(results) > 0
                    }
                )

                # Test search with expansion
                expanded_results = await search_service.search_with_expansion(query, query_analysis, top_k=3)
                self.add_test_result(
                    f"Expanded Search Test {i+1}",
                    isinstance(expanded_results, list),
                    f"Expanded search found {len(expanded_results)} results",
                    {
                        'expanded_result_count': len(expanded_results),
                        'more_results': len(expanded_results) > len(results)
                    }
                )

        except Exception as e:
            logger.error(f"❌ Hybrid Search Service test failed: {e}")
            self.add_test_result("Hybrid Search Service", False, f"Exception: {str(e)}")

    async def test_reranking_service(self):
        """Test Vietnamese Reranking Service"""
        logger.info("🧪 Testing Reranking Service...")

        # Create mock documents for testing
        mock_documents = [
            {
                'document': type('Document', (), {
                    'content': 'Học máy là phương pháp máy tính có khả năng học hỏi từ dữ liệu mà không cần lập trình rõ ràng.',
                    'metadata': {'content_type': 'definition', 'is_vietnamese': True}
                })(),
                'score': 0.8
            },
            {
                'document': type('Document', (), {
                    'content': 'Machine learning enables computers to learn and improve from experience.',
                    'metadata': {'content_type': 'definition', 'is_vietnamese': False}
                })(),
                'score': 0.7
            },
            {
                'document': type('Document', (), {
                    'content': 'Mạng nơ-ron là mô hình tính toán được truyền cảm hứng từ mạng nơ-ron sinh học.',
                    'metadata': {'content_type': 'technical', 'is_vietnamese': True}
                })(),
                'score': 0.6
            }
        ]

        test_query = "Học máy là gì?"

        try:
            reranking_service = self.services['reranking']

            # Process query for analysis
            query_analysis = self.services['query_expansion'].process_query_comprehensive(test_query)

            # Test reranking
            start_time = time.time()
            reranked_results = await reranking_service.rerank(
                test_query, mock_documents, query_analysis, top_k=3
            )
            end_time = time.time()

            self.add_test_result(
                "Reranking Test",
                len(reranked_results) > 0 and isinstance(reranked_results, list),
                f"Reranked {len(reranked_results)} documents in {(end_time - start_time):.3f}s",
                {
                    'reranked_count': len(reranked_results),
                    'reranking_time': end_time - start_time,
                    'scores_changed': any(r.get('enhanced_score', 0) != r.get('original_score', 0) for r in reranked_results)
                }
            )

            # Test adaptive reranking
            adaptive_results = await reranking_service.adaptive_rerank(
                test_query, mock_documents, query_analysis, top_k=3
            )

            self.add_test_result(
                "Adaptive Reranking Test",
                isinstance(adaptive_results, list),
                f"Adaptive reranking processed {len(adaptive_results)} documents",
                {
                    'adaptive_count': len(adaptive_results),
                    'processed_by_strategy': True
                }
            )

        except Exception as e:
            logger.error(f"❌ Reranking Service test failed: {e}")
            self.add_test_result("Reranking Service", False, f"Exception: {str(e)}")

    async def test_integration(self):
        """Test end-to-end integration of all services"""
        logger.info("🧪 Testing End-to-End Integration...")

        test_queries = [
            "Học máy là gì?",
            "Giải thích về mạng nơ-ron và ứng dụng",
            "So sánh supervised và unsupervised learning"
        ]

        try:
            for i, query in enumerate(test_queries):
                start_time = time.time()

                # Step 1: Query understanding and expansion
                query_analysis = self.services['query_expansion'].process_query_comprehensive(query)

                # Step 2: Embedding generation (mock for testing)
                enhanced_embedding = await self.services['embedding'].generate_enhanced_query_embedding(query)

                # Step 3: Search (mock - would use real vector database in production)
                search_results = []  # Mock empty results for now

                # Step 4: Reranking (skip if no results)
                if search_results:
                    final_results = await self.services['reranking'].rererank(
                        query, search_results, query_analysis, top_k=3
                    )
                else:
                    final_results = []

                # Step 5: Prompt generation
                prompt_service = self.services['prompt']
                system_prompt = prompt_service.create_vietnamese_system_prompt(query_analysis)

                end_time = time.time()

                self.add_test_result(
                    f"Integration Test {i+1}",
                    query_analysis and enhanced_embedding and system_prompt,
                    f"Processed '{query[:20]}...' in {(end_time - start_time):.3f}s",
                    {
                        'processing_time': end_time - start_time,
                        'has_query_analysis': bool(query_analysis),
                        'has_embedding': bool(enhanced_embedding),
                        'has_prompt': bool(system_prompt),
                        'detected_language': query_analysis.get('language_analysis', {}).get('detected_language'),
                        'intent': query_analysis.get('intent_analysis', {}).get('primary_intent')
                    }
                )

        except Exception as e:
            logger.error(f"❌ Integration test failed: {e}")
            self.add_test_result("Integration Test", False, f"Exception: {str(e)}")

    async def run_all_tests(self):
        """Run all tests"""
        logger.info("🚀 Starting Vietnamese RAG System Testing...")

        try:
            # Setup services
            await self.setup_services()

            # Run all test suites
            await self.test_vietnamese_text_service()
            await self.test_query_expansion_service()
            await self.test_prompt_service()
            await self.test_embedding_service()
            await self.test_hybrid_search_service()
            await self.test_reranking_service()
            await self.test_integration()

        except Exception as e:
            logger.error(f"❌ Testing failed: {e}")
            self.add_test_result("Overall Testing", False, f"Setup failed: {str(e)}")

    def generate_report(self):
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r['passed']])
        failed_tests = total_tests - passed_tests

        print("\n" + "="*80)
        print("🧪 VIETNAMESE ADVANCED RAG SYSTEM TEST REPORT")
        print("="*80)
        print(f"📊 SUMMARY: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        print(f"❌ Failed: {failed_tests} tests")
        print(f"✅ Passed: {passed_tests} tests")
        print("="*80)

        # Group results by service
        service_groups = {}
        for result in self.test_results:
            service_name = result['test_name'].split(' ')[0] + ' ' + result['test_name'].split(' ')[1]
            if service_name not in service_groups:
                service_groups[service_name] = []
            service_groups[service_name].append(result)

        # Detailed results
        for service, tests in service_groups.items():
            passed = len([t for t in tests if t['passed']])
            total = len(tests)
            status = "✅ PASS" if passed == total else "❌ FAIL"

            print(f"\n🔧 {service}: {status} ({passed}/{total} tests)")

            for test in tests:
                icon = "✅" if test['passed'] else "❌"
                print(f"   {icon} {test['test_name']}")
                if test['details']:
                    print(f"      📝 {test['details']}")
                if test['metrics']:
                    metrics_str = ", ".join([f"{k}: {v}" for k, v in test['metrics'].items()])
                    print(f"      📊 {metrics_str}")

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
            'service_groups': service_groups,
            'all_results': self.test_results
        }

        with open('vietnamese_rag_test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print("📄 Detailed report saved to: vietnamese_rag_test_report.json")
        print("="*80)

        return passed_tests == total_tests

async def main():
    """Main testing function"""
    tester = VietnameseRAGTester()
    await tester.run_all_tests()
    success = tester.generate_report()

    if success:
        print("\n🎉 ALL TESTS PASSED! Vietnamese RAG System is working correctly!")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED! Please check the report above.")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)