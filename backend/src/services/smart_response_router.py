"""
Smart Response Router
Intelligently routes queries to appropriate response strategies
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import asyncio
from dataclasses import dataclass

from .query_intent_classifier import QueryIntentClassifier, IntentAnalysis, QueryIntent
from .ai_service import GeminiAIService
from .rag_service import RAGService

logger = logging.getLogger(__name__)

@dataclass
class ResponseResult:
    """Result of intelligent response generation"""
    response: str
    strategy_used: str
    sources: List[Dict[str, Any]]
    confidence: float
    citations: List[Dict[str, Any]]
    explanation: str
    processing_time: float

class SmartResponseRouter:
    """
    Intelligent router that combines RAG and general AI knowledge
    """

    def __init__(self, rag_service: RAGService, ai_service: GeminiAIService):
        self.rag_service = rag_service
        self.ai_service = ai_service
        self.intent_classifier = QueryIntentClassifier()

        # Templates for hybrid responses
        self.hybrid_templates = {
            'vietnamese': {
                'introduction': "Chào bạn, để trả lời câu hỏi của bạn, mình sẽ kết hợp thông tin từ tài liệu và kiến thức chung:",
                'document_section': "\n📚 **Từ tài liệu tham khảo:**\n{document_content}",
                'general_section': "\n🤖 **Từ kiến thức tổng quan:**\n{general_content}",
                'conclusion': "\nHy vọng thông tin này hữu ích cho bạn! Bạn có muốn mình giải thích sâu hơn phần nào không?"
            },
            'english': {
                'introduction': "Hello! To answer your question, I'll combine information from documents with general knowledge:",
                'document_section': "\n📚 **From referenced documents:**\n{document_content}",
                'general_section': "\n🤖 **From general knowledge:**\n{general_content}",
                'conclusion': "\nI hope this information is helpful! Would you like me to explain any part in more detail?"
            }
        }

    async def generate_intelligent_response(
        self,
        query: str,
        query_analysis: Dict[str, Any],
        session_id: str = "default"
    ) -> ResponseResult:
        """
        Generate intelligent response using optimal strategy
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # Classify query intent
            intent_analysis = self.intent_classifier.classify_intent(query, query_analysis)

            # Get response strategy with reasonable default context quality
            # For deep learning queries with both books indexed, use higher context quality
            default_context_quality = 0.5  # Default to moderate quality for hybrid queries
            if intent_analysis.intent.value in ['hybrid'] and any(keyword in query.lower() for keyword in ['deep learning', 'học sâu', 'neural network', 'mạng nơ-ron']):
                default_context_quality = 0.7  # Higher quality for ML/DL specific queries

            strategy = self.intent_classifier.get_response_strategy(intent_analysis, default_context_quality)

            logger.info(f"Query: '{query[:50]}...' - Intent: {intent_analysis.intent.value}, Strategy: {strategy}")

            # Generate response based on strategy
            if strategy['primary_source'] == 'documents':
                result = await self._generate_rag_response(query, intent_analysis, session_id)
            elif strategy['primary_source'] == 'ai_knowledge':
                result = await self._generate_general_knowledge_response(query, intent_analysis, session_id)
            elif strategy['primary_source'] == 'hybrid':
                result = await self._generate_hybrid_response(query, intent_analysis, session_id, strategy)
            elif strategy['primary_source'] == 'conversational':
                result = await self._generate_conversational_response(query, intent_analysis, session_id)
            else:
                # Fallback to hybrid
                result = await self._generate_hybrid_response(query, intent_analysis, session_id, strategy)

            # Update processing time
            result.processing_time = asyncio.get_event_loop().time() - start_time

            return result

        except Exception as e:
            logger.error(f"Error in intelligent response generation: {e}")
            # Fallback response
            processing_time = asyncio.get_event_loop().time() - start_time
            return ResponseResult(
                response="Xin lỗi, mình gặp lỗi khi xử lý câu hỏi của bạn. Bạn có thể thử lại không?",
                strategy_used="fallback",
                sources=[],
                confidence=0.1,
                citations=[],
                explanation="Error occurred, used fallback response",
                processing_time=processing_time
            )

    async def _generate_rag_response(
        self,
        query: str,
        intent_analysis: IntentAnalysis,
        session_id: str
    ) -> ResponseResult:
        """Generate response using RAG approach"""
        try:
            # Get documents from RAG service
            doc_result = await self.rag_service.retrieve_documents_for_query(query)

            if not doc_result['has_content']:
                # No relevant documents, fallback to general knowledge
                return await self._generate_general_knowledge_response(query, intent_analysis, session_id)

            # Generate response using AI service with context
            ai_response = await self.ai_service.generate_response(
                query=query,
                context=doc_result['context_texts']
            )

            return ResponseResult(
                response=ai_response,
                strategy_used="rag_only",
                sources=doc_result['sources'],
                confidence=intent_analysis.confidence,
                citations=doc_result['sources'],
                explanation=f"Used RAG approach - {intent_analysis.reasoning}",
                processing_time=0.0  # Will be set by caller
            )

        except Exception as e:
            logger.error(f"RAG response generation failed: {e}")
            # Fallback to general knowledge
            return await self._generate_general_knowledge_response(query, intent_analysis, session_id)

    async def _generate_general_knowledge_response(
        self,
        query: str,
        intent_analysis: IntentAnalysis,
        session_id: str
    ) -> ResponseResult:
        """Generate response using general AI knowledge"""
        try:
            # Create prompt for general knowledge response
            is_vietnamese = intent_analysis.specific_entities and any(
                'vietnamese' in str(entity).lower() or 'chào' in query.lower() for entity in intent_analysis.specific_entities
            ) or self._is_vietnamese_query(query)

            if is_vietnamese:
                prompt = self._create_vietnamese_general_prompt(query, intent_analysis)
            else:
                prompt = self._create_english_general_prompt(query, intent_analysis)

            # Generate response using AI service
            response = await self.ai_service.chat_completion([
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ])

            return ResponseResult(
                response=response,
                strategy_used="general_knowledge",
                sources=[],
                confidence=intent_analysis.confidence,
                citations=[],
                explanation=f"Used general AI knowledge - {intent_analysis.reasoning}",
                processing_time=0.0  # Will be set by caller
            )

        except Exception as e:
            logger.error(f"General knowledge response generation failed: {e}")
            # Ultimate fallback
            return ResponseResult(
                response="Xin lỗi, mình không thể trả lời câu hỏi này lúc này. Bạn có thể thử lại hoặc hỏi câu khác không?",
                strategy_used="ultimate_fallback",
                sources=[],
                confidence=0.1,
                citations=[],
                explanation="General knowledge failed, using ultimate fallback",
                processing_time=0.0
            )

    async def _generate_hybrid_response(
        self,
        query: str,
        intent_analysis: IntentAnalysis,
        session_id: str,
        strategy: Dict[str, Any]
    ) -> ResponseResult:
        """Generate hybrid response combining RAG and general knowledge"""
        try:
            # Get both RAG and general knowledge responses
            rag_task = None
            general_task = None

            if strategy.get('use_rag', False):
                rag_task = asyncio.create_task(
                    self.rag_service.retrieve_documents_for_query(query)
                )

            if strategy.get('use_general_knowledge', False):
                general_task = asyncio.create_task(
                    self._generate_general_knowledge_response(query, intent_analysis, session_id)
                )

            # Wait for results
            doc_result = None
            general_response = None

            if rag_task:
                try:
                    doc_result = await rag_task
                except Exception as e:
                    logger.warning(f"RAG document retrieval failed in hybrid mode: {e}")

            if general_task:
                try:
                    general_response = await general_task
                except Exception as e:
                    logger.warning(f"General knowledge response failed in hybrid mode: {e}")

            # Combine responses intelligently
            if doc_result and doc_result['has_content']:
                # We have RAG content, generate response with context
                try:
                    ai_response = await self.ai_service.generate_response(
                        query=query,
                        context=doc_result['context_texts']
                    )

                    return ResponseResult(
                        response=ai_response,
                        strategy_used="hybrid_with_rag",
                        sources=doc_result['sources'],
                        confidence=intent_analysis.confidence,
                        citations=doc_result['sources'],
                        explanation=f"Used hybrid approach with RAG context - {intent_analysis.reasoning}",
                        processing_time=0.0  # Will be set by caller
                    )
                except Exception as e:
                    logger.error(f"Hybrid AI response generation failed: {e}")
                    # Fallback to general knowledge
                    if general_response:
                        return general_response
                    else:
                        return await self._generate_general_knowledge_response(query, intent_analysis, session_id)
            else:
                # No RAG content, use general knowledge
                if general_response:
                    return general_response
                else:
                    return await self._generate_general_knowledge_response(query, intent_analysis, session_id)

        except Exception as e:
            logger.error(f"Hybrid response generation failed: {e}")
            # Ultimate fallback
            return await self._generate_general_knowledge_response(query, intent_analysis, session_id)

    async def _generate_conversational_response(
        self,
        query: str,
        intent_analysis: IntentAnalysis,
        session_id: str
    ) -> ResponseResult:
        """Generate conversational response"""
        conversational_responses = {
            'hello': "Chào bạn! Mình là người hướng dẫn về Machine Learning. Bạn có câu hỏi gì về ML/AI không?",
            'hi': "Xin chào! Rất vui được trò chuyện với bạn. Bạn muốn tìm hiểu về chủ đề gì hôm nay?",
            'thanks': "Không có gì! Mình rất vui được giúp đỡ bạn. Bạn còn câu hỏi nào khác không?",
            'goodbye': "Tạm biệt! Chúc bạn học tập hiệu quả. Hãy quay lại anytime nhé!",
            'help': "Chào bạn! Mình có thể giúp bạn về các chủ đề Machine Learning, AI, Neural Networks, Deep Learning và nhiều hơn nữa. Bạn muốn biết về gì?"
        }

        query_lower = query.lower().strip()
        response = conversational_responses.get(query_lower, "Chào bạn! Mình có thể giúp gì cho bạn về Machine Learning và AI?")

        return ResponseResult(
            response=response,
            strategy_used="conversational",
            sources=[],
            confidence=1.0,
            citations=[],
            explanation="Conversational response pattern matched",
            processing_time=0.0
        )

    def _combine_responses(
        self,
        rag_response: Optional[Dict],
        general_response: Optional[ResponseResult],
        intent_analysis: IntentAnalysis,
        original_query: str
    ) -> Dict[str, str]:
        """Intelligently combine RAG and general knowledge responses"""

        is_vietnamese = self._is_vietnamese_query(original_query)
        templates = self.hybrid_templates['vietnamese' if is_vietnamese else 'english']

        combined_text = templates['introduction']

        # Add RAG content if available
        if rag_response and rag_response.get('response'):
            document_content = rag_response['response']
            # Check if RAG provided meaningful content
            if len(document_content) > 100 and "không có" not in document_content.lower():
                combined_text += templates['document_section'].format(document_content=document_content)
            else:
                # RAG didn't provide useful content
                rag_response = None

        # Add general knowledge content if available
        if general_response and general_response.response:
            general_content = general_response.response
            # Avoid duplication
            if not rag_response or len(general_content) > len(rag_response.get('response', '')):
                combined_text += templates['general_section'].format(general_content=general_content)

        # Add conclusion
        combined_text += templates['conclusion']

        explanation_parts = []
        if rag_response:
            explanation_parts.append("Used document-specific information")
        if general_response:
            explanation_parts.append("Enhanced with general AI knowledge")
        if not rag_response and not general_response:
            explanation_parts.append("Used default response")

        return {
            'text': combined_text,
            'explanation': f"Hybrid approach: {', '.join(explanation_parts)} - {intent_analysis.reasoning}"
        }

    def _create_vietnamese_general_prompt(self, query: str, intent_analysis: IntentAnalysis) -> str:
        """Create prompt for Vietnamese general knowledge response"""
        return f"""Bạn là một người hướng dẫn chuyên nghiệp về Machine Learning và AI. Hãy trả lời câu hỏi sau một cách tự nhiên, thân thiện và chính xác.

Câu hỏi: {query}

Hướng dẫn:
1. Trả lời bằng tiếng Việt tự nhiên
2. Cung cấp thông tin chính xác và cập nhật
3. Dùng ví dụ thực tế để minh họa
4. Giải thích các khái niệm phức tạp một cách dễ hiểu
5. Nếu đây là khái niệm cơ bản, giải thích từ đầu đến cuối
6. Duy trì phong cách trò chuyện thân thiện

Trả lời thật tự nhiên như đang nói chuyện với bạn học!"""

    def _create_english_general_prompt(self, query: str, intent_analysis: IntentAnalysis) -> str:
        """Create prompt for English general knowledge response"""
        return f"""You are a professional guide for Machine Learning and AI. Answer the following question naturally, friendly, and accurately.

Question: {query}

Guidelines:
1. Provide accurate, up-to-date information
2. Use practical examples to illustrate concepts
3. Explain complex concepts in an easy-to-understand way
4. For fundamental concepts, provide comprehensive explanations
5. Maintain a conversational, friendly tone

Answer naturally as if talking to a learner!"""

    def _is_vietnamese_query(self, query: str) -> bool:
        """Simple check if query is in Vietnamese"""
        vietnamese_chars = set('áàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ')
        return any(char in vietnamese_chars for char in query.lower())

    async def get_routing_statistics(self) -> Dict[str, Any]:
        """Get statistics about routing decisions"""
        # This could be expanded to track actual usage patterns
        return {
            'supported_intents': [intent.value for intent in QueryIntent],
            'routing_strategies': ['rag_only', 'general_knowledge', 'hybrid', 'conversational'],
            'hybrid_enabled': True,
            'fallback_mechanisms': ['general_knowledge_fallback', 'ultimate_fallback']
        }