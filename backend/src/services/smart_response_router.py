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
                'introduction': "",  # Remove introduction - start directly with content
                'document_section': "\n**Thông tin từ tài liệu chuyên khảo:**\n{document_content}",
                'general_section': "\n**Thông tin bổ sung:**\n{general_content}",
                'conclusion': ""  # Remove conclusion - end with content
            },
            'english': {
                'introduction': "",  # Remove introduction - start directly with content
                'document_section': "\n**From referenced documents:**\n{document_content}",
                'general_section': "\n**Additional information:**\n{general_content}",
                'conclusion': ""  # Remove conclusion - end with content
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

            # Get response strategy - RAG-first approach without quality thresholds
            strategy = self.intent_classifier.get_response_strategy(intent_analysis)

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
                  # High confidence for general knowledge
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

            # RAG-first approach: always use RAG context when available, then supplement with general knowledge
            try:
                # Always try to generate response with RAG context first
                if doc_result:
                    # Try to generate RAG response even if content quality is low
                    rag_response = await self.ai_service.generate_response(
                        query=query,
                        context=doc_result['context_texts']
                    )

                    # If we have a general knowledge response, enhance the RAG response
                    if general_response and general_response.response:
                        # Use RAG response as primary, but note that general knowledge was available
                        return ResponseResult(
                            response=rag_response,
                            strategy_used="rag_first_with_general_available",
                            sources=doc_result['sources'],
                            citations=doc_result['sources'],
                            explanation=f"RAG-first approach with general knowledge supplement - {intent_analysis.reasoning}",
                            processing_time=0.0  # Will be set by caller
                        )
                    else:
                        # RAG-only response
                        return ResponseResult(
                            response=rag_response,
                            strategy_used="rag_only",
                            sources=doc_result['sources'],
                                                        citations=doc_result['sources'],
                            explanation=f"RAG-first approach - {intent_analysis.reasoning}",
                            processing_time=0.0  # Will be set by caller
                        )
                else:
                    # No RAG content, fall back to general knowledge
                    if general_response:
                        return general_response
                    else:
                        return await self._generate_general_knowledge_response(query, intent_analysis, session_id)

            except Exception as e:
                logger.error(f"RAG-first response generation failed: {e}")
                # Fallback to general knowledge if RAG fails
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
        return f"""Bạn là một người hướng dẫn chuyên nghiệp về Machine Learning và AI. Hãy trả lời câu hỏi sau một cách chi tiết, có cấu trúc và giáo dục.

Câu hỏi: {query}

HƯỚNG DẪN CHO CÂU TRẢ LỜI TOÀN DIỆN BẰNG TIẾNG VIỆT:

1. CẤU TRÚC RÕ RÀNG:
   ● **Định nghĩa cơ bản** - Giải thích khái niệm một cách rõ ràng
   ● **Các thành phần chính** - Các yếu tố kỹ thuật quan trọng
   ● **Quy trình hoạt động** - Cách hệ thống hoạt động
   ● **Ứng dụng thực tế** - Ví dụ và cách sử dụng trong thực tế
   ● **Đặc điểm nổi bật** - Điều gì làm nó độc đáo

2. NỘI DUNG CHI TIẾT:
   ● Dùng bullet points (●) và số thứ tự để dễ đọc
   ● Bao gồm ví dụ cụ thể và ứng dụng thực tế
   ● Giải thích thuật ngữ kỹ thuật khi xuất hiện lần đầu
   ● Cân bằng giữa tính chính xác kỹ thuật và khả năng hiểu biết
   ● Tạo các section rõ ràng với **bold** headings

3. PHONG CÁCH VIẾT:
   ● Chuyên nghiệp nhưng mang tính giáo dục cao
   ● Bắt đầu trực tiếp với nội dung (không chào hỏi)
   ● Dùng "**bold**" cho thuật ngữ quan trọng khi đề cập lần đầu
   ● Duy trì tính nhất quán trong thuật ngữ kỹ thuật tiếng Việt

SAI (quá ngắn gọn):
- "Deep learning là một phương pháp tiếp cận AI..." (quá đơn giản)
- Chỉ định nghĩa mà không có ví dụ hay giải thích

ĐÚNG (toàn diện):
- "Deep learning là một nhánh của học máy sử dụng mạng nơ-ron sâu..." (theo sau là giải thích chi tiết)

Hãy cung cấp câu trả lời toàn diện, giáo dục và có cấu trúc rõ ràng!"""

    def _create_english_general_prompt(self, query: str, intent_analysis: IntentAnalysis) -> str:
        """Create prompt for English general knowledge response"""
        return f"""You are a professional guide for Machine Learning and AI. Answer the following question with comprehensive, structured, and educational content.

Question: {query}

GUIDELINES FOR COMPREHENSIVE ENGLISH RESPONSES:

1. CLEAR STRUCTURE:
   ● **Basic Definition** - Clear explanation of the concept
   ● **Key Components** - Important technical elements
   ● **How It Works** - The process and mechanics
   ● **Real Applications** - Practical examples and use cases
   ● **Key Characteristics** - What makes it unique from traditional approaches

2. DETAILED CONTENT:
   ● Use bullet points (●) and numbered lists for better readability
   ● Include concrete examples and real-world applications
   ● Explain technical terms clearly when they first appear
   ● Balance technical accuracy with accessibility for learners
   ● Create sections with clear headings using **bold** text

3. WRITING STYLE:
   ● Professional but highly educational tone
   ● Start directly with content (no greetings)
   ● Use "**bold**" for important terms on first mention
   ● Maintain consistency in technical vocabulary

WRONG (too brief):
- "Deep learning is an AI approach..." (too simple)
- Only definition without examples or explanations

RIGHT (comprehensive):
- "Deep learning is a branch of machine learning that uses deep neural networks..." (followed by detailed explanation)

Provide comprehensive, educational, and well-structured answers!"""

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