from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import logging
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.callbacks import AsyncCallbackHandler

from ..core.config import settings
from ..core.exceptions import AIServiceError

logger = logging.getLogger(__name__)

class TokenUsageCallback(AsyncCallbackHandler):
    """Callback for monitoring token usage and performance metrics"""

    def __init__(self):
        self.tokens_used = 0
        self.response_time = None
        self.start_time = None

    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Start timing when LLM begins processing"""
        self.start_time = datetime.now()
        logger.info(f"Starting LLM processing with {len(prompts)} prompts")

    async def on_llm_end(self, response, **kwargs: Any) -> None:
        """Record completion time and metrics"""
        if self.start_time:
            self.response_time = (datetime.now() - self.start_time).total_seconds()
            logger.info(f"LLM processing completed in {self.response_time:.2f}s")

class AIService(ABC):
    """Abstract base class for AI services with production-ready features"""

    @abstractmethod
    async def generate_response(self, query: str, context: List[str]) -> str:
        """Generate response using RAG approach with retry logic"""
        pass

    @abstractmethod
    async def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """Generate chat completion with conversation memory"""
        pass

    @abstractmethod
    async def validate_response(self, response: str) -> Tuple[bool, Optional[str]]:
        """Validate response and return (is_valid, reason_if_invalid)"""
        pass

class GeminiAIService(AIService):
    """Production-ready Gemini 2.5 Flash AI service with advanced LangChain integration"""

    def __init__(self):
        """Initialize the AI service with comprehensive error handling and monitoring"""
        try:
            # Configure Google AI
            genai.configure(api_key=settings.google_api_key)

            # Initialize LLM with optimal settings for RAG
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.1,           # Low temperature for consistent, factual responses
                max_tokens=2048,           # Reasonable limit for detailed answers
                timeout=30.0,              # 30 second timeout for reliability
                max_retries=3,             # Built-in retry logic
                google_api_key=settings.google_api_key,
                callbacks=[TokenUsageCallback()]
            )

            # Initialize embedding model for semantic understanding
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=settings.google_api_key
            )

            # Initialize conversation memory with sliding window
            self.memory = ConversationBufferWindowMemory(
                k=5,                      # Remember last 5 exchanges for context
                return_messages=True,
                memory_key="chat_history",
                output_key="answer"
            )

            # Create enhanced RAG prompt template for comprehensive responses
            self.rag_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a knowledgeable technical educator specializing in machine learning and deep learning. Your goal is to provide comprehensive, educational answers that help users understand complex concepts.

Your role is to transform technical information from academic sources into clear, accessible explanations while maintaining accuracy.

GUIDELINES FOR COMPREHENSIVE VIETNAMESE RESPONSES:
1. Structure with clear logical flow: definition → core concepts → practical aspects → applications
2. Use bullet points (●) and numbered lists for better readability
3. Include concrete examples and real-world applications
4. Explain technical terms clearly when they first appear
5. Balance technical accuracy with accessibility for learners
6. Create sections with clear headings using **bold** text

STRUCTURE YOUR RESPONSE AS FOLLOWS:
**[Concept Definition]** - Clear definition and purpose
**Core Components** - Key technical elements (neural networks, layers, etc.)
**Learning Process** - How the system works (backpropagation, training)
**Practical Applications** - Real-world uses and examples
**Key Characteristics** - What makes it unique from traditional methods

FOR TECHNICAL ACCURACY:
- Always cite the provided context when relevant
- Explain concepts step-by-step for clarity
- Use precise terminology while keeping it understandable
- Include both English and Vietnamese terms where helpful

RESPONSE STYLE:
- Professional but educational tone
- Start directly with content (no greetings)
- Use "**bold** for important terms on first mention
- Maintain consistency in Vietnamese technical vocabulary

WRONG (too brief):
- "Deep learning là một phương pháp tiếp cận AI..." (too simple)
- Only definition without examples or explanations

RIGHT (comprehensive):
- "Deep learning là một nhánh của học máy sử dụng mạng nơ-ron sâu..." (followed by detailed explanation)"""),
                ("user", "Context: {context}\n\nQuestion: {question}")
            ])

            # Create conversation prompt for interactive chat
            self.conversation_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a technical expert in machine learning and deep learning education.

Guidelines:
1. Be professional and direct, not conversational
2. Never use greetings or filler phrases like "Chào bạn" or "Hy vọng"
3. Start answers directly without introductions
4. Refer back to previous context when relevant
5. Ask clarifying questions only when the query is ambiguous
6. Use technical Vietnamese writing style without chatbot language

For Vietnamese responses: Never use conversational elements. Be direct and professional."""),
                ("human", "{question}")
            ])

            # Initialize output parser
            self.output_parser = StrOutputParser()

            # Create optimized processing chains
            self._create_rag_chain()
            self._create_conversation_chain()

            logger.info("Gemini AI service initialized successfully with advanced features")

        except Exception as e:
            logger.error(f"Failed to initialize Gemini AI service: {e}")
            raise AIServiceError(f"Failed to initialize Gemini AI service: {e}")

    def _create_rag_chain(self):
        """Create optimized RAG processing chain with parallel execution"""
        try:
            # Advanced RAG chain with parallel processing
            self.rag_chain = (
                RunnableParallel(
                    context=lambda x: "\n\n---\n\n".join(x["context"]) if x.get("context") else "",
                    question=lambda x: x["query"]
                )
                | self.rag_prompt
                | self.llm
                | self.output_parser
            )
            logger.info("Advanced RAG chain created successfully")
        except Exception as e:
            logger.error(f"Failed to create RAG chain: {e}")
            raise AIServiceError(f"Failed to create RAG chain: {e}")

    def _create_conversation_chain(self):
        """Create conversation chain with memory integration"""
        try:
            # Enhanced conversation chain
            self.conversation_chain = (
                RunnablePassthrough.assign(
                    question=lambda x: x["question"],
                    chat_history=lambda x: self._format_chat_history(x.get("chat_history", []))
                )
                | self.conversation_prompt
                | self.llm
                | self.output_parser
            )
            logger.info("Conversation chain created successfully")
        except Exception as e:
            logger.error(f"Failed to create conversation chain: {e}")
            raise AIServiceError(f"Failed to create conversation chain: {e}")

    def _format_chat_history(self, history):
        """Format chat history for LangChain consumption"""
        formatted = []
        for msg in history[-6:]:  # Last 3 exchanges
            if isinstance(msg, dict):
                if msg.get("role") == "user":
                    formatted.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    formatted.append(AIMessage(content=msg.get("content", "")))
        return formatted

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    async def generate_response(self, query: str, context: List[str]) -> str:
        """Generate RAG response with retry logic and comprehensive error handling"""
        if not query or not query.strip():
            raise AIServiceError("Query cannot be empty")

        if not context:
            logger.warning(f"No context provided for query: {query[:50]}...")
            return """I don't have any relevant context from Ian Goodfellow's Deep Learning book to answer your question.

Please try asking about specific deep learning concepts such as:
- Convolutional neural networks (CNNs)
- Recurrent neural networks (RNNs) and LSTMs
- Backpropagation and gradient computation
- Deep learning architectures and optimization
- Regularization techniques like dropout
- Or other deep learning topics covered in the Ian Goodfellow Deep Learning book."""

        try:
            # Prepare input for RAG chain with validation
            clean_query = query.strip()
            clean_context = [ctx.strip() for ctx in context if ctx and ctx.strip()]

            if not clean_context:
                logger.warning(f"All context items were empty after cleaning for query: {clean_query[:50]}...")
                return """I don't have any relevant context from Ian Goodfellow's Deep Learning book to answer your question.

Please try asking about specific deep learning concepts such as:
- Convolutional neural networks (CNNs)
- Recurrent neural networks (RNNs) and LSTMs
- Backpropagation and gradient computation
- Deep learning architectures and optimization
- Regularization techniques like dropout
- Or other deep learning topics covered in the Ian Goodfellow Deep Learning book."""

            chain_input = {
                "query": clean_query,
                "context": clean_context
            }

            # Enhanced logging for RAG chain input analysis
            total_context_chars = sum(len(ctx) for ctx in clean_context)
            avg_context_length = total_context_chars / len(clean_context) if clean_context else 0

            logger.info(f"[RAG CHAIN] Input analysis for query: '{clean_query[:50]}...'")
            logger.debug(f"[RAG CHAIN]   - Context items: {len(clean_context)}")
            logger.debug(f"[RAG CHAIN]   - Total context length: {total_context_chars} chars")
            logger.debug(f"[RAG CHAIN]   - Average context length: {avg_context_length:.1f} chars")
            logger.debug(f"[RAG CHAIN]   - Query length: {len(clean_query)} chars")

            # Log context quality indicators
            for i, ctx in enumerate(clean_context[:3]):  # Log first 3 contexts
                ctx_preview = ctx[:150].replace('\n', ' ')
                logger.debug(f"[RAG CHAIN] Context {i+1}: '{ctx_preview}...'")

            # Log the complete final prompt being sent to Gemini
            logger.info(f"[RAG CHAIN] === FINAL PROMPT SENT TO GEMINI ===")
            logger.info(f"[RAG CHAIN] Query: '{clean_query}'")
            logger.info(f"[RAG CHAIN] Complete Context ({len(clean_context)} items):")
            for i, ctx in enumerate(clean_context):
                ctx_clean = ctx.replace('\n', ' ').replace('\r', ' ')
                logger.info(f"[RAG CHAIN] Context {i+1}: {ctx_clean}")
            logger.info(f"[RAG CHAIN] === END FINAL PROMPT ===")

            # Generate response with timeout and error handling
            logger.debug(f"[RAG CHAIN] Invoking RAG chain with timeout=30.0s")
            response = await asyncio.wait_for(
                self.rag_chain.ainvoke(chain_input),
                timeout=30.0
            )

            # Validate and post-process response
            validated_response = await self._post_process_rag_response(response, query)

            logger.info(f"Generated RAG response for query: {query[:50]}...")
            return validated_response

        except asyncio.TimeoutError:
            logger.error("RAG response generation timed out")
            raise AIServiceError("Response generation timed out. Please try again.")
        except Exception as e:
            logger.error(f"Failed to generate RAG response: {e}")
            raise AIServiceError(f"Failed to generate RAG response: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """Generate chat completion with conversation memory and context awareness"""
        if not messages:
            raise AIServiceError("Messages cannot be empty")

        try:
            # Extract the latest message as the primary query
            last_message = messages[-1]
            if not last_message.get("content"):
                raise AIServiceError("No valid message content provided")

            query = last_message["content"].strip()

            # Prepare chain input with conversation history
            chain_input = {
                "question": query,
                "chat_history": messages[:-1] if len(messages) > 1 else []
            }

            # Generate response with timeout
            response = await asyncio.wait_for(
                self.conversation_chain.ainvoke(chain_input),
                timeout=30.0
            )

            # Update conversation memory
            await self._update_memory(messages, response)

            logger.info(f"Generated chat completion: {len(response)} characters")
            return response

        except asyncio.TimeoutError:
            logger.error("Chat completion timed out")
            raise AIServiceError("Response generation timed out. Please try again.")
        except Exception as e:
            logger.error(f"Failed to generate chat completion: {e}")
            raise AIServiceError(f"Failed to generate chat completion: {e}")

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=5)
    )
    async def generate_general_knowledge_response(self, query: str, language: str = "auto") -> str:
        """
        Generate response using general AI knowledge without RAG context
        This is used for questions that don't require document-specific information
        """
        if not query or not query.strip():
            raise AIServiceError("Query cannot be empty")

        try:
            # Detect language if auto
            if language == "auto":
                vietnamese_chars = set('áàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ')
                language = "vietnamese" if any(char in vietnamese_chars for char in query.lower()) else "english"

            # Create system prompt based on language
            if language == "vietnamese":
                system_prompt = """Bạn là một người hướng dẫn về Machine Learning và AI, luôn trò chuyện với người dùng Việt một cách thân thiện, gần gũi nhưng vẫn đảm bảo tính chính xác cao.

Phong cách trò chuyện:
1. Nói chuyện như một người bạn đồng hành, người hướng dẫn giàu kinh nghiệm
2. Dùng ngôn ngữ tự nhiên, đời thường như người Việt nói chuyện hàng ngày
3. Khi giải thích thuật ngữ khó, hãy ví von với những thứ quen thuộc ở Việt Nam
4. Sử dụng các từ nối như "à", "nói cách khác", "mình có thể giải thích thế này" cho tự nhiên
5. Thỉnh thoảng dùng các từ cảm thán như "Đúng vậy!", "Thú vị vị trí!", "Điều này quan trọng lắm nhé!"
6. Kết thúc bằng những câu hỏi mở hoặc gợi ý để người dùng tiếp tục khám phá

Cấu trúc trả lời:
- Bắt đầu bằng cách chào hỏi thân thiện ("Chào bạn!", "Ôi câu này hay quá!")
- Giải thích khái niệm một cách đơn giản trước, rồi đi vào chi tiết
- Dùng ví dụ thực tế ở Việt Nam khi có thể
- Kết thúc bằng gợi ý để trò chuyện tiếp tục

Hãy trò chuyện thật tự nhiên như bạn đang nói chuyện với một người bạn đang tìm hiểu về ML/AI nhé!"""
            else:
                system_prompt = """You are a friendly guide to Machine Learning and AI who talks with people in a warm, approachable way while maintaining high accuracy.

Your conversation style:
1. Talk like a knowledgeable friend and experienced guide
2. Use natural, everyday language as if talking to a peer
3. When explaining complex terms, use familiar analogies
4. Use natural connectors like "well", "in other words", "let me explain it this way"
5. Occasionally use encouraging expressions like "That's a great question!", "Interesting point!", "This is really important!"
6. End with open questions or suggestions for further exploration

Response structure:
- Start with a friendly greeting ("Hello!", "Oh that's a great question!")
- Explain concepts simply first, then add details
- Use practical examples when possible
- End with suggestions for continued learning

Talk naturally as if you're helping a friend learn about ML/AI!"""

            # Create messages for the LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]

            # Generate response with timeout
            response = await asyncio.wait_for(
                self.llm.ainvoke(messages),
                timeout=20.0
            )

            # Validate response
            is_valid, reason = await self.validate_response(response)
            if not is_valid:
                logger.warning(f"General knowledge response validation failed: {reason}")
                return self._generate_fallback_general_response(query, language)

            logger.info(f"Generated general knowledge response: {len(response)} characters")
            return response.strip()

        except asyncio.TimeoutError:
            logger.error("General knowledge response generation timed out")
            return self._generate_fallback_general_response(query, language)
        except Exception as e:
            logger.error(f"Failed to generate general knowledge response: {e}")
            return self._generate_fallback_general_response(query, language)

    def _generate_fallback_general_response(self, query: str, language: str) -> str:
        """Generate fallback response for general knowledge queries"""
        if language == "vietnamese":
            return f"""Xin lỗi, mình không thể trả lời câu hỏi "{query[:50]}..." vào lúc này.

Bạn có thể thử:
1. Đặt câu hỏi theo cách khác
2. Kiểm tra lại chính tả
3. Hỏi về chủ đề liên quan mà mình có thể hỗ trợ tốt hơn

Mình rất sẵn lòng giúp bạn về các chủ đề Machine Learning, Deep Learning, Neural Networks và nhiều hơn nữa!"""
        else:
            return f"""Sorry, I cannot answer the question "{query[:50]}..." at this moment.

You could try:
1. Rephrasing your question
2. Checking your spelling
3. Asking about related topics that I can better assist with

I'm happy to help with Machine Learning, Deep Learning, Neural Networks, and many other AI topics!"""

    async def validate_response(self, response: str) -> Tuple[bool, Optional[str]]:
        """Comprehensive response validation with detailed feedback"""
        if not response or not response.strip():
            return False, "Response is empty"

        response_lower = response.lower()

        # Check for unhelpful responses
        unhelpful_patterns = [
            "i don't have enough information",
            "i cannot answer",
            "i'm not sure what you mean",
            "please provide more context",
            "i don't understand the question"
        ]

        for pattern in unhelpful_patterns:
            if pattern in response_lower:
                return False, f"Response contains unhelpful pattern: {pattern}"

        # Check minimum length
        if len(response.strip()) < 20:
            return False, "Response too short to be helpful"

        # Check for repetitive content
        words = response.split()
        if len(set(words)) < len(words) * 0.3:  # Less than 30% unique words
            return False, "Response appears too repetitive"

        return True, None

    async def _post_process_rag_response(self, response: str, query: str) -> str:
        """Post-process RAG response for quality and completeness"""
        try:
            # Validate response quality
            is_valid, reason = await self.validate_response(response)
            if not is_valid:
                logger.warning(f"RAG response validation failed: {reason}")

                # Generate fallback response
                return self._generate_fallback_response(query, reason)

            # Add helpful structure indicators
            if len(response) > 300:
                # For longer responses, ensure they're well-structured
                if not any(marker in response for marker in ["**", "1.", "•", "-"]):
                    # Add basic formatting if missing
                    response = response.replace(". ", ".\n\n")

            return response.strip()

        except Exception as e:
            logger.error(f"Error in response post-processing: {e}")
            return response  # Return original response if post-processing fails

    def _generate_fallback_response(self, query: str, reason: str) -> str:
        """Generate helpful fallback response when RAG fails"""
        return f"""I apologize, but I encountered an issue generating a detailed response about your question: "{query}"

This might be because:
- The specific topic isn't covered in the provided context
- The question needs clarification
- I need more specific information from Andrew Ng's book

Could you try:
1. Asking about fundamental ML concepts (supervised learning, neural networks, etc.)
2. Being more specific about which aspect of ML you're interested in
3. Asking questions about model evaluation, training strategies, or ML principles

I'm here to help you learn machine learning from Andrew Ng's expertise!"""

    async def _update_memory(self, messages: List[Dict[str, str]], response: str):
        """Update conversation memory with new exchange"""
        try:
            if messages and response:
                # Add the latest exchange to memory
                last_message = messages[-1]
                if last_message.get("role") == "user":
                    self.memory.chat_memory.add_user_message(last_message["content"])
                self.memory.chat_memory.add_ai_message(response)

        except Exception as e:
            logger.warning(f"Failed to update conversation memory: {e}")

    async def get_conversation_summary(self) -> str:
        """Get summary of current conversation context"""
        try:
            if not self.memory.chat_memory.messages:
                return "No previous conversation context."

            # Summarize recent conversation
            recent_messages = self.memory.chat_memory.messages[-4:]  # Last 2 exchanges
            context_parts = []

            for i, message in enumerate(recent_messages):
                if isinstance(message, (HumanMessage, AIMessage)):
                    role = "User" if isinstance(message, HumanMessage) else "Assistant"
                    content = message.content[:100] + "..." if len(message.content) > 100 else message.content
                    context_parts.append(f"{role}: {content}")

            return "Recent conversation: " + " | ".join(context_parts)

        except Exception as e:
            logger.error(f"Failed to get conversation summary: {e}")
            return "Conversation context unavailable."