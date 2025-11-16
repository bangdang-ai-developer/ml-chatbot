"""
Vietnamese-Optimized Prompt Templates for RAG
Advanced prompt engineering for Vietnamese ML/AI responses
"""

from typing import List, Dict, Any, Optional
import logging
from .query_expansion_service import QueryExpansionService
from .translation_service import get_translation_service
from ..core.exceptions import ChatbotException

logger = logging.getLogger(__name__)

class PromptService:
    """
    Vietnamese-optimized prompt template service for RAG responses
    """

    def __init__(self, query_expansion_service: QueryExpansionService = None):
        self.query_service = query_expansion_service or QueryExpansionService()
        self.translation_service = get_translation_service()

        # Response style templates
        self.response_styles = {
            'formal_academic': {
                'tone': 'academic',
                'language': 'vietnamese',
                'structure': 'definition -> explanation -> examples -> applications',
                'complexity': 'high'
            },
            'practical_tutorial': {
                'tone': 'instructional',
                'language': 'vietnamese',
                'structure': 'problem -> solution -> step-by-step -> code examples',
                'complexity': 'medium'
            },
            'casual_explanation': {
                'tone': 'conversational',
                'language': 'vietnamese',
                'structure': 'simple explanation -> real-world analogy -> key points',
                'complexity': 'low'
            },
            'technical_comparison': {
                'tone': 'analytical',
                'language': 'vietnamese',
                'structure': 'overview -> detailed comparison -> pros/cons -> recommendations',
                'complexity': 'high'
            }
        }

    async def translate_technical_terms(self, text: str, to_vietnamese: bool = True) -> str:
        """
        Translate technical terms between English and Vietnamese using dynamic translation service
        """
        try:
            if not text or not text.strip():
                return text

            # Use the dynamic translation service
            translated = await self.translation_service.translate_text(text, to_vietnamese)

            logger.debug(f"Translated text from {self._detect_language(text)} to {'Vietnamese' if to_vietnamese else 'English'}")
            return translated

        except Exception as e:
            logger.error(f"Error translating technical terms: {e}")
            # Return original text as fallback
            return text

    def _detect_language(self, text: str) -> str:
        """Simple language detection helper"""
        try:
            # Check for Vietnamese characters
            vietnamese_chars = set('áàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ')
            if any(char in vietnamese_chars for char in text.lower()):
                return "Vietnamese"
            return "English"
        except Exception:
            return "Unknown"

    def create_vietnamese_system_prompt(self, query_analysis: Dict[str, Any]) -> str:
        """
        Create system prompt optimized for Vietnamese queries
        """
        try:
            is_vietnamese = query_analysis['language_analysis']['is_vietnamese']
            intent = query_analysis['intent_analysis']['primary_intent']
            complexity = query_analysis['complexity']['complexity_score']

            if is_vietnamese:
                base_prompt = """Bạn là một người hướng dẫn về Machine Learning và AI, luôn trò chuyện với người dùng Việt một cách thân thiện, gần gũi nhưng vẫn đảm bảo tính chính xác cao.

Phong cách trò chuyện:
1. Nói chuyện như một người bạn đồng hành, người hướng dẫn giàu kinh nghiệm
2. Dùng ngôn ngữ tự nhiên, đời thường như người Việt nói chuyện hàng ngày
3. Khi giải thích thuật ngữ khó, hãy ví von với những thứ quen thuộc ở Việt Nam
4. Sử dụng các từ nối như "à", "nói cách khác", "mình có thể giải thích thế này" cho tự nhiên
5. Thỉnh thoảng dùng các từ cảm thán như "Đúng vậy!", "Thú vị vị trí!", "Điều này quan trọng lắm nhé!"
6. Kết thúc bằng những câu hỏi mở hoặc gợi ý để người dùng tiếp tục khám phá"""
            else:
                base_prompt = """You are an AI assistant specializing in Machine Learning and Artificial Intelligence with Vietnamese language support.

Response guidelines:
1. Respond primarily in English but include Vietnamese examples where helpful
2. Explain technical concepts with Vietnamese terminology when relevant
3. Provide practical examples that Vietnamese learners can relate to
4. Maintain professional and educational tone"""

            # Add intent-specific instructions
            if is_vietnamese:
                intent_instructions = self._get_vietnamese_intent_instructions(intent)
            else:
                intent_instructions = self._get_english_intent_instructions(intent)

            # Add complexity-specific instructions
            complexity_instructions = self._get_complexity_instructions(complexity, is_vietnamese)

            # Combine all instructions
            system_prompt = f"""{base_prompt}

{intent_instructions}

{complexity_instructions}

Lưu ý quan trọng: Hãy là một người bạn đồng hành thực sự, chia sẻ kiến thức một cách chân thành và nhiệt tình!"""

            return system_prompt

        except Exception as e:
            logger.error(f"Error creating system prompt: {e}")
            return "Bạn là một trợ lý AI chuyên về Học Máy và Trí tuệ Nhân tạo."

    def _get_vietnamese_intent_instructions(self, intent: str) -> str:
        """
        Get Vietnamese-specific instructions based on query intent
        """
        instructions = {
            'definition': """Khi người dùng hỏi định nghĩa:
- Bắt đầu bằng cách nói: "À, để mình giải thích [khái niệm] một cách đơn giản nhé..."
- Dùng ví von quen thuộc: "Giống như..." ví dụ như cooking, lái xe, học bài...
- Trả lời như đang nói chuyện: "Nói dễ hiểu thì...", "Hiểu nôm na là..."
- Sau khi giải thích xong, hỏi thêm: "Bạn muốn mình giải thích sâu hơn phần nào không?""",

            'comparison': """Khi người dùng muốn so sánh:
- Bắt đầu bằng: "Ôi, đây là câu hỏi hay lắm! Mình phân tích giúp bạn nhé..."
- Dùng cách so sánh đời thường: "A giống như..., trong khi B lại giống như..."
- Nói như đang tư vấn: "Tùy vào trường hợp của bạn nhé..."
- Kết thúc bằng gợi ý: "Theo mình thì bạn nên chọn..., bạn nghĩ sao?""",

            'how_to': """Khi người dùng hỏi cách làm:
- Bắt đầu bằng: "Được chứ! Mình hướng dẫn bạn từng bước nhé..."
- Dùng cách nói thân mật: "Bước đầu tiên mình làm là...", "Tiếp theo nè..."
- Cho thấy sự đồng cảm: "Mình cũng từng gặp trường hợp này..."
- Kết thúc bằng động viên: "Bạn thử làm xem, có gì mình giúp tiếp nhé!""",

            'examples': """Khi người dùng muốn ví dụ:
- Bắt đầu nhiệt tình: "À có nhiều ví dụ thú vị lắm! Mình kể bạn nghe nhé..."
- Dùng ví dụ gần gũi: "Giống như khi bạn...", "Tưởng tượng bạn đang..."
- Liên hệ Việt Nam: "Ở Việt Nam mình thì thường thấy trong..."
- Kết thúc mở: "Bạn thấy ví dụ nào hữu ích nhất?""",

            'advantages_disadvantages': """Khi hỏi ưu/nhược điểm:
- Bắt đầu cân nhắc: "Đây là câu hỏi thực tế! Mình phân tích kỹ giúp bạn..."
- Nói như kinh nghiệm: "Theo mình trải nghiệm thì..."
- Thừa nhận cả hai mặt: "Đúng là nó tốt về..., nhưng hơi kém ở..."
- Cho lời khuyên chân thành: "Nếu mình là bạn thì mình sẽ cân nhắc..."""
        }

        return instructions.get(intent, "Trả lời một cách rõ ràng, có cấu trúc và đầy đủ thông tin.")

    def _get_english_intent_instructions(self, intent: str) -> str:
        """
        Get English instructions with Vietnamese support
        """
        instructions = {
            'definition': """For definition questions:
- Start with a concise, accurate definition
- Explain the meaning and importance
- Provide relevant ML/AI examples
- Include Vietnamese terminology where helpful""",

            'comparison': """For comparison questions:
- Create clear comparison tables
- Analyze pros and cons of each approach
- Provide context-specific recommendations
- Use concrete examples for illustration""",

            'how_to': """For how-to questions:
- Provide detailed step-by-step instructions
- Include both theory and practice
- Recommend appropriate tools and libraries
- Mention common pitfalls and solutions""",

            'examples': """For example questions:
- Provide diverse examples from basic to advanced
- Explain why each example is important
- Show practical applications
- Include Vietnamese context when relevant"""
        }

        return instructions.get(intent, "Provide clear, structured, and comprehensive responses.")

    def _get_complexity_instructions(self, complexity: float, is_vietnamese: bool) -> str:
        """
        Get complexity-specific instructions
        """
        if complexity > 0.7:
            if is_vietnamese:
                return """Ô, đây là câu hỏi chuyên sâu đấy! Mình giải thích kỹ nhé:
- Đi sâu vào khía cạnh lý thuyết một cách dễ hiểu
- Giải thích công thức toán học như đang dạy học
- Kể thêm về nghiên cứu mới nhất cho bạn biết
- Dự đoán xu hướng tương lai cho thú vị"""
            else:
                return """This is a complex question requiring detailed response:
- Provide in-depth theoretical analysis
- Include mathematical formulations when necessary
- Mention current research and developments
- Discuss future directions"""
        elif complexity < 0.3:
            if is_vietnamese:
                return """À câu này đơn giản, mình trả lời nhanh nhé:
- Đi thẳng vào vấn đề, không vòng vo
- Dùng từ ngữ đời thường dễ hiểu
- Nói ngắn gọn nhưng đủ ý"""
            else:
                return """This question needs a direct response:
- Be concise and to the point
- Use simple, clear language
- Focus on the most important information"""
        else:
            if is_vietnamese:
                return """Câu hỏi này ở mức độ vừa phải, mình cân bằng nhé:
- Giải thích đủ hiểu nhưng không quá phức tạp
- Nói theo luồng logic cho dễ theo dõi
- Cho ví dụ thực tế để bạn hình dung"""
            else:
                return """Balance detail and conciseness:
- Provide sufficient information without being overwhelming
- Use logical structure for easy following
- Include illustrative examples"""

    def create_rag_prompt(
        self,
        query: str,
        context: List[str],
        query_analysis: Dict[str, Any],
        max_context_length: int = 4000
    ) -> str:
        """
        Create RAG prompt with Vietnamese optimization
        """
        try:
            is_vietnamese = query_analysis['language_analysis']['is_vietnamese']

            # Prepare context
            context_text = self._prepare_context(context, max_context_length, is_vietnamese)

            # Get system prompt
            system_prompt = self.create_vietnamese_system_prompt(query_analysis)

            # Create the main prompt
            if is_vietnamese:
                prompt = f"""{system_prompt}

Bạn thân mến, mình đang có một câu hỏi cần giúp đỡ đây:

CÂU HỎI CỦA BẠN:
{query}

Thông tin mình tìm thấy trong tài liệu học thuật:
{context_text}

Mình hãy trả lời thật tự nhiên như đang nói chuyện với bạn:
- Bắt đầu bằng lời chào thân thiện
- Giải thích từ từ, dễ hiểu, dùng ví von gần gũi
- Dựa vào tài liệu nhưng diễn đạt theo cách mình hiểu
- Nếu tài liệu không đủ, mình chia sẻ thêm từ kiến thức của mình
- Dùng từ ngữ đời thường, đôi khi hỏi ngược lại để chắc bạn hiểu
- Kết thúc bằng gợi ý để tiếp tục trao đổi

Bây giờ mình trả lời nhé:"""
            else:
                prompt = f"""{system_prompt}

USER QUESTION:
{query}

REFERENCE CONTEXT:
{context_text}

RESPONSE GUIDELINES:
1. Base your answer on the provided context
2. If context is insufficient, clearly indicate what's missing
3. Cite important points from the source material
4. Supplement with expert knowledge when necessary
5. Include Vietnamese terminology where helpful

RESPONSE:"""

            return prompt

        except Exception as e:
            logger.error(f"Error creating RAG prompt: {e}")
            raise ChatbotException(f"Prompt creation failed: {e}")

    def _prepare_context(self, context: List[str], max_length: int, is_vietnamese: bool) -> str:
        """
        Prepare and format context for prompt
        """
        try:
            if not context:
                return "Không có tài liệu tham khảo." if is_vietnamese else "No reference material available."

            # Join context pieces
            context_text = "\n\n---\n\n".join(context)

            # Truncate if too long
            if len(context_text) > max_length:
                if is_vietnamese:
                    context_text = context_text[:max_length] + "...[đã cắt bớt]"
                else:
                    context_text = context_text[:max_length] + "...[truncated]"

            # Add context header
            if is_vietnamese:
                return f"TÀI LIỆU THAM KHẢO:\n{context_text}"
            else:
                return f"REFERENCE MATERIAL:\n{context_text}"

        except Exception as e:
            logger.error(f"Error preparing context: {e}")
            return "Lỗi khi chuẩn bị tài liệu tham khảo." if is_vietnamese else "Error preparing reference material."

    def create_followup_prompt(
        self,
        original_query: str,
        original_response: str,
        followup_query: str,
        query_analysis: Dict[str, Any]
    ) -> str:
        """
        Create prompt for follow-up questions with context
        """
        try:
            is_vietnamese = query_analysis['language_analysis']['is_vietnamese']

            if is_vietnamese:
                return f"""Bạn đã trả lời câu hỏi trước đó của người dùng. Bây giờ họ có câu hỏi tiếp theo:

CÂU HỎI TRƯỚC ĐÓ:
{original_query}

TRẢ LỜI TRƯỚC ĐÓ:
{original_response}

CÂU HỎI TIẾP THEO:
{followup_query}

Hãy trả lời câu hỏi tiếp theo có tính đến ngữ cảnh của cuộc trò chuyện. Nếu câu hỏi liên quan đến câu trả lời trước, hãy tham chiếu lại khi cần thiết."""
            else:
                return f"""You have previously answered the user's question. Now they have a follow-up:

PREVIOUS QUESTION:
{original_query}

PREVIOUS RESPONSE:
{original_response}

FOLLOW-UP QUESTION:
{followup_query}

Please answer the follow-up question considering the conversation context. Reference the previous response if relevant."""

        except Exception as e:
            logger.error(f"Error creating follow-up prompt: {e}")
            return self.create_rag_prompt(followup_query, [], query_analysis)

    def add_vietnamese_examples(self, response_text: str, query: str) -> str:
        """
        Enhance response with Vietnamese-specific examples
        """
        try:
            # Check if response already has sufficient examples
            if "ví dụ" in response_text.lower() or "example" in response_text.lower():
                return response_text

            # Vietnamese ML context examples
            vietnamese_examples = {
                'e-commerce': 'các sàn thương mại điện tử Việt Nam như Tiki, Shopee, Lazada',
                'fintech': 'công ty fintech như MoMo, ZaloPay, Viettel Money',
                'agriculture': 'nông nghiệp công nghệ cao tại Đồng bằng sông Cửu Long',
                'healthcare': 'bệnh viện Việt Đức, Bệnh viện Chợ Rẫy áp dụng AI',
                'education': 'nền tảng học trực tuyến như VUIHOC, Kyna'
            }

            # Add relevant examples based on query content
            enhanced_response = response_text
            for context, example in vietnamese_examples.items():
                if context in query.lower() and example not in response_text:
                    if "ví dụ" in enhanced_response:
                        enhanced_response += f", {example}"
                    else:
                        enhanced_response += f"\n\nVí dụ trong bối cảnh Việt Nam: {example}."

            return enhanced_response

        except Exception as e:
            logger.error(f"Error adding Vietnamese examples: {e}")
            return response_text

    def get_response_style_config(self, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine optimal response style based on query analysis
        """
        try:
            complexity = query_analysis['complexity']['complexity_score']
            is_vietnamese = query_analysis['language_analysis']['is_vietnamese']
            intent = query_analysis['intent_analysis']['primary_intent']

            # Determine style based on intent and complexity
            if intent == 'how_to':
                style = 'practical_tutorial'
            elif intent == 'comparison':
                style = 'technical_comparison'
            elif complexity > 0.7:
                style = 'formal_academic'
            else:
                style = 'casual_explanation'

            config = self.response_styles[style].copy()
            config['is_vietnamese'] = is_vietnamese

            return config

        except Exception as e:
            logger.error(f"Error determining response style: {e}")
            return self.response_styles['casual_explanation']

    def get_contextual_hints(self, query_analysis: Dict[str, Any]) -> List[str]:
        """
        Generate contextual hints for better response generation based on query analysis
        """
        try:
            hints = []

            # Language-specific hints
            is_vietnamese = query_analysis.get('language_analysis', {}).get('is_vietnamese', False)
            if is_vietnamese:
                hints.append("Use natural Vietnamese language with appropriate technical terminology")
                hints.append("Consider Vietnamese cultural and educational context")
            else:
                hints.append("Use clear English with Vietnamese term explanations where helpful")

            # Intent-specific hints
            intent = query_analysis.get('intent_analysis', {}).get('primary_intent', 'general')
            if intent == 'definition':
                hints.append("Start with clear definition, then provide examples")
            elif intent == 'comparison':
                hints.append("Use structured comparison with clear criteria")
            elif intent == 'how_to':
                hints.append("Provide step-by-step instructions with practical examples")
            elif intent == 'examples':
                hints.append("Include diverse real-world examples and use cases")

            # Complexity-based hints
            complexity = query_analysis.get('complexity', {}).get('complexity_score', 0.5)
            if complexity > 0.7:
                hints.append("Include theoretical background and advanced concepts")
                hints.append("Mention current research and future directions")
            elif complexity < 0.3:
                hints.append("Keep explanations simple and accessible")
                hints.append("Focus on practical applications")
            else:
                hints.append("Balance theory with practical examples")

            # Domain-specific hints for ML/AI
            hints.append("Include relevant Machine Learning terminology")
            hints.append("Reference practical ML applications when possible")

            return hints

        except Exception as e:
            logger.error(f"Error generating contextual hints: {e}")
            return ["Provide helpful, accurate information"]