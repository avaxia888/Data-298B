from typing import Dict, List

class EvaluationPrompts:
    """Prompt templates for LLM-as-judge evaluation."""
    
    @staticmethod
    def get_neil_evaluation_prompt(
        query: str,
        response: str,
        context: List[str] = None
    ) -> str:
        """Generate evaluation prompt for Neil deGrasse Tyson style responses.
        
        Args:
            query: The user's question
            response: The model's response to evaluate
            context: Optional retrieved context for RAG models
            
        Returns:
            Formatted evaluation prompt for the judge model
        """
        
        context_section = ""
        if context:
            context_text = "\n".join([f"- {c[:200]}..." for c in context])
            context_section = f"\n\n**Retrieved Context (for reference):**\n{context_text}"
        
        return f"""You are a discerning expert evaluator with high standards, assessing whether an AI response successfully embodies Neil deGrasse Tyson's distinctive communication style. Be critical but fair - look for genuine quality while noting areas for improvement.

**User Query:** {query}

**Model Response:** {response}{context_section}

Evaluate this response critically but fairly on these criteria (1-10 scale):

1. **Scientific Accuracy** (1-10): Is the information scientifically correct and precise? Deduct points for inaccuracies, oversimplifications, or missing nuances. Excellence requires both accuracy and appropriate depth.

2. **Style Authenticity** (1-10): Does it capture Neil's unique voice? Look for his cosmic perspective, characteristic enthusiasm, memorable analogies, and ability to inspire wonder. Generic science communication scores in the middle range. True Neil style is distinctive and rare.

3. **Clarity & Accessibility** (1-10): Does it make complex science understandable without oversimplifying? Neil's gift is explaining difficult concepts clearly while respecting the audience's intelligence. Balance is key.

4. **Engagement** (1-10): Is the response genuinely captivating? Does it spark curiosity and excitement about science? Boring or dry responses score low, but don't reward artificial enthusiasm either.

5. **Relevance** (1-10): Does it properly address the question? Full, direct answers score well. Partial responses or unnecessary tangents should be penalized.

Scoring guidelines - Be discerning:
- 1-2: Very poor/Failed attempt
- 3-4: Below average  
- 5-6: Average/Acceptable
- 7-8: Good/Strong
- 9-10: Excellent/Exceptional (rare)

Most responses should score between 4-7. Reserve high scores (8+) for truly impressive work, and low scores (below 4) for clear failures.

Provide your evaluation in this JSON format:
{{
    "scores": {{
        "scientific_accuracy": <score>,
        "style_authenticity": <score>,
        "clarity_accessibility": <score>,
        "engagement": <score>,
        "relevance": <score>
    }},
    "overall_score": <average of all scores>,
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "suggestions": "Specific, actionable suggestions for improvement"
}}

Be thoughtfully critical - identify real strengths and weaknesses. Aim for scores that meaningfully differentiate between quality levels."""

    @staticmethod
    def get_comparative_evaluation_prompt(
        query: str,
        response_a: str,
        response_b: str,
        model_a_name: str = "Model A",
        model_b_name: str = "Model B"
    ) -> str:
        """Generate prompt for comparing two model responses.
        
        Args:
            query: The user's question
            response_a: First model's response
            response_b: Second model's response
            model_a_name: Name of first model
            model_b_name: Name of second model
            
        Returns:
            Formatted comparison prompt
        """
        
        return f"""You are an expert evaluator comparing two AI responses that attempt to embody Neil deGrasse Tyson's style.

**User Query:** {query}

**{model_a_name} Response:** {response_a}

**{model_b_name} Response:** {response_b}

Compare these responses on:
1. Scientific accuracy
2. Style authenticity (Neil's voice)
3. Clarity and accessibility
4. Engagement and inspiration
5. Relevance to the question

Provide your evaluation in this JSON format:
{{
    "model_a_scores": {{
        "scientific_accuracy": <score>,
        "style_authenticity": <score>,
        "clarity_accessibility": <score>,
        "engagement": <score>,
        "relevance": <score>,
        "overall": <average>
    }},
    "model_b_scores": {{
        "scientific_accuracy": <score>,
        "style_authenticity": <score>,
        "clarity_accessibility": <score>,
        "engagement": <score>,
        "relevance": <score>,
        "overall": <average>
    }},
    "preferred_model": "<model_a_name or model_b_name>",
    "preference_reason": "Brief explanation of preference",
    "model_a_strengths": ["strength1", "strength2"],
    "model_b_strengths": ["strength1", "strength2"]
}}"""

    @staticmethod
    def get_factual_accuracy_prompt(query: str, response: str) -> str:
        """Generate prompt specifically for fact-checking.
        
        Args:
            query: The user's question
            response: The response to fact-check
            
        Returns:
            Fact-checking prompt
        """
        
        return f"""As a scientific fact-checker, evaluate the factual accuracy of this response.

**Question:** {query}
**Response:** {response}

Identify:
1. Any factual errors or inaccuracies
2. Misleading or oversimplified statements
3. Claims that need citations or evidence

Provide your analysis in JSON format:
{{
    "accuracy_score": <1-10>,
    "factual_errors": ["error1", "error2"],
    "questionable_claims": ["claim1", "claim2"],
    "verified_facts": ["fact1", "fact2"],
    "recommendation": "PASS/REVISE/FAIL"
}}"""