import json
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import httpx
from dotenv import load_dotenv
from .prompts import EvaluationPrompts
from .metrics import EvaluationMetrics

@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    scores: Dict[str, float]
    overall_score: float
    strengths: List[str]
    weaknesses: List[str]
    suggestions: str
    raw_response: str

class LLMJudge:
    """LLM-as-judge evaluation system using GPT-5."""
    
    def __init__(self, model_name: str = "gpt-5", timeout: float = 60.0):
        """Initialize the LLM Judge.
        
        Args:
            model_name: The OpenAI model to use for evaluation (default: gpt-5)
            timeout: Request timeout in seconds
        """
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment variables")
        
        self.model_name = model_name
        self.timeout = timeout
        self.prompts = EvaluationPrompts()
        self.metrics = EvaluationMetrics()
        self.cache = {}  # Simple cache for evaluations
        
    def evaluate_response(
        self,
        query: str,
        response: str,
        context: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> EvaluationResult:
        """Evaluate a single model response.
        
        Args:
            query: The user's question
            response: The model's response
            context: Retrieved context for RAG models
            use_cache: Whether to use cached evaluations
            
        Returns:
            EvaluationResult with scores and feedback
        """
        # Check cache
        cache_key = f"{query}:{response[:100]}"
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        # Generate evaluation prompt
        eval_prompt = self.prompts.get_neil_evaluation_prompt(query, response, context)
        
        # Call GPT-4/5 for evaluation
        try:
            eval_response = self._call_judge_api(eval_prompt)
            result = self._parse_evaluation_response(eval_response)
            
            # Cache the result
            if use_cache:
                self.cache[cache_key] = result
                
            return result
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            import traceback
            traceback.print_exc()
            # Return default scores if evaluation fails
            return EvaluationResult(
                scores={
                    "scientific_accuracy": 0,
                    "style_authenticity": 0,
                    "clarity_accessibility": 0,
                    "engagement": 0,
                    "relevance": 0
                },
                overall_score=0,
                strengths=[],
                weaknesses=["Evaluation failed"],
                suggestions="Unable to evaluate",
                raw_response=str(e)
            )
    
    def compare_responses(
        self,
        query: str,
        response_a: str,
        response_b: str,
        model_a_name: str = "Model A",
        model_b_name: str = "Model B"
    ) -> Dict[str, Any]:
        """Compare two model responses.
        
        Args:
            query: The user's question
            response_a: First model's response
            response_b: Second model's response
            model_a_name: Name of first model
            model_b_name: Name of second model
            
        Returns:
            Comparison results dictionary
        """
        # Generate comparison prompt
        comp_prompt = self.prompts.get_comparative_evaluation_prompt(
            query, response_a, response_b, model_a_name, model_b_name
        )
        
        try:
            comp_response = self._call_judge_api(comp_prompt)
            return self._parse_json_response(comp_response)
        except Exception as e:
            print(f"Comparison error: {e}")
            return {
                "error": str(e),
                "model_a_scores": {"overall": 0},
                "model_b_scores": {"overall": 0},
                "preferred_model": "Unable to determine",
                "preference_reason": "Evaluation failed"
            }
    
    def check_factual_accuracy(self, query: str, response: str) -> Dict[str, Any]:
        """Check the factual accuracy of a response.
        
        Args:
            query: The user's question
            response: The response to fact-check
            
        Returns:
            Fact-checking results
        """
        fact_prompt = self.prompts.get_factual_accuracy_prompt(query, response)
        
        try:
            fact_response = self._call_judge_api(fact_prompt)
            return self._parse_json_response(fact_response)
        except Exception as e:
            print(f"Fact-checking error: {e}")
            return {
                "accuracy_score": 0,
                "factual_errors": ["Unable to verify"],
                "recommendation": "UNABLE TO EVALUATE"
            }
    
    def _call_judge_api(self, prompt: str) -> str:
        """Make API call to the judge model.
        
        Args:
            prompt: The evaluation prompt
            
        Returns:
            The judge model's response
        """
        messages = [
            {"role": "system", "content": "You are an expert evaluator for AI responses. Always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        # Try GPT-5 first, fallback to GPT-4 if it fails
        models_to_try = [self.model_name]
        if self.model_name == "gpt-5":
            models_to_try.append("gpt-4-turbo-preview")  # Fallback to GPT-4
        
        last_error = None
        for model in models_to_try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.3,  # Lower temperature for more consistent evaluation
                "max_tokens": 1000,
                # "response_format": {"type": "json_object"}  # Commented out for compatibility
            }
            
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json=payload
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    if model != self.model_name:
                        print(f"Note: Using {model} as fallback (GPT-5 not available)")
                    
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_error = e
                if model != models_to_try[-1]:
                    continue  # Try next model
        
        # If all models failed, raise the last error
        if last_error:
            raise last_error
        raise RuntimeError("Failed to get response from judge API")
    
    def _parse_evaluation_response(self, response: str) -> EvaluationResult:
        """Parse the evaluation response into an EvaluationResult.
        
        Args:
            response: The judge's JSON response
            
        Returns:
            Parsed EvaluationResult
        """
        try:
            # Clean up response - remove markdown code blocks if present
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]  # Remove ```json
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]  # Remove ```
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]  # Remove closing ```
            cleaned = cleaned.strip()
            
            data = json.loads(cleaned)
            return EvaluationResult(
                scores=data.get("scores", {}),
                overall_score=data.get("overall_score", 0),
                strengths=data.get("strengths", []),
                weaknesses=data.get("weaknesses", []),
                suggestions=data.get("suggestions", ""),
                raw_response=response
            )
        except json.JSONDecodeError as e:
            # Fallback if JSON parsing fails
            return EvaluationResult(
                scores={},
                overall_score=0,
                strengths=[],
                weaknesses=["Failed to parse evaluation"],
                suggestions="",
                raw_response=response
            )
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse a JSON response safely.
        
        Args:
            response: JSON string response
            
        Returns:
            Parsed dictionary or error dict
        """
        try:
            # Clean up response - remove markdown code blocks if present
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]  # Remove ```json
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]  # Remove ```
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]  # Remove closing ```
            cleaned = cleaned.strip()
            
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"error": "Failed to parse response", "raw": response}
    
    def batch_evaluate(
        self,
        conversations: List[Dict[str, str]],
        model_name: str = "Unknown Model"
    ) -> Dict[str, Any]:
        """Evaluate a batch of conversations.
        
        Args:
            conversations: List of conversation dicts with 'query' and 'response' keys
            model_name: Name of the model being evaluated
            
        Returns:
            Batch evaluation summary
        """
        results = []
        total_scores = {
            "scientific_accuracy": 0,
            "style_authenticity": 0,
            "clarity_accessibility": 0,
            "engagement": 0,
            "relevance": 0
        }
        
        for conv in conversations:
            result = self.evaluate_response(
                conv.get("query", ""),
                conv.get("response", ""),
                conv.get("context")
            )
            results.append(result)
            
            # Accumulate scores
            for key, value in result.scores.items():
                total_scores[key] = total_scores.get(key, 0) + value
        
        # Calculate averages
        num_conversations = len(conversations)
        if num_conversations > 0:
            avg_scores = {k: v / num_conversations for k, v in total_scores.items()}
            overall_avg = sum(avg_scores.values()) / len(avg_scores)
        else:
            avg_scores = total_scores
            overall_avg = 0
        
        return {
            "model_name": model_name,
            "num_evaluated": num_conversations,
            "average_scores": avg_scores,
            "overall_average": overall_avg,
            "individual_results": results
        }