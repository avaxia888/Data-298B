"""
LLM Council Evaluation System for Neil deGrasse Tyson Chatbot
Uses 4 diverse judges to evaluate 12 models across 3 groups
Based on Karpathy's LLM Council architecture
"""

import json
import asyncio
import time
import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime
import numpy as np
import os
import re
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# Import necessary clients
from openai import OpenAI
import anthropic
import google.generativeai as genai
# Together import removed - now using DeepSeek and Mistral instead
import boto3

# Import existing services
from services.llm_client import LLMClient, load_models_config
from services.rag import RagService
from prompt_template import DEFAULT_SYSTEM_PROMPT
from utils import embed_query, retrieve_context

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global thread pool for async execution
executor = ThreadPoolExecutor(max_workers=10)

def clean_json_response(text: str) -> Dict[str, Any]:
    """
    Robust JSON extraction from LLM responses.
    Handles markdown code blocks, extra text, and various formats.
    """
    try:
        # First, try to parse as-is
        return json.loads(text)
    except:
        pass
    
    # Strip markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Find JSON object in text
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass
    
    # Fallback: return default scores
    logger.warning(f"Could not parse JSON from response: {text[:200]}")
    return {"score": 5, "reasoning": "Could not parse response"}

class CouncilJudge:
    """Base class for council judges"""
    def __init__(self, name: str):
        self.name = name
        
    async def evaluate(self, prompt: str) -> Dict[str, Any]:
        """Evaluate based on prompt, return scores and reasoning"""
        raise NotImplementedError

class GPT4Judge(CouncilJudge):
    """GPT-4o judge - Logic & Reasoning Chairman"""
    def __init__(self):
        super().__init__("GPT-4o")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def _evaluate_sync(self, prompt: str) -> Dict[str, Any]:
        """Synchronous evaluation for use with executor"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert judge evaluating AI responses. Return JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"GPT-4 judge error: {e}")
            return {"score": 5, "reasoning": f"Error: {str(e)}"}
    
    async def evaluate(self, prompt: str) -> Dict[str, Any]:
        """Async wrapper for evaluation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, self._evaluate_sync, prompt)

class ClaudeJudge(CouncilJudge):
    """Claude Sonnet 4.5 judge - Style & Nuance Expert"""
    def __init__(self):
        super().__init__("Claude-Sonnet-4.5")
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    def _evaluate_sync(self, prompt: str) -> Dict[str, Any]:
        """Synchronous evaluation for use with executor"""
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": prompt + "\n\nReturn JSON only."}],
                max_tokens=500,
                temperature=0.3
            )
            text = response.content[0].text
            return clean_json_response(text)
        except Exception as e:
            logger.error(f"Claude judge error: {e}")
            return {"score": 5, "reasoning": f"Error: {str(e)}"}
    
    async def evaluate(self, prompt: str) -> Dict[str, Any]:
        """Async wrapper for evaluation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, self._evaluate_sync, prompt)

class GeminiJudge(CouncilJudge):
    """Gemini 2.5 Pro judge - Context & Knowledge Specialist"""
    def __init__(self):
        super().__init__("Gemini-2.5-Pro")
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel('gemini-2.5-pro')
    
    def _evaluate_sync(self, prompt: str) -> Dict[str, Any]:
        """Synchronous evaluation for use with executor"""
        try:
            response = self.model.generate_content(
                prompt + "\n\nReturn JSON only with 'score' and 'reasoning' fields.",
                generation_config=genai.GenerationConfig(temperature=0.3)
            )
            return clean_json_response(response.text)
        except Exception as e:
            logger.error(f"Gemini judge error: {e}")
            return {"score": 5, "reasoning": f"Error: {str(e)}"}
    
    async def evaluate(self, prompt: str) -> Dict[str, Any]:
        """Async wrapper for evaluation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, self._evaluate_sync, prompt)

class DeepSeekJudge(CouncilJudge):
    """DeepSeek-V3 judge - Advanced reasoning model"""
    def __init__(self):
        super().__init__("DeepSeek-V3")
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )
    
    def _evaluate_sync(self, prompt: str) -> Dict[str, Any]:
        """Synchronous evaluation for use with executor"""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are an expert judge. Return JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            text = response.choices[0].message.content
            return clean_json_response(text)
        except Exception as e:
            logger.error(f"DeepSeek judge error: {e}")
            return {"score": 5, "reasoning": f"Error: {str(e)}"}
    
    async def evaluate(self, prompt: str) -> Dict[str, Any]:
        """Async wrapper for evaluation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, self._evaluate_sync, prompt)

class MistralJudge(CouncilJudge):
    """Mistral Large 2 judge - State-of-the-art open model"""
    def __init__(self):
        super().__init__("Mistral-Large-2")
        from mistralai import Mistral
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        self.last_call_time = 0
        self.min_interval = 2.0  # Minimum 2 seconds between calls
        self.retry_delay = 5.0  # Wait 5 seconds after rate limit error
    
    def _evaluate_sync(self, prompt: str) -> Dict[str, Any]:
        """Synchronous evaluation for use with executor with rate limiting"""
        import time as time_module
        
        # Enforce minimum interval between calls
        current_time = time_module.time()
        time_since_last = current_time - self.last_call_time
        if time_since_last < self.min_interval:
            time_module.sleep(self.min_interval - time_since_last)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.last_call_time = time_module.time()
                response = self.client.chat.complete(
                    model="mistral-large-latest",
                    messages=[
                        {"role": "system", "content": "You are an expert judge. Return JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=200
                )
                text = response.choices[0].message.content
                return clean_json_response(text)
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate" in error_str.lower():
                    logger.warning(f"Mistral rate limited, waiting {self.retry_delay}s before retry {attempt+1}/{max_retries}")
                    time_module.sleep(self.retry_delay)
                    self.retry_delay *= 1.5  # Exponential backoff
                    continue
                else:
                    logger.error(f"Mistral judge error: {e}")
                    return {"score": 5, "reasoning": f"Error: {str(e)}"}
        
        # If all retries failed
        return {"score": 5, "reasoning": "Rate limited after retries"}
    
    async def evaluate(self, prompt: str) -> Dict[str, Any]:
        """Async wrapper for evaluation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, self._evaluate_sync, prompt)

class LLMCouncil:
    """The council of 4 diverse judges (Mistral removed due to rate limiting)"""
    def __init__(self):
        self.judges = [
            GPT4Judge(),
            ClaudeJudge(),
            GeminiJudge(),
            DeepSeekJudge()
            # MistralJudge() removed due to severe rate limiting issues
        ]
        
        # Load Tyson reference examples from ground_truth_evaluation.json
        try:
            with open("ground_truth_evaluation.json", "r") as f:
                ground_truth = json.load(f)
                # Extract ALL Tyson's actual answers to use as style references
                self.tyson_references = [qa["answer"] for qa in ground_truth]
                logger.info(f"Loaded {len(self.tyson_references)} Tyson references")
        except Exception as e:
            logger.warning(f"Could not load Tyson references from ground_truth_evaluation.json: {e}")
            # Fallback to some default references
            self.tyson_references = [
                "I'm Neil deGrasse Tyson, your personal astrophysicist.",
                "love questions that begin with dude.",
                "that's not a question."
            ]
    
    async def evaluate_tyson_style(self, answer: str) -> Dict[str, Any]:
        """Council evaluates how much the answer sounds like Tyson"""
        # Randomly sample 5 references from all available references for each evaluation
        import random
        sample_refs = random.sample(self.tyson_references, min(5, len(self.tyson_references)))
        
        prompt = f"""
        Rate ONLY how much this answer sounds like Neil deGrasse Tyson (0-10).
        
        Reference examples of Tyson's actual style:
        {chr(10).join(f'- "{ref[:100]}..."' if len(ref) > 100 else f'- "{ref}"' for ref in sample_refs)}
        
        Answer to evaluate:
        "{answer}"
        
        Scoring criteria:
        - Vocabulary (cosmic, universe, atoms, etc.): /3 points
        - Enthusiasm and wonder: /2 points
        - Educational storytelling: /2 points
        - Humor and accessibility: /2 points
        - Signature phrases: /1 point
        
        Ignore scientific accuracy completely. Focus ONLY on voice/style.
        
        Return JSON: {{"score": 0-10, "reasoning": "brief explanation"}}
        """
        
        # Run all judges in parallel
        judge_tasks = [judge.evaluate(prompt) for judge in self.judges]
        judge_results = await asyncio.gather(*judge_tasks, return_exceptions=True)
        
        # Process results
        votes = []
        for judge, result in zip(self.judges, judge_results):
            if isinstance(result, Exception):
                logger.error(f"Judge {judge.name} failed: {result}")
                vote = {"score": 5, "reasoning": f"Error: {str(result)}"}
            else:
                vote = result
            
            votes.append({
                "judge": judge.name,
                "score": vote.get("score", 5),
                "reasoning": vote.get("reasoning", "")
            })
        
        # Calculate average
        avg_score = np.mean([v["score"] for v in votes])
        
        return {
            "average_score": float(avg_score),
            "individual_votes": votes
        }
    
    # Removed evaluate_accuracy - focusing only on Tyson style

class ModelEvaluator:
    """Evaluates all 12 models using the council"""
    def __init__(self):
        self.council = LLMCouncil()
        self.llm_client = LLMClient()
        self.rag_service = RagService()
        self.system_prompt = DEFAULT_SYSTEM_PROMPT.rstrip()
        
        # Define the 12 models in 3 groups based on actual models in models.json
        # TEMPORARY: Only testing RAG models as finetuned endpoints are unavailable
        self.model_groups = {
            # Commented out temporarily - endpoints unavailable
            # "group_A_finetuned_only": [
            #     {"key": "tyson-ft-gpt-4o-mini", "name": "GPT-4o-mini Fine-tuned", "type": "finetuned"},
            #     {"key": "llama3-ft-neil", "name": "Llama-3 8B Fine-tuned", "type": "finetuned"},
            #     {"key": "qwen-2.5-7b-merged-neil", "name": "Qwen-2.5 7B Fine-tuned", "type": "finetuned"},
            #     {"key": "gemma-3-ndtv3", "name": "Gemma-2 9B Fine-tuned", "type": "finetuned"}
            # ],
            "group_B_base_rag": [
                {"key": "rag-gpt-4o-mini", "name": "GPT-4o-mini + RAG", "type": "base_rag"},
                {"key": "rag-llama3-router", "name": "Llama-3 8B + RAG", "type": "base_rag"},
                {"key": "rag-qwen25-router", "name": "Qwen-2.5 7B + RAG", "type": "base_rag"},
                {"key": "rag-claude-3.5-haiku", "name": "Claude 3.5 Haiku + RAG", "type": "base_rag"}
            ],
            # Commented out temporarily - finetuned endpoints unavailable
            # "group_C_finetuned_rag": [
            #     {"key": "tyson-ft-gpt-4o-mini", "name": "GPT-4o-mini FT + RAG", "type": "finetuned_rag"},
            #     {"key": "llama3-ft-neil", "name": "Llama-3 8B FT + RAG", "type": "finetuned_rag"},
            #     {"key": "qwen-2.5-7b-merged-neil", "name": "Qwen-2.5 7B FT + RAG", "type": "finetuned_rag"},
            #     {"key": "gemma-3-ndtv3", "name": "Gemma-2 9B FT + RAG", "type": "finetuned_rag"}
            # ]
        }
        
        # Load actual model configurations
        self.models = load_models_config("models.json")
        
        # Load ground truth questions
        with open("ground_truth_evaluation.json", "r") as f:
            self.ground_truth = json.load(f)
    
    def generate_answer(self, model_config: Dict, question: str, retry_count: int = 2) -> Tuple[str, bool]:
        """
        Generate answer from a model based on its configuration
        
        Returns:
            Tuple of (answer, success_flag)
        """
        for attempt in range(retry_count + 1):
            try:
                model_type = model_config["type"]
                model_key = model_config["key"]
                
                # Find the actual model configuration
                actual_model = next((m for m in self.models if m.key == model_key), None)
                if not actual_model:
                    logger.error(f"Model {model_key} not found in models.json")
                    return f"Model configuration not found: {model_key}", False
                
                if model_type == "base_rag":
                    # Group B: Use RAG service for base models
                    answer, _ = self.rag_service.answer(
                        query=question,
                        history=[],
                        temperature=0.7,
                        endpoint=actual_model,  # Use endpoint instead of model_id
                        system_prompt=self.system_prompt
                    )
                    return answer, True
                
                elif model_type == "finetuned_rag":
                    # Group C: Finetuned models with manual RAG augmentation
                    try:
                        # Ensure RAG service is initialized
                        self.rag_service._ensure()
                        
                        # Get embedding using OpenAI client
                        qv = embed_query(self.rag_service._openai_client, question)
                        
                        # Retrieve context from Pinecone
                        chunks = retrieve_context(self.rag_service._pinecone_index, qv, top_k=5)
                        
                        # Build augmented prompt with context
                        context_block = "\n\n".join(chunks) if chunks else ""
                        augmented_system_prompt = f"{self.system_prompt}\n\nRelevant context:\n{context_block}" if chunks else self.system_prompt
                        
                        # Generate response with augmented context
                        messages = [
                            {"role": "system", "content": augmented_system_prompt},
                            {"role": "user", "content": question}
                        ]
                        
                        answer = self.llm_client.generate(
                            endpoint=actual_model,
                            prompt="",
                            parameters={"temperature": 0.7, "max_new_tokens": 256},
                            messages=messages,
                            system_prompt=None  # System prompt already in messages
                        )
                        return answer, True
                    except Exception as e:
                        logger.error(f"Error in manual RAG for {model_key}: {e}")
                        if attempt < retry_count:
                            logger.info(f"Retrying {model_key} (attempt {attempt + 2}/{retry_count + 1})...")
                            time.sleep(2)  # Wait before retry
                            continue
                        return f"Error in RAG after {retry_count + 1} attempts: {str(e)}", False
                    
                else:
                    # Group A: Finetuned models without RAG
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": question}
                    ]
                    
                    answer = self.llm_client.generate(
                        endpoint=actual_model,
                        prompt="",
                        parameters={"temperature": 0.7, "max_new_tokens": 256},
                        messages=messages,
                        system_prompt=None  # System prompt already in messages
                    )
                    return answer, True
                    
            except Exception as e:
                logger.error(f"Error generating answer for {model_config['name']} (attempt {attempt + 1}/{retry_count + 1}): {e}")
                if attempt < retry_count:
                    logger.info(f"Retrying {model_config['name']}...")
                    time.sleep(2)  # Wait before retry
                    continue
                return f"Error after {retry_count + 1} attempts: {str(e)}", False
        
        return f"Failed to generate answer after all retries", False
    
    async def evaluate_all_models(self) -> Dict[str, Any]:
        """Run the complete evaluation"""
        # Count total models from active groups
        total_models = sum(len(models) for models in self.model_groups.values())
        
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "questions_evaluated": len(self.ground_truth),
                "models_tested": total_models,
                "council_judges": 4  # Reduced from 5 - Mistral removed due to rate limiting
            },
            "group_results": {},
            "detailed_evaluations": [],
            "failed_models": {},  # Track which models failed
            "model_success_rates": {}  # Track success rate per model
        }
        
        # Process each question - FULL EVALUATION: All 16 questions
        for qa_idx, qa_pair in enumerate(self.ground_truth):
            question = qa_pair["question"]
            ground_truth = qa_pair["answer"]
            
            logger.info(f"Evaluating question {qa_idx + 1}/{len(self.ground_truth)}: {question[:50]}...")
            
            question_results = {
                "question": question,
                "ground_truth": ground_truth,
                "model_evaluations": {}
            }
            
            # Evaluate each group
            for group_name, models in self.model_groups.items():
                for model_config in models:
                    model_name = model_config['name']
                    model_key = f"{model_config['key']}_{group_name}"  # Unique key per group
                    
                    # Initialize success tracking if needed
                    if model_key not in results["model_success_rates"]:
                        results["model_success_rates"][model_key] = {"success": 0, "total": 0}
                    
                    results["model_success_rates"][model_key]["total"] += 1
                    
                    logger.info(f"  Testing {model_name}...")
                    
                    # Generate answer with error handling
                    answer, success = self.generate_answer(model_config, question)
                    
                    if not success:
                        # Track failed model
                        if model_key not in results["failed_models"]:
                            results["failed_models"][model_key] = {
                                "model_name": model_name,
                                "group": group_name,
                                "failures": []
                            }
                        results["failed_models"][model_key]["failures"].append({
                            "question": question[:50] + "...",
                            "error": answer
                        })
                        
                        # Store error result
                        question_results["model_evaluations"][model_key] = {
                            "model_name": model_name,
                            "group": group_name,
                            "answer": answer,  # Error message
                            "tyson_score": 0,
                            "error": True,
                            "tyson_votes": []
                        }
                        continue
                    
                    results["model_success_rates"][model_key]["success"] += 1
                    
                    # Get council evaluations (style only)
                    try:
                        tyson_eval = await self.council.evaluate_tyson_style(answer)
                    except Exception as e:
                        logger.error(f"Council evaluation failed for {model_name}: {e}")
                        tyson_eval = {"average_score": 0, "individual_votes": []}
                    
                    # Store results (style only)
                    question_results["model_evaluations"][model_key] = {
                        "model_name": model_name,
                        "group": group_name,
                        "answer": answer[:500],  # Truncate for storage
                        "tyson_score": tyson_eval["average_score"],
                        "error": False,
                        "tyson_votes": tyson_eval["individual_votes"]
                    }
            
            results["detailed_evaluations"].append(question_results)
        
        # Calculate group averages (excluding failed evaluations)
        for group_name in self.model_groups.keys():
            group_scores = []
            for eval in results["detailed_evaluations"]:
                for model_key, model_eval in eval["model_evaluations"].items():
                    if model_eval["group"] == group_name and not model_eval.get("error", False):
                        group_scores.append(model_eval["tyson_score"])
            
            if group_scores:
                results["group_results"][group_name] = {
                    "average_tyson_score": np.mean(group_scores)
                }
        
        # Determine winner based on Tyson style score
        results["winner"] = max(
            results["group_results"].items(),
            key=lambda x: x[1]["average_tyson_score"]
        )[0]
        
        return results
    
    def save_results(self, results: Dict, filename: str = "council_evaluation_results.json"):
        """Save results to JSON file"""
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {filename}")
    
    def print_summary(self, results: Dict):
        """Print a summary of results"""
        print("\n" + "="*60)
        print("COUNCIL EVALUATION RESULTS")
        print("="*60)
        
        # Print failed models if any
        if results.get("failed_models"):
            print("\n⚠️  FAILED MODELS:")
            for model_key, failure_info in results["failed_models"].items():
                success_rate = results["model_success_rates"][model_key]
                print(f"  - {failure_info['model_name']} ({failure_info['group']})")
                print(f"    Success rate: {success_rate['success']}/{success_rate['total']} ")
                print(f"    ({100 * success_rate['success'] / success_rate['total']:.1f}%)")
        
        for group_name, group_stats in results["group_results"].items():
            group_label = {
                "group_A_finetuned_only": "Group A (Fine-tuned Only)",
                "group_B_base_rag": "Group B (Base + RAG)",
                "group_C_finetuned_rag": "Group C (Fine-tuned + RAG)"
            }.get(group_name, group_name)
            
            print(f"\n{group_label}:")
            print(f"  Tyson Style Score: {group_stats['average_tyson_score']:.1f}/10")
        
        print(f"\n{'='*60}")
        print(f"WINNER: {results['winner']}")
        print("="*60)
        
        # Style Analysis
        print("\nSTYLE ANALYSIS:")
        
        if "group_A_finetuned_only" in results["group_results"]:
            a_tyson = results["group_results"]["group_A_finetuned_only"]["average_tyson_score"]
            print(f"✓ Group A (Fine-tuned): Style score {a_tyson:.1f}/10")
        
        if "group_B_base_rag" in results["group_results"]:
            b_tyson = results["group_results"]["group_B_base_rag"]["average_tyson_score"]
            print(f"✓ Group B (Base + RAG): Style score {b_tyson:.1f}/10")
        
        if "group_C_finetuned_rag" in results["group_results"]:
            c_tyson = results["group_results"]["group_C_finetuned_rag"]["average_tyson_score"]
            print(f"✓ Group C (Fine-tuned + RAG): Style score {c_tyson:.1f}/10")
            
            # Check if finetuning improves style
            if "group_A_finetuned_only" in results["group_results"] and "group_B_base_rag" in results["group_results"]:
                if a_tyson > b_tyson:
                    print("\n📈 Fine-tuning DOES improve Tyson style!")
                if c_tyson > b_tyson:
                    print("📊 Fine-tuned + RAG outperforms Base + RAG in style")

async def main():
    """Main execution function"""
    logger.info("Starting LLM Council Evaluation...")
    
    evaluator = ModelEvaluator()
    
    # Run evaluation
    results = await evaluator.evaluate_all_models()
    
    # Save results
    evaluator.save_results(results)
    
    # Print summary
    evaluator.print_summary(results)
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    asyncio.run(main())