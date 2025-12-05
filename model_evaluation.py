"""
Model Evaluation Script for Neil deGrasse Tyson Chatbot
Evaluates base, RAG, and fine-tuned models against ground truth answers
"""

import json
import os
import time
from typing import Dict, List, Tuple, Any
from datetime import datetime
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import boto3
from services.llm_client import LLMClient, load_models_config
from services.rag import RagService
from prompt_template import DEFAULT_SYSTEM_PROMPT
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import logging
from utils import embed_query, retrieve_context, evaluate_retrieval

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self):
        """Initialize the evaluator with necessary services"""
        load_dotenv()
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.llm_client = LLMClient()
        self.rag_service = RagService()
        self.models = load_models_config("models.json")
        self.ground_truth = self._load_ground_truth()
        self.system_prompt = DEFAULT_SYSTEM_PROMPT.rstrip()
        
    def _load_ground_truth(self) -> List[Dict]:
        """Load ground truth Q&A pairs"""
        with open("ground_truth_evaluation.json", "r", encoding="utf-8") as f:
            return json.load(f)
    
    def compute_embedding(self, text: str) -> List[float]:
        """
        Compute text embedding using OpenAI text-embedding-3-small
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error computing embedding: {e}")
            return [0.0] * 1536  # Return zero vector on error
    
    def compute_cosine_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts using embeddings
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Get embeddings for both texts
            emb1 = self.compute_embedding(text1)
            emb2 = self.compute_embedding(text2)
            
            # Compute cosine similarity
            similarity = cosine_similarity(
                np.array(emb1).reshape(1, -1),
                np.array(emb2).reshape(1, -1)
            )[0][0]
            
            return float(similarity)
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def generate_response(self, model_config: Any, question: str) -> Tuple[str, float]:
        """
        Generate response from a model based on its configuration
        
        Args:
            model_config: Model configuration object
            question: Question to ask
            
        Returns:
            Tuple of (response_text, response_time_ms)
        """
        start_time = time.time()
        response_text = ""
        
        try:
            if model_config.mode == "rag":
                # RAG model - use RagService
                response_text, _ = self.rag_service.answer(
                    query=question,
                    history=[],
                    temperature=0.7,
                    endpoint=model_config,
                    system_prompt=self.system_prompt
                )
            elif model_config.mode in ["openai", "huggingface", "bedrock"]:
                # Direct model call
                
                # Check if this is a Gemma model (doesn't support system messages)
                if "gemma" in model_config.key.lower():
                    # For Gemma, combine system prompt with user message
                    combined_prompt = f"{self.system_prompt}\n\nUser: {question}"
                    messages = [{"role": "user", "content": combined_prompt}]
                    system_prompt_to_use = None
                else:
                    # Standard message format for other models
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": question}
                    ]
                    system_prompt_to_use = self.system_prompt
                
                if model_config.mode == "bedrock":
                    # Special handling for Bedrock/Claude
                    response_text = self._call_bedrock_model(model_config, question)
                else:
                    # OpenAI or HuggingFace models
                    response_text = self.llm_client.generate(
                        endpoint=model_config,
                        prompt="",
                        parameters={"temperature": 0.7, "max_new_tokens": 256},
                        messages=messages,
                        system_prompt=system_prompt_to_use
                    )
            else:
                logger.error(f"Unknown model mode: {model_config.mode}")
                response_text = "Error: Unknown model mode"
                
        except Exception as e:
            logger.error(f"Error generating response for {model_config.key}: {e}")
            response_text = f"Error: {str(e)}"
        
        response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        return response_text, response_time
    
    def generate_response_with_rag(self, model_config: Any, question: str) -> Tuple[str, float]:
        """
        Generate response from a finetuned model with manual RAG augmentation
        
        Args:
            model_config: Model configuration object (finetuned model)
            question: Question to ask
            
        Returns:
            Tuple of (response_text, response_time_ms)
        """
        start_time = time.time()
        response_text = ""
        
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
            
            # Check if this is a Gemma model (doesn't support system messages)
            if "gemma" in model_config.key.lower():
                # For Gemma, combine augmented system prompt with user message
                combined_prompt = f"{augmented_system_prompt}\n\nUser: {question}"
                messages = [{"role": "user", "content": combined_prompt}]
            else:
                # Standard message format for other models
                messages = [
                    {"role": "system", "content": augmented_system_prompt},
                    {"role": "user", "content": question}
                ]
            
            response_text = self.llm_client.generate(
                endpoint=model_config,
                prompt="",
                parameters={"temperature": 0.7, "max_new_tokens": 256},
                messages=messages,
                system_prompt=None  # System prompt already in messages
            )
            
        except Exception as e:
            logger.error(f"Error generating RAG response for finetuned model {model_config.key}: {e}")
            response_text = f"Error: {str(e)}"
        
        response_time = (time.time() - start_time) * 1000
        return response_text, response_time
    
    def _call_bedrock_model(self, model_config: Any, question: str) -> str:
        """
        Special handler for AWS Bedrock models (Claude)
        
        Args:
            model_config: Model configuration
            question: Question to ask
            
        Returns:
            Response text
        """
        try:
            bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
            
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 256,
                "temperature": 0.7,
                "system": self.system_prompt,
                "messages": [
                    {"role": "user", "content": question}
                ]
            }
            
            response = bedrock.invoke_model(
                modelId=model_config.url,
                body=json.dumps(payload),
                contentType='application/json',
                accept='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            return response_body.get('content', [{}])[0].get('text', '')
        except Exception as e:
            logger.error(f"Bedrock API error: {e}")
            return f"Error calling Bedrock: {str(e)}"
    
    def evaluate_model(self, model_config: Any, use_rag: bool = False, category: str = None) -> Dict[str, Any]:
        """
        Evaluate a single model on all ground truth questions
        
        Args:
            model_config: Model configuration
            use_rag: Whether to use RAG augmentation for finetuned models
            category: Category to assign to this evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        eval_name = f"{model_config.name} ({category})" if category else model_config.name
        logger.info(f"Evaluating model: {eval_name}")
        
        results = {
            "key": model_config.key,
            "name": model_config.name,
            "category": category or getattr(model_config, 'category', 'unknown'),
            "responses": [],
            "average_cosine_similarity": 0.0,
            "average_response_time": 0.0
        }
        
        total_similarity = 0.0
        total_time = 0.0
        
        for qa_pair in tqdm(self.ground_truth, desc=f"Testing {eval_name}"):
            question = qa_pair["question"]
            ground_truth = qa_pair["answer"]
            
            # Generate response based on configuration
            if use_rag and model_config.mode != "rag":
                # Use manual RAG for finetuned models
                response, response_time = self.generate_response_with_rag(model_config, question)
            else:
                # Use standard generation (either built-in RAG or direct)
                response, response_time = self.generate_response(model_config, question)
            
            # Compute similarity
            similarity = self.compute_cosine_similarity(response, ground_truth)
            
            # Store results
            results["responses"].append({
                "question": question,
                "generated_answer": response,
                "ground_truth": ground_truth,
                "cosine_similarity": similarity,
                "response_time_ms": response_time
            })
            
            total_similarity += similarity
            total_time += response_time
        
        # Compute averages
        num_questions = len(self.ground_truth)
        results["average_cosine_similarity"] = total_similarity / num_questions if num_questions > 0 else 0
        results["average_response_time"] = total_time / num_questions if num_questions > 0 else 0
        
        return results
    
    def evaluate_all_models(self) -> Dict[str, Any]:
        """
        Evaluate all models in three configurations:
        1. Base models with built-in RAG
        2. Finetuned models without RAG
        3. Finetuned models with manual RAG
        
        Returns:
            Complete evaluation results dictionary
        """
        evaluation_results = {
            "timestamp": datetime.now().isoformat(),
            "models": {},
            "summary": {
                "base_rag": {"models": [], "avg_similarity": 0.0, "avg_time": 0.0},
                "finetuned": {"models": [], "avg_similarity": 0.0, "avg_time": 0.0},
                "finetuned_rag": {"models": [], "avg_similarity": 0.0, "avg_time": 0.0}
            }
        }
        
        # Categorize models
        rag_models = []
        finetuned_models = []
        
        for model_config in self.models:
            mode = (getattr(model_config, "mode", "") or "").lower()
            if mode == "rag":
                rag_models.append(model_config)
            elif mode in ["openai", "huggingface", "bedrock"]:
                finetuned_models.append(model_config)
        
        logger.info(f"Found {len(rag_models)} RAG models and {len(finetuned_models)} finetuned models")
        
        # 1. Evaluate base models with built-in RAG
        for model_config in rag_models:
            try:
                logger.info(f"Evaluating base RAG model: {model_config.key}")
                model_results = self.evaluate_model(model_config, use_rag=False, category="base_rag")
                result_key = f"{model_config.key}_base_rag"
                evaluation_results["models"][result_key] = model_results
                evaluation_results["summary"]["base_rag"]["models"].append(result_key)
            except Exception as e:
                logger.error(f"Error evaluating base RAG model {model_config.key}: {e}")
        
        # 2. Evaluate finetuned models without RAG
        for model_config in finetuned_models:
            try:
                logger.info(f"Evaluating finetuned model without RAG: {model_config.key}")
                model_results = self.evaluate_model(model_config, use_rag=False, category="finetuned")
                result_key = f"{model_config.key}_finetuned"
                evaluation_results["models"][result_key] = model_results
                evaluation_results["summary"]["finetuned"]["models"].append(result_key)
            except Exception as e:
                logger.error(f"Error evaluating finetuned model {model_config.key}: {e}")
        
        # 3. Evaluate finetuned models WITH manual RAG
        for model_config in finetuned_models:
            try:
                logger.info(f"Evaluating finetuned model with RAG: {model_config.key}")
                model_results = self.evaluate_model(model_config, use_rag=True, category="finetuned_rag")
                result_key = f"{model_config.key}_finetuned_rag"
                evaluation_results["models"][result_key] = model_results
                evaluation_results["summary"]["finetuned_rag"]["models"].append(result_key)
            except Exception as e:
                logger.error(f"Error evaluating finetuned+RAG model {model_config.key}: {e}")
        
        # Calculate category averages
        for category in ["base_rag", "finetuned", "finetuned_rag"]:
            category_models = evaluation_results["summary"][category]["models"]
            if category_models:
                total_sim = sum(
                    evaluation_results["models"][key]["average_cosine_similarity"]
                    for key in category_models
                    if key in evaluation_results["models"]
                )
                total_time = sum(
                    evaluation_results["models"][key]["average_response_time"]
                    for key in category_models
                    if key in evaluation_results["models"]
                )
                num_models = len(category_models)
                evaluation_results["summary"][category]["avg_similarity"] = total_sim / num_models
                evaluation_results["summary"][category]["avg_time"] = total_time / num_models
        
        return evaluation_results
    
    def save_results(self, results: Dict[str, Any], filename: str = "evaluation_results.json"):
        """
        Save evaluation results to JSON file
        
        Args:
            results: Evaluation results dictionary
            filename: Output filename
        """
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {filename}")
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a text report summarizing the evaluation results
        
        Args:
            results: Evaluation results dictionary
            
        Returns:
            Report text
        """
        report = []
        report.append("=" * 60)
        report.append("MODEL EVALUATION REPORT")
        report.append(f"Timestamp: {results['timestamp']}")
        report.append("=" * 60)
        report.append("")
        
        # Category summaries
        report.append("CATEGORY PERFORMANCE SUMMARY:")
        report.append("-" * 30)
        
        category_names = {
            "base_rag": "BASE MODELS WITH RAG",
            "finetuned": "FINETUNED MODELS (NO RAG)",
            "finetuned_rag": "FINETUNED MODELS WITH RAG"
        }
        
        for category in ["base_rag", "finetuned", "finetuned_rag"]:
            if category in results["summary"]:
                cat_data = results["summary"][category]
                report.append(f"\n{category_names.get(category, category.upper())}:")
                report.append(f"  Average Similarity: {cat_data['avg_similarity']:.3f}")
                report.append(f"  Average Response Time: {cat_data['avg_time']:.1f} ms")
                report.append(f"  Models evaluated: {len(cat_data['models'])}")
        
        # Individual model rankings
        report.append("\n" + "=" * 60)
        report.append("TOP PERFORMING MODELS (by similarity):")
        report.append("-" * 30)
        
        # Sort models by similarity
        sorted_models = sorted(
            results["models"].items(),
            key=lambda x: x[1]["average_cosine_similarity"],
            reverse=True
        )
        
        for rank, (key, data) in enumerate(sorted_models[:5], 1):
            report.append(f"{rank}. {data['name']}")
            report.append(f"   Category: {data['category']}")
            report.append(f"   Similarity: {data['average_cosine_similarity']:.3f}")
            report.append(f"   Response Time: {data['average_response_time']:.1f} ms")
            report.append("")
        
        report.append("=" * 60)
        return "\n".join(report)


def main():
    """Main evaluation function"""
    logger.info("Starting model evaluation...")
    
    evaluator = ModelEvaluator()
    
    # Run evaluation
    results = evaluator.evaluate_all_models()
    
    # Save results
    evaluator.save_results(results)
    
    # Generate and print report
    report = evaluator.generate_report(results)
    print(report)
    
    # Save report to file
    with open("evaluation_report.txt", "w") as f:
        f.write(report)
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()