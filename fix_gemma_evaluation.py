#!/usr/bin/env python3
"""
Re-run evaluation for Gemma-3-ndtv3 model only and update results
"""

import json
import os
import sys
from datetime import datetime
from llm_client import LLMClient
from rag import RagService
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_existing_results():
    """Load existing evaluation results"""
    results_file = "results/evaluation_results.json"
    if not os.path.exists(results_file):
        logger.error(f"Results file not found: {results_file}")
        sys.exit(1)
    
    with open(results_file, "r") as f:
        return json.load(f)

def load_ground_truth():
    """Load ground truth Q&A pairs"""
    with open("ground_truth_evaluation.json", "r") as f:
        data = json.load(f)
        return [(qa["question"], qa["answer"]) for qa in data]

def test_gemma_model():
    """Test if Gemma model is working"""
    logger.info("Testing Gemma-3-ndtv3 model connection...")
    
    try:
        client = LLMClient()
        test_response = client.get_model_response(
            "gemma-3-ndtv3", 
            "Hello, can you respond?",
            ""
        )
        logger.info(f"Test response: {test_response[:100]}...")
        return True
    except Exception as e:
        logger.error(f"Gemma model test failed: {e}")
        return False

def evaluate_gemma_model():
    """Re-run evaluation for Gemma model only"""
    
    # Load existing results
    logger.info("Loading existing results...")
    results = load_existing_results()
    
    # Test model first
    if not test_gemma_model():
        logger.error("Gemma model is not responding. Please check the endpoint.")
        return
    
    # Initialize clients
    client = LLMClient()
    rag = RagService()
    
    # Load ground truth
    qa_pairs = load_ground_truth()
    logger.info(f"Loaded {len(qa_pairs)} Q&A pairs")
    
    # Model configurations to update
    models_to_update = [
        ("gemma-3-ndtv3", "gemma-3-ndtv3_finetuned", False),  # Finetuned only
        ("gemma-3-ndtv3", "gemma-3-ndtv3_finetuned_rag", True)  # Finetuned + RAG
    ]
    
    for model_id, result_key, use_rag in models_to_update:
        logger.info(f"\nEvaluating {result_key} (RAG: {use_rag})...")
        
        if result_key not in results["models"]:
            logger.warning(f"Model {result_key} not found in results, skipping...")
            continue
        
        # Clear previous error responses
        results["models"][result_key]["responses"] = []
        
        # Evaluate each question
        for idx, (question, ground_truth) in enumerate(qa_pairs, 1):
            logger.info(f"  Question {idx}/{len(qa_pairs)}: {question[:50]}...")
            
            try:
                # Get context if using RAG
                context = ""
                if use_rag:
                    context = rag.get_context(question)
                    logger.debug(f"    Retrieved context: {len(context)} chars")
                
                # Get model response
                start_time = datetime.now()
                generated_answer = client.get_model_response(model_id, question, context)
                response_time = (datetime.now() - start_time).total_seconds()
                
                # Calculate cosine similarity (placeholder - would need embeddings)
                cosine_similarity = 0.0  # You can implement actual similarity calculation if needed
                
                # Store result
                response_data = {
                    "question": question,
                    "generated_answer": generated_answer,
                    "ground_truth": ground_truth,
                    "cosine_similarity": cosine_similarity,
                    "response_time_seconds": response_time
                }
                
                results["models"][result_key]["responses"].append(response_data)
                logger.info(f"    ✓ Answer generated ({response_time:.2f}s)")
                
            except Exception as e:
                logger.error(f"    ✗ Error generating answer: {e}")
                # Store error response
                response_data = {
                    "question": question,
                    "generated_answer": f"Error: {str(e)}",
                    "ground_truth": ground_truth,
                    "cosine_similarity": 0.0,
                    "response_time_seconds": 0.0
                }
                results["models"][result_key]["responses"].append(response_data)
        
        # Update statistics
        valid_responses = [r for r in results["models"][result_key]["responses"] 
                          if not r["generated_answer"].startswith("Error:")]
        
        if valid_responses:
            avg_similarity = sum(r["cosine_similarity"] for r in valid_responses) / len(valid_responses)
            avg_response_time = sum(r["response_time_seconds"] for r in valid_responses) / len(valid_responses)
        else:
            avg_similarity = 0.0
            avg_response_time = 0.0
        
        results["models"][result_key]["average_cosine_similarity"] = avg_similarity
        results["models"][result_key]["average_response_time_seconds"] = avg_response_time
        
        logger.info(f"  Completed {result_key}: {len(valid_responses)}/{len(qa_pairs)} successful")
    
    # Update metadata
    results["metadata"]["last_updated"] = datetime.now().isoformat()
    results["metadata"]["gemma_rerun"] = True
    
    # Save updated results
    logger.info("\nSaving updated results...")
    with open("results/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("✓ Evaluation results updated successfully!")
    logger.info(f"  File: results/evaluation_results.json")
    
    # Show summary
    for model_id, result_key, use_rag in models_to_update:
        if result_key in results["models"]:
            valid_count = sum(1 for r in results["models"][result_key]["responses"] 
                            if not r["generated_answer"].startswith("Error:"))
            total_count = len(results["models"][result_key]["responses"])
            logger.info(f"  {result_key}: {valid_count}/{total_count} successful responses")

if __name__ == "__main__":
    evaluate_gemma_model()