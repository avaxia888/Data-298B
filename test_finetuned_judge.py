#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from evaluation import LLMJudge

# Load environment variables
load_dotenv()

# Test the judge with GPT-5 (will fallback to GPT-4 if not available)
try:
    print("Testing LLM Judge with GPT-5...")
    judge = LLMJudge(model_name="gpt-5")
    
    # Test evaluation for a finetuned model response
    test_query = "What are black holes?"
    test_response = """Black holes are cosmic vacuum cleaners gone rogue! These gravitational monsters form when massive stars collapse, 
    creating a region where gravity is so intense that not even light can escape. Picture space-time as a trampoline, 
    and a black hole as a bowling ball so heavy it creates a hole you can't climb out of. Fascinating stuff!"""
    
    print(f"\nQuery: {test_query}")
    print(f"Response: {test_response}\n")
    
    result = judge.evaluate_response(test_query, test_response, use_cache=False)
    
    print(f"Evaluation Result:")
    print(f"Overall Score: {result.overall_score:.1f}/10")
    print(f"\nDetailed Scores:")
    for key, value in result.scores.items():
        formatted_key = key.replace("_", " ").title()
        print(f"  • {formatted_key}: {value:.1f}/10")
    
    if result.strengths:
        print(f"\nStrengths:")
        for strength in result.strengths:
            print(f"  ✓ {strength}")
    
    if result.weaknesses:
        print(f"\nAreas for Improvement:")
        for weakness in result.weaknesses:
            print(f"  • {weakness}")
    
    if result.suggestions:
        print(f"\nSuggestions: {result.suggestions}")
    
except Exception as e:
    print(f"\nError occurred: {e}")
    import traceback
    traceback.print_exc()