#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from evaluation import LLMJudge

# Load environment variables
load_dotenv()

# Check if API key exists
api_key = os.getenv("OPENAI_API_KEY")
print(f"API Key exists: {bool(api_key)}")
if api_key:
    print(f"API Key starts with: {api_key[:10]}...")

# Test the judge
try:
    print("\nInitializing LLM Judge...")
    judge = LLMJudge(model_name="gpt-4-turbo-preview")
    
    print("\nTesting evaluation...")
    test_query = "What causes a solar eclipse?"
    test_response = "A solar eclipse happens when the Moon passes between Earth and the Sun, casting a shadow on our planet. It's like cosmic hide-and-seek on a grand scale!"
    
    result = judge.evaluate_response(test_query, test_response, use_cache=False)
    
    print(f"\nEvaluation Result:")
    print(f"Overall Score: {result.overall_score}")
    print(f"Scores: {result.scores}")
    print(f"Strengths: {result.strengths}")
    print(f"Weaknesses: {result.weaknesses}")
    
except Exception as e:
    print(f"\nError occurred: {e}")
    import traceback
    traceback.print_exc()