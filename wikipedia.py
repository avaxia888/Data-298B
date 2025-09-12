import requests
import re
import json
import uuid
import datetime
import boto3

def lambda_handler(event, context):
    title = event.get('title', 'Neil deGrasse Tyson')  # Default title if none provided

    # Fetch, clean, and chunk the article content
    article_content = fetch_wikipedia_article(title)
    cleaned_content = clean_text(article_content)
    article_chunks = chunk_text(cleaned_content, max_tokens=1000, overlap=120)

    # Initialize DynamoDB client
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('test_DB')

    chunk_count = 0
    for chunk in article_chunks:
        json_chunk = create_json_chunk(chunk, "wikipedia")
        json_chunk["title"] = title  # Add title for reference
        table.put_item(Item=json_chunk)
        chunk_count += 1

    return {
        'statusCode': 200,
        'body': json.dumps(f"{chunk_count} chunks from '{title}' have been stored in DynamoDB table 'test_DB'.")
    }

def fetch_wikipedia_article(title):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'format': 'json',
        'titles': title,
        'prop': 'extracts',
        'explaintext': True,
    }
    response = requests.get(url, params=params)
    data = response.json()
    page = next(iter(data['query']['pages'].values()))
    return page['extract']

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_text(text, max_tokens=1024, overlap=120):
    tokens = text.split()
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens - overlap)]
    return [' '.join(chunk) for chunk in chunks]

def create_json_chunk(chunk, source):
    chunk_id = uuid.uuid4().hex
    timestamp = datetime.datetime.now().isoformat()
    return {
        "chunk_id": chunk_id,
        "source": source,
        "text": chunk,
        "timestamp": timestamp
    }