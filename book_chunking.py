import boto3
import json
import re
import time
import hashlib
from datetime import datetime, timezone

# AWS Setup
region = "us-east-2"
bucket = "tyson-chatbot-pipeline-storage"
textract = boto3.client("textract", region_name=region)
s3 = boto3.client("s3", region_name=region)
dynamodb = boto3.resource("dynamodb", region_name=region)
table = dynamodb.Table("demoDB")

def clean_text(text):
    text = text.lower()
    text = text.replace("\r", " ").replace("\t", " ")
    text = re.sub(r"[\[\]‘’“”—–]", "", text)
    text = re.sub(r"page\s+\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z0-9.,!?'\n ]+", "", text)
    return text.strip()

def run_textract_from_key(s3_key):
    response = textract.start_document_text_detection(
        DocumentLocation={"S3Object": {"Bucket": bucket, "Name": s3_key}}
    )
    job_id = response["JobId"]
    while True:
        result = textract.get_document_text_detection(JobId=job_id)
        if result["JobStatus"] in ["SUCCEEDED", "FAILED"]:
            break
        time.sleep(5)
    return job_id

def collect_text(job_id):
    lines = []
    next_token = None
    while True:
        result = textract.get_document_text_detection(
            JobId=job_id, NextToken=next_token
        ) if next_token else textract.get_document_text_detection(JobId=job_id)

        for block in result["Blocks"]:
            if block["BlockType"] == "LINE":
                lines.append(block["Text"])
        next_token = result.get("NextToken")
        if not next_token:
            break
    return "\n".join(lines)

def chunk_by_sentences(text, max_words=500, overlap_words=100):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    word_count = 0

    for sentence in sentences:
        words = sentence.split()
        if word_count + len(words) > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap_words:] + words
            word_count = len(current_chunk)
        else:
            current_chunk.extend(words)
            word_count += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def generate_chunk_id(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def save_chunks_to_dynamodb(chunks, source_name):
    timestamp = datetime.now(timezone.utc).isoformat()
    for chunk_text in chunks:
        if not chunk_text.strip():
            continue
        chunk_id = generate_chunk_id(chunk_text)
        item = {
            "chunk_id": chunk_id,
            "source": source_name,
            "text": chunk_text,
            "timestamp": timestamp
        }
        try:
            table.put_item(Item=item, ConditionExpression="attribute_not_exists(chunk_id)")
        except:
            pass

def process_existing_pdfs():
    response = s3.list_objects_v2(Bucket=bucket, Prefix="raw/books/")
    for obj in response.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".pdf"):
            job_id = run_textract_from_key(key)
            full_text = collect_text(job_id)
            cleaned = clean_text(full_text)
            chunks = chunk_by_sentences(cleaned)
            save_chunks_to_dynamodb(chunks, source_name="books")

# Run locally or as a one-time Lambda
def lambda_handler(event=None, context=None):
    process_existing_pdfs()
    return {
        "status": "success",
        "message": "All existing PDFs processed and stored in DynamoDB"
    }
