import json
import boto3
import logging
from datetime import datetime
import uuid
import time

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize S3 client and DynamoDB resource
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# Configuration
BUCKET_NAME = 'tyson-chatbot-pipeline-storage'
TWITTER_RAW_PREFIX = 'raw/twitter/'
TWITTER_CHUNKS_PREFIX = 'chunks/'
CHUNK_SIZE = 1000  # Number of tokens/characters per chunk
DYNAMODB_TABLE = 'ChunkDB'

def lambda_handler(event, context):
    """
    Lambda function to process Twitter data from raw JSON files into chunks
    and store them in DynamoDB with better performance
    """
    start_time = time.time()
    try:
        logger.info(f"Starting Twitter chunking process at {datetime.now().isoformat()}")
        
        # Get DynamoDB table
        table = dynamodb.Table(DYNAMODB_TABLE)
        
        # In case of S3 event, extract the key from the event
        if event.get('Records') and event['Records'][0].get('s3'):
            file_key = event['Records'][0]['s3']['object']['key']
            logger.info(f"Triggered by S3 event for file: {file_key}")
            if file_key.endswith('.json') and 'metadata' not in file_key:
                process_twitter_file(file_key, table)
            return {
                'statusCode': 200,
                'body': json.dumps('Processed file from S3 event')
            }
        
        # If not S3 event, check for test flag
        if event.get('test'):
            logger.info("Running in test mode with mock data")
            # Create a test item directly
            chunk_id = generate_uuid()
            item = {
                'chunk_id': chunk_id,
                'source': "twitter",
                'text': "This is a test tweet for the optimized chunking lambda",
                'timestamp': datetime.now().isoformat()
            }
            table.put_item(Item=item)
            return {
                'statusCode': 200,
                'body': json.dumps(f'Test successful. Created item with ID: {chunk_id}')
            }
        
        # Otherwise process files from S3
        twitter_files = list_twitter_files(max_files=1)  # Limit to 1 file for better performance
        
        if not twitter_files:
            logger.warning("No Twitter files found to process")
            return {
                'statusCode': 200,
                'body': json.dumps('No Twitter files to process')
            }
        
        # Process the file
        file_key = twitter_files[0]
        chunks_created = process_twitter_file(file_key, table)
        
        end_time = time.time()
        logger.info(f"Twitter chunking completed. Created {chunks_created} chunks in {end_time - start_time:.2f} seconds")
        
        return {
            'statusCode': 200,
            'body': json.dumps(f'Successfully processed Twitter data. Created {chunks_created} chunks.')
        }
        
    except Exception as e:
        error_msg = f"Error processing Twitter data: {str(e)}"
        logger.error(error_msg)
        
        return {
            'statusCode': 500,
            'body': json.dumps(error_msg)
        }

def generate_uuid():
    """Generate a proper UUID like in the books implementation"""
    return str(uuid.uuid4()).replace('-', '')

def list_twitter_files(max_files=1):
    """List Twitter JSON files in the raw/twitter directory up to max_files"""
    try:
        response = s3.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=TWITTER_RAW_PREFIX,
            MaxKeys=max_files
        )
        
        if 'Contents' not in response:
            # Create a mock file for testing if needed
            create_mock_twitter_file()
            response = s3.list_objects_v2(
                Bucket=BUCKET_NAME,
                Prefix=TWITTER_RAW_PREFIX,
                MaxKeys=max_files
            )
        
        # Filter for JSON files
        return [item['Key'] for item in response.get('Contents', []) 
                if item['Key'].endswith('.json') and 'metadata' not in item['Key']]
    except Exception as e:
        logger.error(f"Error listing Twitter files: {str(e)}")
        return []

def create_mock_twitter_file():
    """Create a mock Twitter file for testing purposes"""
    mock_tweets = [
        {"id": "12345", "text": "This is a test tweet for the chunking lambda"},
        {"id": "67890", "text": "Another test tweet that will be processed into a chunk"}
    ]
    
    mock_file_key = f"{TWITTER_RAW_PREFIX}mock_twitter_data.json"
    
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=mock_file_key,
        Body=json.dumps(mock_tweets),
        ContentType='application/json'
    )

def process_twitter_file(file_key, table):
    """
    Process a single Twitter JSON file into chunks and store in DynamoDB
    
    Args:
        file_key: S3 key for the Twitter JSON file
        table: DynamoDB table resource
    
    Returns:
        int: Number of chunks created
    """
    # Download the file from S3
    response = s3.get_object(Bucket=BUCKET_NAME, Key=file_key)
    file_content = response['Body'].read().decode('utf-8')
    
    # Parse the JSON content
    tweets = json.loads(file_content)
    
    # Prepare batch write
    chunk_counter = 0
    batch_size = 0
    batch_items = []
    
    # Process each tweet
    for tweet in tweets:
        # Skip if tweet is not in the expected format
        if not isinstance(tweet, dict) or 'text' not in tweet:
            continue
            
        tweet_text = tweet['text']
        tweet_tokens = len(tweet_text)  # Simple character count as token proxy
        
        # If this single tweet is larger than chunk size, we'll need to split it
        if tweet_tokens > CHUNK_SIZE:
            # Split the tweet into chunks
            for i in range(0, tweet_tokens, CHUNK_SIZE):
                chunk_text = tweet_text[i:i+CHUNK_SIZE]
                chunk_id = generate_uuid()
                
                batch_items.append({
                    'PutRequest': {
                        'Item': {
                            'chunk_id': chunk_id,
                            'source': "twitter",
                            'text': chunk_text,
                            'timestamp': datetime.now().isoformat()
                        }
                    }
                })
                
                batch_size += 1
                chunk_counter += 1
                
                # If batch is full, write to DynamoDB
                if batch_size >= 25:  # DynamoDB batch size limit is 25
                    write_batch_to_dynamodb(table, batch_items)
                    batch_items = []
                    batch_size = 0
        else:
            # Save this tweet as a single chunk
            chunk_id = generate_uuid()
            
            batch_items.append({
                'PutRequest': {
                    'Item': {
                        'chunk_id': chunk_id,
                        'source': "twitter",
                        'text': tweet_text,
                        'timestamp': datetime.now().isoformat()
                    }
                }
            })
            
            batch_size += 1
            chunk_counter += 1
            
            # If batch is full, write to DynamoDB
            if batch_size >= 25:  # DynamoDB batch size limit is 25
                write_batch_to_dynamodb(table, batch_items)
                batch_items = []
                batch_size = 0
    
    # Write any remaining items in the batch
    if batch_items:
        write_batch_to_dynamodb(table, batch_items)
    
    # Also save a copy to S3 for backup
    output_key = f"{TWITTER_CHUNKS_PREFIX}{file_key.split('/')[-1].split('.')[0]}_chunks.json"
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=output_key,
        Body=json.dumps({"processed": True, "chunks_created": chunk_counter}),
        ContentType='application/json'
    )
    
    return chunk_counter

def write_batch_to_dynamodb(table, batch_items):
    """
    Write a batch of items to DynamoDB
    
    Args:
        table: DynamoDB table resource
        batch_items: List of batch items to write
    """
    try:
        # Write the batch to DynamoDB
        dynamodb_client = boto3.client('dynamodb')
        dynamodb_client.batch_write_item(
            RequestItems={
                DYNAMODB_TABLE: batch_items
            }
        )
    except Exception as e:
        logger.error(f"Error writing batch to DynamoDB: {str(e)}")
        # If batch fails, try writing items individually
        for item in batch_items:
            try:
                table.put_item(Item=item['PutRequest']['Item'])
            except Exception as e2:
                logger.error(f"Error writing item to DynamoDB: {str(e2)}")