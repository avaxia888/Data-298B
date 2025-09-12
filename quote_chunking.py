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
BRAINYQUOTE_RAW_PREFIX = 'raw/brainyquotes/'
CHUNKS_PREFIX = 'chunks/'
CHUNK_SIZE = 1000  # Number of characters per chunk - same as Twitter implementation
DYNAMODB_TABLE = 'ChunkDB'

def lambda_handler(event, context):
    """
    Lambda function to process BrainyQuote data from raw JSON files into chunks
    and store them in DynamoDB with better performance
    
    Args:
        event: AWS Lambda event object (can contain bucket/key info if triggered by S3)
        context: AWS Lambda context object
    
    Returns:
        dict: Response with processing status
    """
    start_time = time.time()
    try:
        # Log the start of processing
        logger.info(f"Starting BrainyQuote chunking process at {datetime.now().isoformat()}")
        
        # Get DynamoDB table
        table = dynamodb.Table(DYNAMODB_TABLE)
        
        # Check if this is an S3 event
        if 'Records' in event and len(event['Records']) > 0 and 's3' in event['Records'][0]:
            # Extract bucket and key from the S3 event
            bucket = event['Records'][0]['s3']['bucket']['name']
            key = event['Records'][0]['s3']['object']['key']
            
            logger.info(f"Processing file from S3 event: {bucket}/{key}")
            
            # Process the file directly from the event
            chunks_created = process_brainyquote_file(bucket, key, table)
            
            end_time = time.time()
            processing_time = end_time - start_time
            logger.info(f"Processed {key} - created {chunks_created} chunks in {processing_time:.2f} seconds")
            
            return {
                'statusCode': 200,
                'body': json.dumps(f'Successfully processed file {key}. Created {chunks_created} chunks.')
            }
        
        # If not an S3 event, test for a test flag
        if event.get('test'):
            logger.info("Running in test mode with mock data")
            # Create a test item directly
            chunk_id = generate_uuid()
            item = {
                'chunk_id': chunk_id,
                'source': "brainyquote",
                'text': "This is a test quote for the optimized chunking lambda",
                'timestamp': datetime.now().isoformat()
            }
            table.put_item(Item=item)
            return {
                'statusCode': 200,
                'body': json.dumps(f'Test successful. Created item with ID: {chunk_id}')
            }
        
        # If not an S3 event, look for files in the brainyquotes directory
        brainyquote_files = list_brainyquote_files(max_files=1)  # Limit to 1 file for better performance
        
        if not brainyquote_files:
            logger.warning(f"No BrainyQuote files found to process in {BRAINYQUOTE_RAW_PREFIX}")
            return {
                'statusCode': 200,
                'body': json.dumps('No BrainyQuote files to process')
            }
        
        # Process the first BrainyQuote file
        file_key = brainyquote_files[0]
        logger.info(f"Found BrainyQuote file: {file_key}")
        chunks_created = process_brainyquote_file(BUCKET_NAME, file_key, table)
        
        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"BrainyQuote chunking completed. Created {chunks_created} chunks in {processing_time:.2f} seconds")
        
        return {
            'statusCode': 200,
            'body': json.dumps(f'Successfully processed BrainyQuote data. Created {chunks_created} chunks.')
        }
        
    except Exception as e:
        error_msg = f"Error processing BrainyQuote data: {str(e)}"
        logger.error(error_msg)
        
        return {
            'statusCode': 500,
            'body': json.dumps(error_msg)
        }

def generate_uuid():
    """Generate a proper UUID like in the books and Twitter implementation"""
    return str(uuid.uuid4()).replace('-', '')

def list_brainyquote_files(max_files=1):
    """
    List BrainyQuote JSON files in the raw/brainyquotes directory up to max_files
    
    Args:
        max_files: Maximum number of files to return
        
    Returns:
        list: List of file keys
    """
    try:
        response = s3.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=BRAINYQUOTE_RAW_PREFIX,
            MaxKeys=max_files
        )
        
        if 'Contents' not in response:
            return []
        
        # Filter for JSON files (exclude directories and metadata)
        return [item['Key'] for item in response.get('Contents', []) 
                if item['Key'].endswith('.json') and 'metadata' not in item['Key']]
    except Exception as e:
        logger.error(f"Error listing BrainyQuote files: {str(e)}")
        return []

def process_brainyquote_file(bucket, file_key, table):
    """
    Process a single BrainyQuote JSON file into chunks and store in DynamoDB
    
    Args:
        bucket: S3 bucket name
        file_key: S3 object key for the BrainyQuote JSON file
        table: DynamoDB table resource
    
    Returns:
        int: Number of chunks created
    """
    try:
        # Download the file from S3
        logger.info(f"Downloading file: {bucket}/{file_key}")
        response = s3.get_object(Bucket=bucket, Key=file_key)
        file_content = response['Body'].read().decode('utf-8')
        
        # Parse the JSON content
        quotes_data = json.loads(file_content)
        logger.info(f"JSON parsed successfully. Data type: {type(quotes_data)}")
        
        # Extract quotes from the JSON structure
        if isinstance(quotes_data, dict):
            quotes = quotes_data.get('quotes', [])
            logger.info(f"Found quotes in dictionary: {len(quotes)}")
        elif isinstance(quotes_data, list):
            quotes = quotes_data
            logger.info(f"Treating JSON array as quotes list: {len(quotes)}")
        else:
            logger.error(f"Unexpected JSON structure: {type(quotes_data)}")
            return 0
        
        if not quotes:
            logger.warning(f"No quotes found in {file_key}")
            return 0
        
        # Initialize counters and batching
        chunk_counter = 0
        batch_size = 0
        batch_items = []
        
        # Process each quote
        for quote in quotes:
            # Handle both string quotes and dictionary quotes
            if isinstance(quote, dict) and 'text' in quote:
                quote_text = quote['text'].strip()
            elif isinstance(quote, str):
                quote_text = quote.strip()
            else:
                logger.warning(f"Skipping invalid quote format: {type(quote)}")
                continue
                
            quote_length = len(quote_text)
            
            # If this single quote is larger than chunk size, we'll need to split it
            if quote_length > CHUNK_SIZE:
                # Split the quote into chunks
                for i in range(0, quote_length, CHUNK_SIZE):
                    chunk_text = quote_text[i:i+CHUNK_SIZE]
                    chunk_id = generate_uuid()
                    
                    batch_items.append({
                        'PutRequest': {
                            'Item': {
                                'chunk_id': chunk_id,
                                'source': "brainyquote",
                                'text': chunk_text,
                                'is_partial': True,
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
                # Save this quote as a single chunk
                chunk_id = generate_uuid()
                
                batch_items.append({
                    'PutRequest': {
                        'Item': {
                            'chunk_id': chunk_id,
                            'source': "brainyquote",
                            'text': quote_text,
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
        output_key = f"{CHUNKS_PREFIX}{file_key.split('/')[-1].split('.')[0]}_chunks.json"
        logger.info(f"Saving chunks metadata to {output_key}")
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=output_key,
            Body=json.dumps({"processed": True, "chunks_created": chunk_counter}),
            ContentType='application/json'
        )
        
        logger.info(f"Successfully created {chunk_counter} chunks")
        return chunk_counter
        
    except Exception as e:
        logger.error(f"Error processing file {file_key}: {str(e)}")
        raise

def write_batch_to_dynamodb(table, batch_items):
    """
    Write a batch of items to DynamoDB
    
    Args:
        table: DynamoDB table resource
        batch_items: List of batch items to write
    """
    try:
        # The boto3 resource level API handles conversion properly
        with table.batch_writer() as batch:
            for item in batch_items:
                batch.put_item(Item=item['PutRequest']['Item'])
        logger.info(f"Successfully wrote batch of {len(batch_items)} items to DynamoDB")
    except Exception as e:
        logger.error(f"Error writing batch to DynamoDB: {str(e)}")
        # If batch fails, try writing items individually
        for item in batch_items:
            try:
                table.put_item(Item=item['PutRequest']['Item'])
            except Exception as e2:
                logger.error(f"Error writing item to DynamoDB: {str(e2)}")