import boto3
from config.config_env import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, BEDROCK_SERVICE_NAME, BEDROCK_REGION_NAME

BedrockClient = boto3.client(
    service_name=BEDROCK_SERVICE_NAME,
    region_name=BEDROCK_REGION_NAME,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)