import boto3
import pandas as pd
import io
from src.logger import setup_logger
import json

logging = setup_logger()

class S3Storage:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')

    # function to upload a file to S3 as a CSV file
    def upload_bytes(self, data, s3_key, content_type):
        try:
            self.s3_client.put_object(Bucket=self.bucket_name, Key=s3_key, Body=data, ContentType=content_type)

            logging.info(f"Data uploaded to S3 at {s3_key}")
        except Exception as e:
            logging.error(f"Error uploading the file to S3: {e}")
            raise

   

    def load_csv(self, s3_key):
        try:
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)

            df = pd.read_csv(io.BytesIO(obj["Body"].read()))

            logging.info(f"loaded csv from S3: {s3_key}, shape: {df.shape}")

            return df

        except Exception as e:
            logging.error(f"Error loading CSV from S3: {e}")
            raise

   
    def load_json(self, s3_key):
        try:
            obj = self.s3_client.get_object(Bucket=self.bucket_name,Key=s3_key)

            data = json.loads(obj["Body"].read().decode("utf-8"))

            logging.info(f"JSON data downloaded from S3 at {s3_key}")

            return data

        except Exception as e:
            logging.error(f"Error loading JSON from S3: {e}")
            raise

    def get_latest_file(self, prefix, keyword): 
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)

            if 'Contents' not in response:
                raise ValueError("No files found in S3 with prefix")
                
            #filter only relevant files 
            files =  [obj for obj in response['Contents'] if keyword in obj['Key']]

            if not files:
                raise ValueError(f"No files found in S3 with keyword: {keyword}")

            #select the latest file based on LastModified
            latest_file = max(files, key=lambda x: x['LastModified'])
            
            logging.info(f"Latest file selected:{latest_file['Key']}")
            
            return latest_file['Key']    
            
        except Exception as e:
            logging.error(f"Error retrieving latest file from S3: {e}")
            raise

    def upload_to_s3(self, local_path: str, model_name: str, run_id: str) -> str:
        # Upload model file to S3 and return the S3 and return the full s3 uri

        try:
            s3_key = f'mlflow_artifacts/{model_name}/{run_id}.joblib'
            boto3.client('s3').upload_file(local_path, self.bucket_name, s3_key)
            s3_uri = f's3://{self.bucket_name}/{s3_key}'

            logging.info(f"Model file uploaded to {s3_uri}")
            return s3_uri
        
        except Exception as e:
            logging.error(f"Error uploading model file to S3: {e}")
            raise
       