from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env file

DATABASE_URL = os.getenv("DATABASE_URL")
S3_BUCKET = os.getenv("S3_BUCKET")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
SHOPIFY_API_KEY = os.getenv("SHOPIFY_API_KEY")
SHOPIFY_API_SECRET = os.getenv("SHOPIFY_API_SECRET")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL")