#!/usr/bin/env python3
"""
Upload test images to S3 for renderer testing
"""

import boto3
import os
from app.config import S3_BUCKET, S3_RENDER_BUCKET, S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_REGION

def upload_to_s3():
    """Upload test images to S3"""
    print("📤 Uploading test images to S3...")
    
    try:
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            endpoint_url=S3_ENDPOINT_URL,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=S3_REGION
        )
        
        # Upload room image
        print("Uploading room image...")
        s3_client.upload_file(
            'test_room.jpg',
            S3_BUCKET,
            'uploads/test_room.jpg',
            ExtraArgs={'ContentType': 'image/jpeg'}
        )
        print("✅ Room image uploaded")
        
        # Upload item cutout
        print("Uploading item cutout...")
        s3_client.upload_file(
            'test_item_cutout.png',
            S3_BUCKET,
            'renders/test_item_cutout.png',
            ExtraArgs={'ContentType': 'image/png'}
        )
        print("✅ Item cutout uploaded")
        
        # Upload item shadow
        print("Uploading item shadow...")
        s3_client.upload_file(
            'test_item_shadow.png',
            S3_BUCKET,
            'renders/test_item_shadow.png',
            ExtraArgs={'ContentType': 'image/png'}
        )
        print("✅ Item shadow uploaded")
        
        print("\n🎉 All images uploaded successfully!")
        print(f"Room image: s3://{S3_BUCKET}/uploads/test_room.jpg")
        print(f"Item cutout: s3://{S3_BUCKET}/renders/test_item_cutout.png")
        print(f"Item shadow: s3://{S3_BUCKET}/renders/test_item_shadow.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

if __name__ == "__main__":
    upload_to_s3()
