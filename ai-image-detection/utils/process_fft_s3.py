import boto3
from io import BytesIO
from data_utils.fft_transform import fft_magnitude_transform

s3 = boto3.client('s3')

def process_s3_folder(bucket_name, source_prefix, target_prefix):
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=source_prefix)
    for obj in response.get('Contents', []):
        key = obj['Key']
        if key.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Download original image
            img_obj = s3.get_object(Bucket=bucket_name, Key=key)
            img_stream = BytesIO(img_obj['Body'].read())

            # Apply FFT
            fft_img = fft_magnitude_transform(img_stream)

            # Upload transformed image
            buffer = BytesIO()
            fft_img.save(buffer, format='PNG')
            buffer.seek(0)
            new_key = key.replace(source_prefix, target_prefix)

            s3.put_object(Bucket=bucket_name, Key=new_key, Body=buffer, ContentType='image/png')
