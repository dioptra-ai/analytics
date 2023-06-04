import datetime
import logging
import urllib.parse
from enum import Enum
import json
import boto3
from botocore.exceptions import ClientError
from botocore.client import Config
from bson.objectid import ObjectId
from google.cloud import storage
from providers import get_mongo_client
import re
from s3urls import parse_url as parse_s3_url

class CloudStorageType(Enum):
    UNKNOWN = 1
    AWS_S3 = 2
    GOOGLE_STORAGE = 3


class ParsedFileUri:
    host: str
    bucket_name: str
    file_path: str
    cloud_storage_type: CloudStorageType

    def __init__(
        self,
        host: str,
        bucket_name: str,
        file_path: str,
        cloud_storage_type: CloudStorageType,
    ) -> None:
        self.host = host
        self.bucket_name = bucket_name
        self.file_path = file_path
        self.cloud_storage_type = cloud_storage_type

    @staticmethod
    def from_uri(uri: str):

        try:
            parsed_uri = parse_s3_url(uri)
            bucket_name = parsed_uri['bucket']
            file_path = parsed_uri['key']
            host = f'https://{bucket_name}.s3.amazonaws.com'
            cloud_storage_type = CloudStorageType.AWS_S3
        except:
            parsed_uri = urllib.parse.urlparse(uri)
            if parsed_uri.scheme == 'gs':
                bucket_name = parsed_uri.netloc
                file_path = parsed_uri.path
                cloud_storage_type = CloudStorageType.GOOGLE_STORAGE
                host = 'https://storage.cloud.google.com'
            elif 'amazon' in parsed_uri.netloc:
                cloud_storage_type = CloudStorageType.AWS_S3
                host = f"{parsed_uri.scheme}://{parsed_uri.netloc}"
                bucket_name = parsed_uri.netloc.split(".")[0]
                file_path = parsed_uri.path
            elif 'google' in parsed_uri.netloc:
                bucket_name = parsed_uri.path.split('/')[1]
                file_path = '/' + '/'.join(parsed_uri.path.split('/')[2:])
                cloud_storage_type = CloudStorageType.GOOGLE_STORAGE
                host = 'https://storage.cloud.google.com'
            else:
                host = f"{parsed_uri.scheme}://{parsed_uri.netloc}"
                bucket_name = parsed_uri.netloc.split(".")[0]
                file_path = parsed_uri.path
                cloud_storage_type = CloudStorageType.UNKNOWN

        return ParsedFileUri(
            host,
            bucket_name,
            file_path,
            cloud_storage_type=cloud_storage_type
        )

    def to_uri(self):

        return f'{self.host}/{self.file_path}'

class CloudStorageUtil:
    google_storage_client = None

    def generate_presigned_file_url(self, uri: str, organization_id: str) -> str:
        parsed_file_uri = None
        try:
            parsed_file_uri = ParsedFileUri.from_uri(uri)
            uri = parsed_file_uri.to_uri()

            if parsed_file_uri.cloud_storage_type == CloudStorageType.AWS_S3:

                s3_client = CloudStorageUtil._init_s3_client(organization_id, parsed_file_uri.bucket_name)

                if generated_presigned_s3_file_url := self._generate_presigned_s3_file_url(
                    s3_client,
                    parsed_file_uri.bucket_name, parsed_file_uri.file_path
                ):
                    return generated_presigned_s3_file_url
                else:
                    return uri
            elif parsed_file_uri.cloud_storage_type == CloudStorageType.GOOGLE_STORAGE:
                if self.google_storage_client is None:
                    self.google_storage_client = self._init_google_cloud_client(organization_id)
                if generated_presigned_google_storage_file_url := self._generate_presigned_google_storage_file_url(
                    parsed_file_uri.bucket_name, parsed_file_uri.file_path
                ):
                    return generated_presigned_google_storage_file_url
                else:
                    return uri
            else:
                logging.error(
                    f"Unknown cloud storage type detected for uri: {uri}, organization_id: {organization_id}"
                )
                return uri
        except Exception as e:
            logging.error(
                f"Cannot generate presigned url for uri: {uri}, organization_id: {organization_id}. Reason: {e}"
            )
            return uri

    @staticmethod
    def _init_s3_client(organization_id: str, bucket_name: str):
        aws_s3_integration = get_mongo_client().dioptra.integrations.find_one({"organization": ObjectId(organization_id), "type": "AWS_S3"})

        if aws_s3_integration is not None:
            aws_s3_integration_config = aws_s3_integration["data"]["aws"]

            tmp_s3_client = boto3.client('s3',
                aws_access_key_id=aws_s3_integration_config["aws_access_key_id"],
                aws_secret_access_key=aws_s3_integration_config["aws_secret_access_key"],
                aws_session_token=aws_s3_integration_config["aws_session_token"],
                config=Config(signature_version='s3v4'))
            response = tmp_s3_client.get_bucket_location(Bucket=bucket_name)
            region_name=response['LocationConstraint']

            return boto3.client(
                "s3",
                aws_access_key_id=aws_s3_integration_config["aws_access_key_id"],
                aws_secret_access_key=aws_s3_integration_config["aws_secret_access_key"],
                aws_session_token=aws_s3_integration_config["aws_session_token"],
                config=Config(signature_version='s3v4'),
                region_name=region_name
            )
        else:

            raise Exception('No AWS credentials found')

    def _init_google_cloud_client(self, organization_id: str):
        google_storage_integration = get_mongo_client().dioptra.integrations.find_one(
            {"organization": ObjectId(organization_id), "type": "GOOGLE_CLOUD_STORAGE"}
        )

        if google_storage_integration is not None:
            google_storage_credentials = google_storage_integration["data"]["google_cloud_storage"]["credentials_json"]

            return storage.Client.from_service_account_info(json.loads(google_storage_credentials))
        else:

            raise Exception('No GCS credentials found')

    def _generate_presigned_s3_file_url(self, s3_client, bucket_name: str, file_name: str) -> str:
        try:
            return s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket_name, "Key": file_name},
                ExpiresIn=14400
            )
        except ClientError as e:
            logging.error(
                f"""An error occurred during generating AWS S3 presigned url
                for bucket: {bucket_name}, filename: {file_name}. Reason: {e}
                """
            )

    def _generate_presigned_google_storage_file_url(
        self, bucket_name: str, file_name: str
    ) -> str:
        try:
            bucket = self.google_storage_client.get_bucket(bucket_name)
            blob = bucket.blob(file_name[1:])
            return blob.generate_signed_url(
                version="v4", expiration=datetime.timedelta(minutes=15), method="GET"
            )
        except Exception as e:
            logging.error(
                f"""An error occurred during generating Google Storage presigned url
                for bucket: {bucket_name}, filename: {file_name}. Reason: {e}
                """
            )
