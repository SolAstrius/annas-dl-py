"""S3 storage operations for caching downloaded books."""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

    from .config import Settings

logger = logging.getLogger(__name__)


# Default prefixes (can be overridden via Settings)
DEFAULT_RAW_PREFIX = "raw/annas"
DEFAULT_META_PREFIX = "meta/annas"
DEFAULT_COVERS_PREFIX = "covers/annas"


def build_key(hash: str, format: str, prefix: str = DEFAULT_RAW_PREFIX) -> str:
    """Build S3 key for a book file."""
    return f"{prefix}/{hash}.{format}"


def build_meta_key(hash: str, prefix: str = DEFAULT_META_PREFIX) -> str:
    """Build S3 key for book metadata."""
    return f"{prefix}/{hash}.json"


@dataclass
class S3Storage:
    """S3 storage backend for book caching."""

    _client: "S3Client"
    _bucket: str
    _presign_expiry: int
    _raw_prefix: str
    _meta_prefix: str

    @classmethod
    def create(cls, settings: "Settings") -> "S3Storage":
        """Create S3 storage from settings."""
        assert settings.s3_bucket is not None, "s3_bucket is required"

        config = BotoConfig(
            region_name=settings.s3_region,
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "adaptive"},
        )

        client: S3Client = boto3.client(
            "s3",
            config=config,
            endpoint_url=settings.s3_endpoint,
        )

        return cls(
            _client=client,
            _bucket=settings.s3_bucket,
            _presign_expiry=settings.s3_presign_expiry,
            _raw_prefix=settings.s3_raw_prefix,
            _meta_prefix=settings.s3_meta_prefix,
        )

    def book_key(self, hash: str, format: str) -> str:
        """Build S3 key for a book file using configured prefix."""
        return build_key(hash, format, self._raw_prefix)

    def meta_key(self, hash: str) -> str:
        """Build S3 key for metadata using configured prefix."""
        return build_meta_key(hash, self._meta_prefix)

    def cover_key(self, hash: str, ext: str = "jpg") -> str:
        """Build S3 key for a cover image."""
        return f"{DEFAULT_COVERS_PREFIX}/{hash}.{ext}"

    def exists(self, key: str) -> bool:
        """Check if a key exists in S3."""
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "404":
                return False
            raise

    def upload(self, key: str, content: bytes, content_type: str | None = None) -> None:
        """Upload content to S3."""
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type

        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=content,
            **extra_args,
        )
        logger.info("Uploaded %d bytes to s3://%s/%s", len(content), self._bucket, key)

    def download(self, key: str) -> bytes:
        """Download content from S3."""
        response = self._client.get_object(Bucket=self._bucket, Key=key)
        return response["Body"].read()

    def get_presigned_url(self, key: str, filename: str | None = None) -> str:
        """Generate a presigned URL for downloading a file.

        Args:
            key: S3 object key
            filename: Optional filename for Content-Disposition header

        Returns:
            Presigned URL valid for presign_expiry seconds
        """
        params = {
            "Bucket": self._bucket,
            "Key": key,
        }

        if filename:
            # Set Content-Disposition to suggest filename
            params["ResponseContentDisposition"] = f'attachment; filename="{filename}"'

        url = self._client.generate_presigned_url(
            "get_object",
            Params=params,
            ExpiresIn=self._presign_expiry,
        )
        return url


def content_type_for_format(format: str) -> str:
    """Get MIME content type for ebook format."""
    return {
        "epub": "application/epub+zip",
        "pdf": "application/pdf",
        "mobi": "application/x-mobipocket-ebook",
        "azw3": "application/vnd.amazon.ebook",
        "fb2": "application/x-fictionbook+xml",
        "djvu": "image/vnd.djvu",
        "cbr": "application/x-cbr",
        "cbz": "application/x-cbz",
    }.get(format, "application/octet-stream")
