"""Configuration from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Service configuration loaded from environment."""

    # Anna's Archive API key (optional - can be passed via X-Annas-Key header)
    annas_secret_key: str | None = None

    # S3 configuration
    s3_bucket: str
    s3_region: str = "us-east-1"
    s3_endpoint: str | None = None
    s3_presign_expiry: int = 604800  # 7 days in seconds

    # CDN configuration
    cdn_start_index: int = 5  # Start with higher indices (0-4 often overloaded)
    cdn_max_attempts: int = 5
    cdn_connect_timeout: float = 5.0
    cdn_download_timeout: float = 45.0

    # Concurrency settings (free-threaded Python can use real threads)
    max_concurrent_downloads: int = 8

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_prefix": "ANNAS_DL_", "env_file": ".env", "extra": "ignore"}


def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
