"""Configuration from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Service configuration loaded from environment."""

    # Anna's Archive API key (optional - can be passed via X-Annas-Key header)
    annas_secret_key: str | None = None

    # S3 configuration (optional for CLI commands that don't use S3)
    s3_bucket: str | None = None
    s3_region: str = "us-east-1"
    s3_endpoint: str | None = None
    s3_presign_expiry: int = 604800  # 7 days in seconds
    s3_raw_prefix: str = "raw/annas"  # Path prefix for book files
    s3_meta_prefix: str = "meta/annas"  # Path prefix for metadata files

    # CDN configuration
    cdn_start_index: int = 5  # Start with higher indices (0-4 often overloaded)
    cdn_max_attempts: int = 5
    cdn_connect_timeout: float = 5.0
    cdn_download_timeout: float = 45.0

    # IPFS fallback configuration
    ipfs_gateways: list[str] = [
        "http://ipfs:8080",
        "https://ipfs.io",
    ]
    ipfs_timeout: float = 60.0  # IPFS can be slower than CDN
    ipfs_enabled: bool = False  # IPFS content not actively seeded; disabled

    # Torrent fallback configuration
    torrent_enabled: bool = True  # Set to False to disable torrent fallback
    torrent_timeout: float = 180.0  # Max seconds per torrent download

    # Concurrency settings (free-threaded Python can use real threads)
    max_concurrent_downloads: int = 8

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Public base URL for generating external links (e.g., https://dl.example.com)
    base_url: str | None = None

    # FlareSolverr for DDoS-Guard bypass
    flaresolverr_url: str | None = None

    model_config = {"env_prefix": "ANNAS_DL_", "env_file": ".env", "extra": "ignore"}


def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
