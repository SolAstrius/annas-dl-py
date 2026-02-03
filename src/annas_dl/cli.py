"""CLI for testing the Anna's Archive downloader."""

import argparse
import asyncio
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


async def test_download(id: str, secret_key: str, format: str = "pdf") -> None:
    """Test downloading a book (no S3, just CDN)."""
    from .annas_client import AnnasClient
    from .config import get_settings
    from .downloader import download_book
    from .urn import parse_urn, to_urn, WrongResolverError, InvalidUrnError

    # Parse URN or raw hash
    try:
        parsed = parse_urn(id)
        hash = parsed.hash
        urn = to_urn(hash)
    except WrongResolverError as e:
        print(f"✗ {e}")
        print(f"  This CLI only handles anna URNs (urn:anna:<hash>)")
        sys.exit(1)
    except InvalidUrnError as e:
        print(f"✗ {e}")
        sys.exit(1)

    settings = get_settings()
    client = AnnasClient.create(timeout=15.0)

    print(f"Downloading {urn}")
    print(f"  Hash: {hash}")
    print(f"  Format hint: {format}")

    try:
        result = await download_book(client, settings, secret_key, hash, format)
        print(f"✓ Downloaded {result.size_bytes} bytes in {result.duration_ms}ms")
        print(f"  Format: {result.format}")
        print(f"  CDN: {result.cdn_host}")

        # Save to local file for inspection
        filename = f"{hash}.{result.format}"
        with open(filename, "wb") as f:
            f.write(result.content)
        print(f"  Saved: {filename}")

    finally:
        await client.close()


async def test_api(id: str, secret_key: str, format: str = "pdf") -> None:
    """Test the full API endpoint (requires S3 config)."""
    import httpx

    from .config import get_settings
    from .urn import parse_urn, to_urn, WrongResolverError, InvalidUrnError

    # Parse URN or raw hash (for display)
    try:
        parsed = parse_urn(id)
        hash = parsed.hash
        urn = to_urn(hash)
    except WrongResolverError as e:
        print(f"✗ {e}")
        sys.exit(1)
    except InvalidUrnError as e:
        print(f"✗ {e}")
        sys.exit(1)

    settings = get_settings()
    # Use URN in the URL (server accepts both URN and hash)
    url = f"http://{settings.host}:{settings.port}/book/{urn}/download"

    print(f"POST {url}")

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            url,
            json={"title": "", "format": format},
            headers={"X-Annas-Key": secret_key},
        )

        if response.is_success:
            data = response.json()
            print(f"✓ Success")
            print(f"  ID: {data['id']}")
            print(f"  Hash: {data['hash']}")
            print(f"  Format: {data['format']}")
            print(f"  Size: {data['size_bytes']} bytes")
            print(f"  Duration: {data['duration_ms']}ms")
            print(f"  CDN: {data['cdn_host']}")
            print(f"  Cached: {data['cached']}")
            print(f"  URL: {data['download_url'][:80]}...")
        else:
            print(f"✗ Error {response.status_code}: {response.text}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Anna's Archive download CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  annas-dl download --id urn:anna:f29c245d8956dfbab89f3001f3ae5ad2
  annas-dl download --id f29c245d8956dfbab89f3001f3ae5ad2 --format epub
  annas-dl serve
  annas-dl api --id urn:anna:f29c245d8956dfbab89f3001f3ae5ad2
""",
    )
    parser.add_argument("command", choices=["download", "api", "serve"])
    parser.add_argument(
        "--id", "-i",
        help="Book URN (urn:anna:<hash>) or raw MD5 hash",
        metavar="URN",
    )
    # Keep --hash as alias for backwards compatibility
    parser.add_argument("--hash", "-H", dest="id", help=argparse.SUPPRESS)
    parser.add_argument("--format", "-f", default="pdf", help="Format hint (default: pdf)")
    parser.add_argument("--key", "-k", help="API key (or set ANNAS_DL_ANNAS_SECRET_KEY)")

    args = parser.parse_args()

    # Load settings (reads .env file)
    from .config import get_settings
    settings = get_settings()

    secret_key = args.key or settings.annas_secret_key

    if args.command == "serve":
        from .main import main as serve_main
        serve_main()

    elif args.command == "download":
        if not args.id:
            print("Error: --id required (URN or hash)")
            sys.exit(1)
        if not secret_key:
            print("Error: --key or ANNAS_DL_ANNAS_SECRET_KEY required")
            sys.exit(1)
        asyncio.run(test_download(args.id, secret_key, args.format))

    elif args.command == "api":
        if not args.id:
            print("Error: --id required (URN or hash)")
            sys.exit(1)
        if not secret_key:
            print("Error: --key or ANNAS_DL_ANNAS_SECRET_KEY required")
            sys.exit(1)
        asyncio.run(test_api(args.id, secret_key, args.format))


if __name__ == "__main__":
    main()
