"""CLI for testing the Anna's Archive downloader."""

import argparse
import asyncio
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


async def search_books(
    query: str,
    rows: int = 20,
    content: list[str] | None = None,
    ext: list[str] | None = None,
    lang: list[str] | None = None,
    sort: str = "",
) -> None:
    """Search Anna's Archive catalog."""
    from .annas_client import AnnasClient
    from .search import ContentType, SearchFilters, SortField

    # Map sort string to enum
    sort_map = {
        "": SortField.RELEVANCE,
        "newest": SortField.NEWEST,
        "oldest": SortField.OLDEST,
        "largest": SortField.LARGEST,
        "smallest": SortField.SMALLEST,
    }
    sort_field = sort_map.get(sort.lower(), SortField.RELEVANCE)

    # Map content type strings to enums
    content_types = []
    for ct in content or []:
        try:
            content_types.append(ContentType(ct.lower()))
        except ValueError:
            print(f"Warning: Unknown content type '{ct}', ignoring")

    filters = SearchFilters(
        query=query,
        content_types=content_types,
        formats=[f.lower() for f in (ext or [])],
        languages=[l.lower() for l in (lang or [])],
        page=1,
        sort=sort_field,
    )

    client = AnnasClient.create(timeout=30.0)

    try:
        results = await client.search(filters)

        if not results:
            print("No results found.")
            return

        # Limit to requested rows
        results = results[:rows]

        for item in results:
            # Format author
            author = item.author or ""
            if len(author) > 50:
                author = author[:47] + "..."

            print(f"  {item.id}")
            print(f"    Title: {item.title[:70]}{'...' if len(item.title) > 70 else ''}")
            if author:
                print(f"    Author: {author}")
            if item.year:
                print(f"    Year: {item.year}")
            if item.format:
                print(f"    Format: {item.format}", end="")
                if item.size:
                    print(f"  Size: {item.size}")
                else:
                    print()
            if item.content_type:
                print(f"    Type: {item.content_type.value}")
            print()

        print(f"Showing {len(results)} results")

    finally:
        await client.close()


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
        print("  This CLI only handles anna URNs (urn:anna:<hash>)")
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
            print("✓ Success")
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
  annas-dl search "Hitchhiker's Guide"
  annas-dl search "python" --ext pdf --ext epub --lang en
  annas-dl search "Dickens" --content book_fiction --sort newest
  annas-dl download --id urn:anna:f29c245d8956dfbab89f3001f3ae5ad2
  annas-dl download --id f29c245d8956dfbab89f3001f3ae5ad2 --format epub
  annas-dl serve
  annas-dl api --id urn:anna:f29c245d8956dfbab89f3001f3ae5ad2
""",
    )
    parser.add_argument("command", choices=["search", "download", "api", "serve"])
    parser.add_argument("query", nargs="?", help="Search query (for search command)")
    parser.add_argument(
        "--id", "-i",
        help="Book URN (urn:anna:<hash>) or raw MD5 hash",
        metavar="URN",
    )
    # Keep --hash as alias for backwards compatibility
    parser.add_argument("--hash", "-H", dest="id", help=argparse.SUPPRESS)
    parser.add_argument("--format", "-f", default="pdf", help="Format hint (default: pdf)")
    parser.add_argument("--key", "-k", help="API key (or set ANNAS_DL_ANNAS_SECRET_KEY)")
    # Search-specific options
    parser.add_argument("--rows", "-n", type=int, default=20, help="Number of results (default: 20)")
    parser.add_argument("--content", "-c", action="append", help="Content type filter (repeatable)")
    parser.add_argument("--ext", "-e", action="append", help="Format filter (repeatable)")
    parser.add_argument("--lang", "-l", action="append", help="Language filter (repeatable)")
    parser.add_argument("--sort", "-s", default="", help="Sort: newest, oldest, largest, smallest")

    args = parser.parse_args()

    # Load settings (reads .env file)
    from .config import get_settings
    settings = get_settings()

    secret_key = args.key or settings.annas_secret_key

    if args.command == "serve":
        from .main import main as serve_main
        serve_main()

    elif args.command == "search":
        if not args.query:
            print("Error: query required for search")
            print("Usage: annas-dl search \"your query\"")
            sys.exit(1)
        asyncio.run(search_books(
            args.query,
            rows=args.rows,
            content=args.content,
            ext=args.ext,
            lang=args.lang,
            sort=args.sort,
        ))

    elif args.command == "download":
        if not args.id:
            print("Error: --id required (URN or hash)")
            sys.exit(1)
        if not secret_key:
            print("Error: --key or ANNAS_DL_ANNAS_SECRET_KEY required")
            sys.exit(1)
        assert secret_key is not None  # narrowed by check above
        asyncio.run(test_download(args.id, secret_key, args.format))

    elif args.command == "api":
        if not args.id:
            print("Error: --id required (URN or hash)")
            sys.exit(1)
        if not secret_key:
            print("Error: --key or ANNAS_DL_ANNAS_SECRET_KEY required")
            sys.exit(1)
        assert secret_key is not None  # narrowed by check above
        asyncio.run(test_api(args.id, secret_key, args.format))


if __name__ == "__main__":
    main()
