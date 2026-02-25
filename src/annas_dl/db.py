"""PostgreSQL integration for persisting book metadata to the shared annas-mcp database.

Writes to the same `books` and `book_identifiers` tables defined in
annas-mcp-rs/k8s/init.sql. Connection is optional — if DATABASE_URL is not
configured, all operations silently no-op.

All queries use asyncpg prepared statements ($1, $2, ...) — values are never
string-interpolated into SQL.
"""

import datetime
import json
import logging
import re
from dataclasses import asdict

import asyncpg

from .annas_client import BookMetadata

logger = logging.getLogger(__name__)

# Identifier types worth persisting (matches import_anna_bookinfo.py)
_ID_TYPES = {
    "isbn13", "isbn10", "oclc", "goodreads", "librarything", "ol",
    "gbooks", "lccn", "doi", "asin", "ipfs_cid", "md5", "sha1", "sha256",
    "ocaid", "issn", "wikidata",
}

_UPSERT_BOOK = """\
INSERT INTO books (
    urn, source, title, title_alt, authors, language, format, year,
    edition, publisher, description, size_bytes, content_type,
    cover_s3_key, cover_urls, added_date, metadata
) VALUES (
    $1, 'anna', $2, $3, $4, $5,
    $6, $7, $8, $9, $10,
    $11, $12, $13, $14,
    $15, $16
)
ON CONFLICT (urn) DO UPDATE SET
    title      = EXCLUDED.title,
    title_alt  = EXCLUDED.title_alt,
    authors    = EXCLUDED.authors,
    language   = EXCLUDED.language,
    format     = EXCLUDED.format,
    year       = EXCLUDED.year,
    edition    = EXCLUDED.edition,
    publisher  = EXCLUDED.publisher,
    description = EXCLUDED.description,
    size_bytes = EXCLUDED.size_bytes,
    content_type = EXCLUDED.content_type,
    cover_s3_key = COALESCE(EXCLUDED.cover_s3_key, books.cover_s3_key),
    cover_urls = EXCLUDED.cover_urls,
    added_date = EXCLUDED.added_date,
    metadata   = EXCLUDED.metadata,
    updated_at = now()
"""

_UPSERT_ID = """\
INSERT INTO book_identifiers (urn, type, value)
VALUES ($1, $2, $3)
ON CONFLICT DO NOTHING
"""


def _safe_year(val: str) -> int | None:
    if not val:
        return None
    m = re.search(r"\b(\d{4})\b", val)
    if m:
        y = int(m.group(1))
        return y if 0 < y < 3000 else None
    return None


def _safe_date(val: str) -> str | None:
    if not val:
        return None
    m = re.match(r"(\d{4}-\d{2}-\d{2})", val)
    return m.group(1) if m else None


class Database:
    """Thin async wrapper around the shared PostgreSQL books table."""

    def __init__(self, dsn: str):
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=4)
        logger.info("Connected to PostgreSQL")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            logger.info("Closed PostgreSQL connection pool")

    async def ping(self) -> bool:
        """Check if the database is reachable."""
        if not self._pool:
            return False
        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    async def upsert_book(
        self,
        hash: str,
        meta: BookMetadata,
        *,
        cover_s3_key: str | None = None,
    ) -> None:
        """Upsert a book and its identifiers into PostgreSQL."""
        if not self._pool:
            return

        urn = f"urn:anna:{hash}"

        # Cover URLs
        cover_urls: list[str] = []
        if meta.cover_url_best:
            cover_urls.append(meta.cover_url_best)
        cover_urls.extend(u for u in meta.cover_url_additional if u)

        # Collect identifiers
        id_rows: list[tuple[str, str, str]] = []
        for id_type, values in meta.identifiers_unified.items():
            if id_type in _ID_TYPES and values:
                for val in values:
                    if val:
                        id_rows.append((urn, id_type, str(val)))

        # Convert added_date string to a date object for asyncpg
        added_date_str = _safe_date(meta.added_date_best)
        added_date = datetime.date.fromisoformat(added_date_str) if added_date_str else None

        args = (
            urn,                                                    # $1
            meta.title_best or hash,                                # $2
            [t for t in meta.title_additional if t] or None,        # $3
            meta.author_best or None,                               # $4
            meta.language_codes or None,                            # $5
            meta.extension_best or None,                            # $6
            _safe_year(meta.year_best),                             # $7
            meta.edition_varia_best or None,                        # $8
            meta.publisher_best or None,                            # $9
            meta.stripped_description_best or None,                  # $10
            meta.filesize_best or None,                             # $11
            meta.content_type_best or None,                         # $12
            cover_s3_key,                                           # $13
            cover_urls or None,                                     # $14
            added_date,                                             # $15
            json.dumps(asdict(meta)),                               # $16
        )

        try:
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(_UPSERT_BOOK, *args)
                    if id_rows:
                        await conn.executemany(_UPSERT_ID, id_rows)
            logger.info("Upserted book %s (%s) to PostgreSQL", urn, meta.title_best)
        except Exception:
            logger.exception("Failed to upsert book %s to PostgreSQL", urn)
