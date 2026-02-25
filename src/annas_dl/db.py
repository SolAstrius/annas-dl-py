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

# ISO 639-1 (2-letter) → ISO 639-3 (3-letter) mapping.
# Anna's Archive uses 639-1; we store 639-3 in the database.
_ISO1_TO_ISO3: dict[str, str] = {
    "aa": "aar", "ab": "abk", "af": "afr", "ak": "aka", "am": "amh",
    "an": "arg", "ar": "ara", "as": "asm", "av": "ava", "ay": "aym",
    "az": "aze", "ba": "bak", "be": "bel", "bg": "bul", "bh": "bih",
    "bi": "bis", "bm": "bam", "bn": "ben", "bo": "bod", "br": "bre",
    "bs": "bos", "ca": "cat", "ce": "che", "ch": "cha", "co": "cos",
    "cr": "cre", "cs": "ces", "cu": "chu", "cv": "chv", "cy": "cym",
    "da": "dan", "de": "deu", "dv": "div", "dz": "dzo", "ee": "ewe",
    "el": "ell", "en": "eng", "eo": "epo", "es": "spa", "et": "est",
    "eu": "eus", "fa": "fas", "ff": "ful", "fi": "fin", "fj": "fij",
    "fo": "fao", "fr": "fra", "fy": "fry", "ga": "gle", "gd": "gla",
    "gl": "glg", "gn": "grn", "gu": "guj", "gv": "glv", "ha": "hau",
    "he": "heb", "hi": "hin", "ho": "hmo", "hr": "hrv", "ht": "hat",
    "hu": "hun", "hy": "hye", "hz": "her", "ia": "ina", "id": "ind",
    "ie": "ile", "ig": "ibo", "ii": "iii", "ik": "ipk", "io": "ido",
    "is": "isl", "it": "ita", "iu": "iku", "ja": "jpn", "jv": "jav",
    "ka": "kat", "kg": "kon", "ki": "kik", "kj": "kua", "kk": "kaz",
    "kl": "kal", "km": "khm", "kn": "kan", "ko": "kor", "kr": "kau",
    "ks": "kas", "ku": "kur", "kv": "kom", "kw": "cor", "ky": "kir",
    "la": "lat", "lb": "ltz", "lg": "lug", "li": "lim", "ln": "lin",
    "lo": "lao", "lt": "lit", "lu": "lub", "lv": "lav", "mg": "mlg",
    "mh": "mah", "mi": "mri", "mk": "mkd", "ml": "mal", "mn": "mon",
    "mr": "mar", "ms": "msa", "mt": "mlt", "my": "mya", "na": "nau",
    "nb": "nob", "nd": "nde", "ne": "nep", "ng": "ndo", "nl": "nld",
    "nn": "nno", "no": "nor", "nr": "nbl", "nv": "nav", "ny": "nya",
    "oc": "oci", "oj": "oji", "om": "orm", "or": "ori", "os": "oss",
    "pa": "pan", "pi": "pli", "pl": "pol", "ps": "pus", "pt": "por",
    "qu": "que", "rm": "roh", "rn": "run", "ro": "ron", "ru": "rus",
    "rw": "kin", "sa": "san", "sc": "srd", "sd": "snd", "se": "sme",
    "sg": "sag", "si": "sin", "sk": "slk", "sl": "slv", "sm": "smo",
    "sn": "sna", "so": "som", "sq": "sqi", "sr": "srp", "ss": "ssw",
    "st": "sot", "su": "sun", "sv": "swe", "sw": "swa", "ta": "tam",
    "te": "tel", "tg": "tgk", "th": "tha", "ti": "tir", "tk": "tuk",
    "tl": "tgl", "tn": "tsn", "to": "ton", "tr": "tur", "ts": "tso",
    "tt": "tat", "tw": "twi", "ty": "tah", "ug": "uig", "uk": "ukr",
    "ur": "urd", "uz": "uzb", "ve": "ven", "vi": "vie", "vo": "vol",
    "wa": "wln", "wo": "wol", "xh": "xho", "yi": "yid", "yo": "yor",
    "za": "zha", "zh": "zho", "zu": "zul",
}


def _to_iso3(codes: list[str]) -> list[str]:
    """Convert language codes to ISO 639-3. Passes through 3-letter codes as-is."""
    result = []
    for code in codes:
        code = code.strip().lower()
        if len(code) == 2:
            result.append(_ISO1_TO_ISO3.get(code, code))
        elif len(code) == 3:
            result.append(code)  # already 639-3
        elif code:
            result.append(code)  # unknown format, pass through
    return result

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
            _to_iso3(meta.language_codes) or None,                     # $5
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
