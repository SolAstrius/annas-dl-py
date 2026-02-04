"""Anna's Archive search via table view scraping.

Uses the table display mode (?display=table) which provides cleaner,
positional HTML structure compared to the card view.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlencode

from selectolax.parser import HTMLParser

logger = logging.getLogger(__name__)


class ContentType(str, Enum):
    """Content type filter values."""

    BOOK_FICTION = "book_fiction"
    BOOK_NONFICTION = "book_nonfiction"
    BOOK_UNKNOWN = "book_unknown"
    MAGAZINE = "magazine"
    BOOK_COMIC = "book_comic"


class SortField(str, Enum):
    """Sort field options."""

    RELEVANCE = ""
    NEWEST = "newest"
    OLDEST = "oldest"
    LARGEST = "largest"
    SMALLEST = "smallest"


# Map emoji prefixes to content types
CONTENT_TYPE_MAP = {
    "📕": ContentType.BOOK_FICTION,
    "📗": ContentType.BOOK_NONFICTION,
    "📘": ContentType.BOOK_NONFICTION,
    "📙": ContentType.BOOK_UNKNOWN,
    "📰": ContentType.MAGAZINE,
    "🎨": ContentType.BOOK_COMIC,
}


@dataclass
class SearchFilters:
    """Search query parameters."""

    query: str
    content_types: list[ContentType] = field(default_factory=list)
    formats: list[str] = field(default_factory=list)  # pdf, epub, mobi, etc.
    languages: list[str] = field(default_factory=list)  # en, ru, zh, etc.
    page: int = 1
    sort: SortField = SortField.RELEVANCE


@dataclass
class SearchResult:
    """A single search result from Anna's Archive."""

    id: str  # URN: urn:anna:<hash>
    hash: str  # MD5 hash
    title: str
    author: str | None = None
    publisher: str | None = None
    year: str | None = None
    language: str | None = None
    content_type: ContentType | None = None
    format: str | None = None
    size: str | None = None
    url: str | None = None  # Full URL to Anna's Archive page


@dataclass
class SearchResponse:
    """Paginated search response."""

    results: list[SearchResult]
    query: str
    page: int
    total_on_page: int


def build_search_url(domain: str, filters: SearchFilters) -> str:
    """Build search URL with table display mode.

    Args:
        domain: Base domain (e.g., https://annas-archive.li)
        filters: Search filters

    Returns:
        Full search URL with query parameters
    """
    params: list[tuple[str, str]] = [
        ("q", filters.query),
        ("display", "table"),
    ]

    if filters.page > 1:
        params.append(("page", str(filters.page)))

    if filters.sort and filters.sort != SortField.RELEVANCE:
        params.append(("sort", filters.sort.value))

    for ct in filters.content_types:
        params.append(("content", ct.value))

    for fmt in filters.formats:
        params.append(("ext", fmt.lower()))

    for lang in filters.languages:
        params.append(("lang", lang.lower()))

    return f"{domain}/search?{urlencode(params)}"


def _get_cell_text(row, index: int) -> str:
    """Extract text from a table cell by index."""
    cells = row.css("td")
    if index >= len(cells):
        return ""

    # Get the span inside the cell (where actual content is)
    span = cells[index].css_first("span")
    if span is None:
        return ""

    # Get text, but only from the first text node (not nested .text-gray-500 spans)
    # Clone and remove nested spans to get just the main text
    text = span.text(strip=True, separator=" ")

    # If there are nested gray spans, they contain secondary info
    # We want only the primary text before them
    gray_spans = span.css("span.text-gray-500")
    for gs in gray_spans:
        gs_text = gs.text(strip=True)
        if gs_text and gs_text in text:
            # Remove the secondary text
            text = text.replace(gs_text, "").strip()

    return text.strip()


def _extract_hash_from_row(row) -> str | None:
    """Extract MD5 hash from the first cell's link."""
    link = row.css_first("td:first-child a[href^='/md5/']")
    if link is None:
        return None

    href = link.attributes.get("href", "")
    if href.startswith("/md5/"):
        return href[5:].lower()  # Remove "/md5/" prefix
    return None


def _parse_content_type(text: str) -> ContentType | None:
    """Parse content type from cell text like '📕 Book (fiction)'."""
    if not text:
        return None

    # Check emoji prefix
    for emoji, ct in CONTENT_TYPE_MAP.items():
        if text.startswith(emoji):
            return ct

    # Fallback to text matching
    text_lower = text.lower()
    if "fiction" in text_lower:
        if "non" in text_lower:
            return ContentType.BOOK_NONFICTION
        return ContentType.BOOK_FICTION
    if "magazine" in text_lower:
        return ContentType.MAGAZINE
    if "comic" in text_lower:
        return ContentType.BOOK_COMIC

    return ContentType.BOOK_UNKNOWN


def parse_search_results(html: str, domain: str = "https://annas-archive.li") -> list[SearchResult]:
    """Parse search results from table view HTML.

    Table columns (0-indexed):
    0: Cover thumbnail + hash in href
    1: Title
    2: Author
    3: Publisher (may include year)
    4: Year
    5: Filenames (internal paths)
    6: Sources (🚀/lgli/lgrs/etc)
    7: Language
    8: Content type (📕 Book (fiction), etc)
    9: Format (epub, pdf, etc)
    10: Size (0.2MB, 7.9MB)
    11: (empty/ISBN)

    Args:
        html: Raw HTML from search page
        domain: Base domain for building URLs

    Returns:
        List of SearchResult objects
    """
    tree = HTMLParser(html)
    results: list[SearchResult] = []

    # Find all table rows with the group class
    for row in tree.css("table.text-sm tr.group"):
        hash = _extract_hash_from_row(row)
        if not hash:
            continue

        # Validate hash format (32 hex chars)
        if not re.match(r"^[a-f0-9]{32}$", hash, re.IGNORECASE):
            logger.warning("Invalid hash format: %s", hash)
            continue

        title = _get_cell_text(row, 1)
        if not title:
            continue  # Skip rows without titles

        author = _get_cell_text(row, 2) or None
        publisher = _get_cell_text(row, 3) or None
        year = _get_cell_text(row, 4) or None
        language = _get_cell_text(row, 7) or None
        content_type_text = _get_cell_text(row, 8)
        format_text = _get_cell_text(row, 9) or None
        size = _get_cell_text(row, 10) or None

        results.append(
            SearchResult(
                id=f"urn:anna:{hash}",
                hash=hash,
                title=title,
                author=author,
                publisher=publisher,
                year=year,
                language=language,
                content_type=_parse_content_type(content_type_text),
                format=format_text.lower() if format_text else None,
                size=size,
                url=f"{domain}/md5/{hash}",
            )
        )

    logger.info("Parsed %d search results from HTML", len(results))
    return results
