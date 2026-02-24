"""Anna's Archive API client with mirror failover and DDoS-Guard bypass."""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

import httpx

if TYPE_CHECKING:
    from .search import SearchFilters, SearchResult

logger = logging.getLogger(__name__)

# Default user agent (will be replaced by FlareSolverr's)
DEFAULT_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64; rv:143.0) Gecko/20100101 Firefox/143.0"

# Available Anna's Archive mirror domains (in order of preference)
MIRROR_DOMAINS = [
    "https://annas-archive.li",
    "https://annas-archive.pm",
    "https://annas-archive.in",
]


@dataclass
class BookMetadata:
    """Metadata fetched from Anna's Archive API (/db/aarecord_elasticsearch/).

    All fields mirror the file_unified_data structure from the API.
    Lists default to empty, strings default to empty, ints default to 0.
    """

    # Core fields - always present as strings (may be empty)
    title_best: str = ""
    author_best: str = ""
    publisher_best: str = ""
    extension_best: str = ""  # format: epub, pdf, etc.
    year_best: str = ""

    # Additional/alternate values - always lists (may be empty)
    title_additional: list[str] = field(default_factory=list)
    author_additional: list[str] = field(default_factory=list)
    publisher_additional: list[str] = field(default_factory=list)
    extension_additional: list[str] = field(default_factory=list)
    year_additional: list[str] = field(default_factory=list)

    # Language - always lists
    language_codes: list[str] = field(default_factory=list)
    language_codes_detected: list[str] = field(default_factory=list)
    most_likely_language_codes: list[str] = field(default_factory=list)

    # Size
    filesize_best: int = 0
    filesize_additional: list[int] = field(default_factory=list)

    # Content info
    content_type_best: str = ""  # book_fiction, book_nonfiction, magazine
    stripped_description_best: str = ""
    stripped_description_additional: list[str] = field(default_factory=list)

    # Cover images
    cover_url_best: str = ""
    cover_url_additional: list[str] = field(default_factory=list)

    # Edition info
    edition_varia_best: str = ""
    edition_varia_additional: list[str] = field(default_factory=list)

    # Dates
    added_date_best: str = ""
    added_date_unified: dict[str, str] = field(default_factory=dict)

    # Identifiers (isbn, doi, oclc, etc.)
    identifiers_unified: dict[str, list[str]] = field(default_factory=dict)

    # IPFS - list of {"ipfs_cid": str, "from": str}
    ipfs_infos: list[dict[str, str]] = field(default_factory=list)

    # Availability flags - always ints
    has_aa_downloads: int = 0
    has_aa_exclusive_downloads: int = 0
    has_torrent_paths: int = 0
    has_scidb: int = 0

    # Problems/issues - always list/int
    problems: list[dict] = field(default_factory=list)
    has_meaningful_problems: int = 0

    # Other
    original_filename_best: str = ""
    comments_multiple: list[str] = field(default_factory=list)
    classifications_unified: dict = field(default_factory=dict)
    ol_is_primary_linked: bool = False

    def to_jsonld(self, urn: str, base_url: str = "") -> dict:
        """Convert to JSON-LD format using schema.org + Dublin Core.

        Args:
            urn: The URN identifier (e.g., "urn:anna:2d4d89...")
            base_url: Optional base URL for resource links

        Returns:
            JSON-LD dict compatible with schema.org/Book and Dublin Core
        """
        md5_hash = urn.split(":")[-1]

        # Map content_type to schema.org types
        schema_type = "Book"
        if self.content_type_best == "magazine":
            schema_type = "Periodical"

        # Build the JSON-LD document with multiple vocabularies
        doc: dict = {
            "@context": {
                "@vocab": "https://schema.org/",
                "dc": "http://purl.org/dc/terms/",
                "dcterms": "http://purl.org/dc/terms/",
                "bibo": "http://purl.org/ontology/bibo/",
                "prism": "http://prismstandard.org/namespaces/basic/2.0/",
            },
            "@type": schema_type,
            "@id": urn,
        }

        # === sameAs: All resolvable URN forms ===
        same_as = [
            f"urn:md5:{md5_hash}",  # MD5 URN (equivalent to urn:anna:)
        ]

        # Add identifier-based URNs and URLs
        for id_type, values in self.identifiers_unified.items():
            if not values:
                continue
            for v in values:
                if id_type == "isbn":
                    same_as.append(f"urn:isbn:{v}")
                elif id_type == "doi":
                    same_as.append(f"urn:doi:{v}")
                    same_as.append(f"https://doi.org/{v}")
                elif id_type == "oclc":
                    same_as.append(f"urn:oclc:{v}")
                    same_as.append(f"https://www.worldcat.org/oclc/{v}")
                elif id_type == "ol":
                    same_as.append(f"https://openlibrary.org/works/{v}")
                elif id_type == "asin":
                    same_as.append(f"https://www.amazon.com/dp/{v}")

        # Anna's Archive page
        same_as.append(f"https://annas-archive.li/md5/{md5_hash}")

        if same_as:
            doc["sameAs"] = same_as

        # === Schema.org fields ===
        if self.title_best:
            doc["name"] = self.title_best
        if self.author_best:
            doc["author"] = {"@type": "Person", "name": self.author_best}
        if self.publisher_best:
            doc["publisher"] = {"@type": "Organization", "name": self.publisher_best}
        if self.year_best:
            doc["datePublished"] = self.year_best
        if self.stripped_description_best:
            doc["description"] = self.stripped_description_best
        if self.language_codes:
            doc["inLanguage"] = self.language_codes[0]
        if self.extension_best:
            doc["encodingFormat"] = f"application/{self.extension_best}"
        if self.filesize_best:
            doc["contentSize"] = f"{self.filesize_best} bytes"
        if self.cover_url_best:
            doc["image"] = self.cover_url_best
        if self.edition_varia_best:
            doc["bookEdition"] = self.edition_varia_best

        # ISBN as schema.org property
        isbns = self.identifiers_unified.get("isbn", [])
        if isbns:
            doc["isbn"] = isbns[0] if len(isbns) == 1 else isbns

        # === Dublin Core fields (parallel representation) ===
        if self.title_best:
            doc["dc:title"] = self.title_best
        if self.author_best:
            doc["dc:creator"] = self.author_best
        if self.publisher_best:
            doc["dc:publisher"] = self.publisher_best
        if self.year_best:
            doc["dc:date"] = self.year_best
        if self.language_codes:
            doc["dc:language"] = self.language_codes[0]
        if self.stripped_description_best:
            doc["dc:description"] = self.stripped_description_best
        if self.extension_best:
            doc["dc:format"] = f"application/{self.extension_best}"

        # DC identifiers
        dc_identifiers = [f"urn:md5:{md5_hash}"]
        for id_type, values in self.identifiers_unified.items():
            for v in values:
                if id_type == "isbn":
                    dc_identifiers.append(f"urn:isbn:{v}")
                elif id_type == "doi":
                    dc_identifiers.append(f"doi:{v}")
                elif id_type == "oclc":
                    dc_identifiers.append(f"oclc:{v}")
        doc["dc:identifier"] = dc_identifiers

        # === IPFS content URLs (canonical ipfs:// URIs) ===
        if self.ipfs_infos:
            ipfs_urls = list({
                f"ipfs://{info['ipfs_cid']}"
                for info in self.ipfs_infos
                if info.get("ipfs_cid")
            })
            if ipfs_urls:
                doc["contentUrl"] = ipfs_urls

        # === Structured identifiers (schema.org PropertyValue) ===
        identifiers = [
            {"@type": "PropertyValue", "propertyID": "md5", "value": md5_hash}
        ]
        for id_type, values in self.identifiers_unified.items():
            if not values:
                continue
            for v in values:
                if id_type in ("isbn", "doi", "oclc", "issn", "lccn"):
                    identifiers.append(
                        {"@type": "PropertyValue", "propertyID": id_type, "value": v}
                    )
        doc["identifier"] = identifiers

        # Resource URL if base_url provided
        if base_url:
            doc["url"] = f"{base_url}/urn/{urn}"

        # Additional authors
        if self.author_additional:
            existing_author = doc.get("author")
            authors = [existing_author] if existing_author else []
            authors.extend({"@type": "Person", "name": name} for name in self.author_additional)
            doc["author"] = authors if len(authors) > 1 else authors[0] if authors else None
            # Also update DC creator
            all_authors = [self.author_best] + self.author_additional if self.author_best else self.author_additional
            doc["dc:creator"] = all_authors if len(all_authors) > 1 else all_authors[0] if all_authors else None

        # Content type hint
        if self.content_type_best:
            doc["additionalType"] = f"https://annas-archive.li/types/{self.content_type_best}"

        return doc


@dataclass
class FastDownloadResult:
    """Result from fast download API including quota info."""

    download_url: str
    downloads_left: int
    downloads_per_day: int
    downloads_done_today: int


class AnnasClientError(Exception):
    """Error from Anna's Archive API."""

    # Known error messages from Anna's Archive API
    # See: https://software.annas-archive.li/AnnaArchivist/annas-archive/-/raw/main/allthethings/dyn/views.py
    ERROR_INVALID_MD5 = "Invalid md5"
    ERROR_INVALID_KEY = "Invalid secret key"
    ERROR_FETCH_ERROR = "Error during fetching"
    ERROR_NOT_FOUND = "Record not found"
    ERROR_NOT_MEMBER = "Not a member"
    ERROR_INVALID_INDICES = "Invalid domain_index or path_index"
    ERROR_NO_DOWNLOADS = "No downloads left"


class NoDownloadsLeftError(AnnasClientError):
    """Fast downloads exhausted (429)."""


class InvalidKeyError(AnnasClientError):
    """Invalid API secret key (401)."""


class NotMemberError(AnnasClientError):
    """Account is not a member (403)."""


class RecordNotFoundError(AnnasClientError):
    """Book not found in Anna's Archive (404)."""


class LoginError(AnnasClientError):
    """Failed to log in with secret key."""


class DDoSGuardError(AnnasClientError):
    """DDoS-Guard challenge encountered."""


@dataclass
class Session:
    """Authenticated session with Anna's Archive including DDoS-Guard cookies."""

    account_id: str
    cookies: dict[str, str] = field(default_factory=dict)
    user_agent: str = DEFAULT_USER_AGENT

    def as_cookie_header(self) -> str:
        """Return cookie header value for requests."""
        return "; ".join(f"{k}={v}" for k, v in self.cookies.items())

    def as_cookies_dict(self) -> dict[str, str]:
        """Return cookies dict for httpx."""
        return self.cookies.copy()


class AnnasClient:
    """Async client for Anna's Archive with automatic mirror failover and DDoS-Guard bypass."""

    def __init__(
        self,
        http: httpx.AsyncClient,
        timeout: float = 15.0,
        flaresolverr_url: str | None = None,
        secret_key: str | None = None,
    ):
        self._http = http
        self._timeout = timeout
        self._flaresolverr_url = flaresolverr_url
        self._secret_key = secret_key
        self._current_domain_idx = 0
        self._session: Session | None = None

    @classmethod
    def create(
        cls,
        timeout: float = 15.0,
        flaresolverr_url: str | None = None,
        secret_key: str | None = None,
    ) -> Self:
        """Create a client with a new HTTP client."""
        http = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
            headers={"User-Agent": DEFAULT_USER_AGENT},
        )
        return cls(http, timeout, flaresolverr_url, secret_key)

    @property
    def current_domain(self) -> str:
        """Get the current active domain."""
        return MIRROR_DOMAINS[self._current_domain_idx % len(MIRROR_DOMAINS)]

    @property
    def session(self) -> Session | None:
        """Get current session if authenticated."""
        return self._session

    def _rotate_domain(self) -> str:
        """Rotate to the next domain, returns the new domain."""
        old_domain = self.current_domain
        self._current_domain_idx = (self._current_domain_idx + 1) % len(MIRROR_DOMAINS)
        new_domain = self.current_domain
        logger.info(
            "Rotating Anna's Archive mirror: %s -> %s", old_domain, new_domain
        )
        return new_domain

    @staticmethod
    def _is_recoverable_error(exc: Exception) -> bool:
        """Check if an error is recoverable by trying another domain."""
        if isinstance(exc, httpx.TimeoutException | httpx.ConnectError):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code >= 500
        return False

    @staticmethod
    def _is_ddos_guard_challenge(response: httpx.Response) -> bool:
        """Check if response is a DDoS-Guard JS challenge."""
        if response.status_code != 403:
            return False
        # DDoS-Guard returns HTML with challenge script
        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type:
            return False
        return "ddos-guard" in response.text.lower()

    @staticmethod
    def _is_auth_required(response: httpx.Response) -> bool:
        """Check if response indicates authentication is required."""
        if response.status_code != 403:
            return False
        # "Not a member" error from Anna's Archive
        return "not a member" in response.text.lower()

    async def _refresh_session_via_flaresolverr(self) -> Session:
        """Use FlareSolverr to bypass DDoS-Guard and login.

        FlareSolverr handles the JS challenge, then we POST the login form
        to get both DDoS-Guard cookies and the session cookie.
        """
        if not self._flaresolverr_url:
            raise DDoSGuardError("DDoS-Guard challenge but no FlareSolverr configured")
        if not self._secret_key:
            raise DDoSGuardError("DDoS-Guard challenge but no secret key configured")

        logger.info("Refreshing session via FlareSolverr at %s", self._flaresolverr_url)

        # POST login form via FlareSolverr - this bypasses DDoS-Guard and logs in
        try:
            response = await self._http.post(
                f"{self._flaresolverr_url}/v1",
                json={
                    "cmd": "request.post",
                    "url": f"{self.current_domain}/account/",
                    "postData": f"key={self._secret_key}",
                    "maxTimeout": 60000,
                },
                timeout=90.0,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "ok":
                raise DDoSGuardError(f"FlareSolverr error: {data.get('message')}")

            solution = data.get("solution", {})
            cookies_list = solution.get("cookies", [])
            user_agent = solution.get("userAgent", DEFAULT_USER_AGENT)

            # Convert cookies list to dict
            cookies = {c["name"]: c["value"] for c in cookies_list}

            if "aa_account_id2" not in cookies:
                raise LoginError("Login via FlareSolverr did not return session cookie")

            # Extract account_id from secret_key (first 7 chars)
            account_id = self._secret_key[:7]

            self._session = Session(
                account_id=account_id,
                cookies=cookies,
                user_agent=user_agent,
            )

            logger.info(
                "Session refreshed for account %s with %d cookies",
                account_id,
                len(cookies),
            )
            return self._session

        except httpx.HTTPError as exc:
            raise DDoSGuardError(f"FlareSolverr request failed: {exc}") from exc

    async def _request(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> httpx.Response:
        """Make a request with session cookies, auto-refreshing on DDoS-Guard challenge."""
        # Apply session cookies and user-agent if we have a session
        if self._session:
            headers = kwargs.pop("headers", {})
            headers["Cookie"] = self._session.as_cookie_header()
            headers["User-Agent"] = self._session.user_agent
            kwargs["headers"] = headers

        response = await self._http.request(method, url, **kwargs)

        # Check for DDoS-Guard challenge or auth required (when no session)
        needs_refresh = (
            self._is_ddos_guard_challenge(response)
            or (self._is_auth_required(response) and not self._session)
        )

        if needs_refresh:
            if self._is_ddos_guard_challenge(response):
                logger.warning("DDoS-Guard challenge detected, refreshing session")
            else:
                logger.warning("Auth required, refreshing session via FlareSolverr")

            await self._refresh_session_via_flaresolverr()

            # Retry with new session
            if self._session:
                headers = kwargs.pop("headers", {})
                headers["Cookie"] = self._session.as_cookie_header()
                headers["User-Agent"] = self._session.user_agent
                kwargs["headers"] = headers
            response = await self._http.request(method, url, **kwargs)

        return response

    async def get_download_url(
        self, hash: str, secret_key: str, domain_index: int | None = None
    ) -> FastDownloadResult:
        """Get download URL for a book with automatic failover.

        Args:
            hash: MD5 hash of the book
            secret_key: Anna's Archive API key
            domain_index: Optional CDN server index (0-9+)

        Returns:
            FastDownloadResult with URL and quota info
        """
        last_error: Exception | None = None

        for attempt in range(len(MIRROR_DOMAINS)):
            domain = self.current_domain
            url = f"{domain}/dyn/api/fast_download.json?md5={hash}&key={secret_key}"
            if domain_index is not None:
                url += f"&domain_index={domain_index}"

            logger.debug(
                "Fetching download URL (attempt %d): %s",
                attempt + 1,
                url.replace(secret_key, "***"),
            )

            try:
                response = await self._request("GET", url, timeout=self._timeout)

                # Check for rate limiting before raise_for_status() so we can
                # parse the JSON body for the specific error message
                if response.status_code == 429:
                    try:
                        data = response.json()
                        error = data.get("error", "")
                    except Exception:
                        error = ""
                    if error == AnnasClientError.ERROR_NO_DOWNLOADS or not error:
                        raise NoDownloadsLeftError(error or "Rate limited (429)")
                    raise AnnasClientError(f"API error: {error}")

                response.raise_for_status()
                data = response.json()

                if error := data.get("error"):
                    # Raise specific exceptions for known error types
                    if error == AnnasClientError.ERROR_NO_DOWNLOADS:
                        raise NoDownloadsLeftError(error)
                    if error == AnnasClientError.ERROR_INVALID_KEY:
                        raise InvalidKeyError(error)
                    if error == AnnasClientError.ERROR_NOT_MEMBER:
                        raise NotMemberError(error)
                    if error == AnnasClientError.ERROR_NOT_FOUND:
                        raise RecordNotFoundError(error)
                    # Generic error for others
                    raise AnnasClientError(f"API error: {error}")

                download_url = data.get("download_url")
                if not download_url:
                    raise AnnasClientError("No download URL in response")

                # Extract quota info
                quota = data.get("account_fast_download_info", {})
                return FastDownloadResult(
                    download_url=download_url,
                    downloads_left=quota.get("downloads_left", 0),
                    downloads_per_day=quota.get("downloads_per_day", 0),
                    downloads_done_today=quota.get("downloads_done_today", 0),
                )

            except (DDoSGuardError, NoDownloadsLeftError, InvalidKeyError, NotMemberError, RecordNotFoundError):
                raise
            except Exception as exc:
                logger.warning(
                    "Download URL request failed (domain=%s, attempt=%d): %s",
                    domain,
                    attempt + 1,
                    exc,
                )

                if self._is_recoverable_error(exc) and attempt < len(MIRROR_DOMAINS) - 1:
                    self._rotate_domain()
                    last_error = exc
                    continue

                raise AnnasClientError(str(exc)) from exc

        raise AnnasClientError(
            f"All mirrors failed: {last_error}"
        ) from last_error

    async def fetch_metadata(self, hash: str) -> BookMetadata:
        """Fetch metadata by hash with automatic failover and DDoS-Guard bypass."""
        last_error: Exception | None = None

        for attempt in range(len(MIRROR_DOMAINS)):
            domain = self.current_domain
            url = f"{domain}/db/aarecord_elasticsearch/md5:{hash}.json"

            logger.debug("Fetching metadata (attempt %d): %s", attempt + 1, url)

            try:
                response = await self._request("GET", url, timeout=self._timeout)
                if response.status_code == 404:
                    raise RecordNotFoundError(f"No record found for md5:{hash}")
                response.raise_for_status()
                data = response.json()

                unified = data.get("file_unified_data", {})

                return BookMetadata(
                    # Core fields
                    title_best=unified.get("title_best", ""),
                    author_best=unified.get("author_best", ""),
                    publisher_best=unified.get("publisher_best", ""),
                    extension_best=unified.get("extension_best", ""),
                    year_best=unified.get("year_best", ""),
                    # Additional values
                    title_additional=unified.get("title_additional") or [],
                    author_additional=unified.get("author_additional") or [],
                    publisher_additional=unified.get("publisher_additional") or [],
                    extension_additional=unified.get("extension_additional") or [],
                    year_additional=unified.get("year_additional") or [],
                    # Language
                    language_codes=unified.get("language_codes") or [],
                    language_codes_detected=unified.get("language_codes_detected") or [],
                    most_likely_language_codes=unified.get("most_likely_language_codes") or [],
                    # Size
                    filesize_best=unified.get("filesize_best") or 0,
                    filesize_additional=unified.get("filesize_additional") or [],
                    # Content info
                    content_type_best=unified.get("content_type_best") or "",
                    stripped_description_best=unified.get("stripped_description_best") or "",
                    stripped_description_additional=unified.get("stripped_description_additional") or [],
                    # Cover images
                    cover_url_best=unified.get("cover_url_best") or "",
                    cover_url_additional=unified.get("cover_url_additional") or [],
                    # Edition
                    edition_varia_best=unified.get("edition_varia_best") or "",
                    edition_varia_additional=unified.get("edition_varia_additional") or [],
                    # Dates
                    added_date_best=unified.get("added_date_best") or "",
                    added_date_unified=unified.get("added_date_unified") or {},
                    # Identifiers
                    identifiers_unified=unified.get("identifiers_unified") or {},
                    # IPFS
                    ipfs_infos=unified.get("ipfs_infos") or [],
                    # Availability flags
                    has_aa_downloads=unified.get("has_aa_downloads") or 0,
                    has_aa_exclusive_downloads=unified.get("has_aa_exclusive_downloads") or 0,
                    has_torrent_paths=unified.get("has_torrent_paths") or 0,
                    has_scidb=unified.get("has_scidb") or 0,
                    # Problems
                    problems=unified.get("problems") or [],
                    has_meaningful_problems=unified.get("has_meaningful_problems") or 0,
                    # Other
                    original_filename_best=unified.get("original_filename_best") or "",
                    comments_multiple=unified.get("comments_multiple") or [],
                    classifications_unified=unified.get("classifications_unified") or {},
                    ol_is_primary_linked=bool(unified.get("ol_is_primary_linked")),
                )

            except (DDoSGuardError, RecordNotFoundError):
                raise
            except Exception as exc:
                logger.warning(
                    "Metadata request failed (domain=%s, attempt=%d): %s",
                    domain,
                    attempt + 1,
                    exc,
                )

                if self._is_recoverable_error(exc) and attempt < len(MIRROR_DOMAINS) - 1:
                    self._rotate_domain()
                    last_error = exc
                    continue

                raise AnnasClientError(str(exc)) from exc

        raise AnnasClientError(
            f"All mirrors failed: {last_error}"
        ) from last_error

    async def fetch_torrent_paths(self, hash: str) -> list[dict]:
        """Fetch torrent path info from metadata's additional field.

        Returns list of dicts with keys: collection, torrent_path, file_level1, file_level2.
        """
        last_error: Exception | None = None

        for attempt in range(len(MIRROR_DOMAINS)):
            domain = self.current_domain
            url = f"{domain}/db/aarecord_elasticsearch/md5:{hash}.json"

            try:
                response = await self._request("GET", url, timeout=self._timeout)
                response.raise_for_status()
                data = response.json()

                additional = data.get("additional", {})
                return additional.get("torrent_paths", [])

            except DDoSGuardError:
                raise
            except Exception as exc:
                if self._is_recoverable_error(exc) and attempt < len(MIRROR_DOMAINS) - 1:
                    self._rotate_domain()
                    last_error = exc
                    continue
                raise AnnasClientError(str(exc)) from exc

        raise AnnasClientError(f"All mirrors failed: {last_error}") from last_error

    async def login(self, secret_key: str) -> Session:
        """Log in with secret key and obtain session cookie.

        Posts the secret key to /account and extracts the session cookie
        from the response. This enables access to cookie-authenticated
        endpoints (lists, comments, account settings, etc.).

        Note: This direct login will fail if DDoS-Guard is active.
        Use FlareSolverr integration for automatic bypass.

        Args:
            secret_key: Anna's Archive account secret key

        Returns:
            Session object with cookies for authenticated requests

        Raises:
            LoginError: If login fails (invalid key, network error, etc.)
            DDoSGuardError: If DDoS-Guard challenge is encountered
        """
        last_error: Exception | None = None

        for attempt in range(len(MIRROR_DOMAINS)):
            domain = self.current_domain
            url = f"{domain}/account/"

            logger.debug("Login attempt %d to %s", attempt + 1, domain)

            try:
                # POST the secret key as form data
                response = await self._http.post(
                    url,
                    data={"key": secret_key},
                    timeout=self._timeout,
                    follow_redirects=False,  # We want to capture the Set-Cookie header
                )

                # Check for DDoS-Guard challenge
                if self._is_ddos_guard_challenge(response):
                    raise DDoSGuardError(
                        "DDoS-Guard challenge during login. "
                        "Configure FlareSolverr for automatic bypass."
                    )

                # Successful login returns 302 redirect with Set-Cookie
                if response.status_code not in (302, 303):
                    # Check if we got an error page (200 with invalid_key message)
                    if response.status_code == 200:
                        raise LoginError("Invalid secret key")
                    raise LoginError(f"Unexpected status code: {response.status_code}")

                # Extract all cookies from response
                cookies = dict(response.cookies.items())

                if not any(name.startswith("aa_") for name in cookies):
                    raise LoginError("No session cookie in response")

                # Extract account_id from secret_key (first 7 chars)
                account_id = secret_key[:7]

                logger.info("Login successful for account %s via %s", account_id, domain)

                self._session = Session(
                    account_id=account_id,
                    cookies=cookies,
                )
                return self._session

            except LoginError | DDoSGuardError:
                raise
            except Exception as exc:
                logger.warning(
                    "Login failed (domain=%s, attempt=%d): %s",
                    domain,
                    attempt + 1,
                    exc,
                )

                if self._is_recoverable_error(exc) and attempt < len(MIRROR_DOMAINS) - 1:
                    self._rotate_domain()
                    last_error = exc
                    continue

                raise LoginError(str(exc)) from exc

        raise LoginError(f"All mirrors failed: {last_error}") from last_error

    async def search(self, filters: "SearchFilters") -> list["SearchResult"]:
        """Search Anna's Archive with automatic failover and DDoS-Guard bypass.

        Uses the table display mode for cleaner HTML parsing.

        Args:
            filters: Search filters (query, content types, formats, etc.)

        Returns:
            List of SearchResult objects
        """
        from .search import build_search_url, parse_search_results

        last_error: Exception | None = None

        for attempt in range(len(MIRROR_DOMAINS)):
            domain = self.current_domain
            url = build_search_url(domain, filters)

            logger.debug("Search attempt %d: %s", attempt + 1, url)

            try:
                response = await self._request("GET", url, timeout=self._timeout)
                response.raise_for_status()

                results = parse_search_results(response.text, domain)
                logger.info(
                    "Search returned %d results (domain=%s, query=%s)",
                    len(results),
                    domain,
                    filters.query,
                )
                return results

            except DDoSGuardError:
                raise
            except Exception as exc:
                logger.warning(
                    "Search failed (domain=%s, attempt=%d): %s",
                    domain,
                    attempt + 1,
                    exc,
                )

                if self._is_recoverable_error(exc) and attempt < len(MIRROR_DOMAINS) - 1:
                    self._rotate_domain()
                    last_error = exc
                    continue

                raise AnnasClientError(str(exc)) from exc

        raise AnnasClientError(f"All mirrors failed: {last_error}") from last_error

    async def close(self):
        """Close the HTTP client."""
        await self._http.aclose()


def _format_file_size(bytes_: int) -> str | None:
    """Format file size in human-readable form."""
    if bytes_ <= 0:
        return None
    for unit, threshold in [("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)]:
        if bytes_ >= threshold:
            return f"{bytes_ / threshold:.1f} {unit}"
    return f"{bytes_} bytes"
