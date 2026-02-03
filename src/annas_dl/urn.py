"""URN parsing for Anna's Archive identifiers."""

import re
from dataclasses import dataclass

# URN patterns
URN_ANNA_PATTERN = re.compile(r"^urn:anna:([a-f0-9]{32})$", re.IGNORECASE)
URN_OTHER_PATTERN = re.compile(r"^urn:([a-z]+):(.+)$", re.IGNORECASE)
HASH_PATTERN = re.compile(r"^[a-f0-9]{32}$", re.IGNORECASE)


class WrongResolverError(ValueError):
    """URN is valid but for a different resolver."""
    def __init__(self, source: str, urn: str):
        self.source = source
        self.urn = urn
        super().__init__(f"Wrong resolver: '{urn}' is a {source} URN, not anna")


class InvalidUrnError(ValueError):
    """URN format is invalid."""
    def __init__(self, urn: str, reason: str):
        self.urn = urn
        self.reason = reason
        super().__init__(f"Invalid URN: {reason}")


@dataclass
class ParsedUrn:
    """Parsed URN components."""

    source: str  # "anna"
    hash: str    # MD5 hash


def parse_urn(urn: str) -> ParsedUrn:
    """Parse a URN into its components.

    Accepts:
    - Full URN: urn:anna:abc123...
    - Raw hash: abc123... (32 hex chars)

    Raises:
        WrongResolverError: If URN is for a different source (royalroad, etc.)
        InvalidUrnError: If URN format is invalid
    """
    # Try full anna URN format
    if match := URN_ANNA_PATTERN.match(urn):
        return ParsedUrn(source="anna", hash=match.group(1).lower())

    # Check if it's a URN for a different source
    if match := URN_OTHER_PATTERN.match(urn):
        source = match.group(1).lower()
        if source != "anna":
            raise WrongResolverError(source, urn)
        # It's urn:anna but hash is malformed
        raise InvalidUrnError(urn, "anna URN must have 32-char hex hash")

    # Try raw hash (32 hex chars)
    if HASH_PATTERN.match(urn):
        return ParsedUrn(source="anna", hash=urn.lower())

    # Not a URN, not a valid hash
    raise InvalidUrnError(urn, "expected urn:anna:<32-char-hash> or raw MD5 hash")


def to_urn(hash: str) -> str:
    """Convert a hash to URN format."""
    return f"urn:anna:{hash.lower()}"
