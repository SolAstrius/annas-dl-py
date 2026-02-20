"""BitTorrent selective download using libtorrent.

Provides a persistent libtorrent session that keeps DHT routing tables warm
across downloads, dramatically reducing cold-start latency. Downloads only
the target file from multi-file torrents using piece-level selection.

The .torrent files are cached in-memory (LRU) to avoid re-fetching.
"""

import asyncio
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from .config import Settings

logger = logging.getLogger(__name__)

# Default trackers for faster peer discovery (from Anna's Archive magnet links)
DEFAULT_TRACKERS = [
    "udp://tracker.opentrackr.org:1337/announce",
    "udp://open.tracker.cl:1337/announce",
    "udp://tracker.openbittorrent.com:6969/announce",
    "udp://open.demonii.com:1337/announce",
    "udp://open.stealth.si:80/announce",
    "udp://exodus.desync.com:6969/announce",
]


@dataclass
class TorrentPath:
    """Torrent path info from Anna's Archive metadata."""

    collection: str  # e.g. "libgen_rs_non_fic"
    torrent_path: str  # e.g. "external/libgen_rs_non_fic/r_626000.torrent"
    file_level1: str  # filename inside torrent dir (usually MD5 hash)
    file_level2: str = ""  # optional second level


@dataclass
class TorrentDownloadResult:
    """Result from a torrent download."""

    content: bytes
    file_path: str  # path within the torrent
    duration_s: float
    peers: int
    seeds: int


class TorrentSession:
    """Persistent libtorrent session with warm DHT and .torrent caching.

    Designed as a singleton — create once at startup, reuse across downloads.
    The DHT routing table stays warm between downloads, avoiding the ~3-18s
    cold start penalty on each download.
    """

    def __init__(self, settings: "Settings"):
        import libtorrent as lt  # type: ignore[import-not-found]

        self._settings = settings
        self._lt = lt

        # Configure session with optimized settings
        sess_params = lt.session_params()
        sess_params.settings = self._build_settings(lt)

        self._session = lt.session(sess_params)
        self._session.add_dht_router("router.bittorrent.com", 6881)
        self._session.add_dht_router("router.utorrent.com", 6881)
        self._session.add_dht_router("dht.transmissionbt.com", 6881)

        # In-memory .torrent file cache (torrent_path -> bytes)
        self._torrent_cache: dict[str, bytes] = {}
        self._max_cache_size = 200  # max cached .torrent files

        logger.info(
            "libtorrent session started (version=%s)",
            lt.version,
        )

    @staticmethod
    def _build_settings(lt) -> dict:  # noqa: ANN001
        """Build optimized libtorrent settings dict."""
        s = lt.default_settings()
        s["enable_dht"] = True
        s["enable_lsd"] = True
        s["enable_natpmp"] = False
        s["enable_upnp"] = False
        s["connections_limit"] = 200
        s["active_downloads"] = 3
        return s

    def _cache_torrent(self, torrent_path: str, data: bytes) -> None:
        """Cache a .torrent file, evicting oldest if over limit."""
        if len(self._torrent_cache) >= self._max_cache_size:
            # Evict oldest entry
            oldest_key = next(iter(self._torrent_cache))
            del self._torrent_cache[oldest_key]
        self._torrent_cache[torrent_path] = data

    async def _fetch_torrent_file(
        self,
        http: httpx.AsyncClient,
        domain: str,
        torrent_path: str,
    ) -> bytes:
        """Fetch .torrent file from Anna's Archive, using cache if available."""
        if torrent_path in self._torrent_cache:
            logger.debug("Torrent cache hit: %s", torrent_path)
            return self._torrent_cache[torrent_path]

        url = f"{domain}/dyn/small_file/torrents/{torrent_path}"
        logger.info("Downloading torrent from %s", url)

        response = await http.get(url, timeout=15.0)
        response.raise_for_status()

        data = response.content
        logger.info("Torrent file: %d bytes", len(data))
        self._cache_torrent(torrent_path, data)
        return data

    async def download_file(
        self,
        http: httpx.AsyncClient,
        domain: str,
        torrent_info: TorrentPath,
        timeout: float = 120.0,
    ) -> TorrentDownloadResult:
        """Download a single file from a torrent using selective download.

        Args:
            http: HTTP client for fetching .torrent file
            domain: Anna's Archive mirror domain
            torrent_info: Torrent path info from metadata
            timeout: Max seconds to wait for download

        Returns:
            TorrentDownloadResult with file content

        Raises:
            TorrentError: If download fails
        """
        lt = self._lt
        start = time.monotonic()

        # Fetch the .torrent file
        try:
            torrent_data = await self._fetch_torrent_file(
                http, domain, torrent_info.torrent_path
            )
        except Exception as exc:
            raise TorrentError(f"Failed to fetch .torrent: {exc}") from exc

        # Parse torrent
        try:
            info = lt.torrent_info(lt.bdecode(torrent_data))
        except Exception as exc:
            raise TorrentError(f"Failed to parse .torrent: {exc}") from exc

        # Find target file index
        target_file, target_idx = self._find_target_file(info, torrent_info)
        if target_idx is None:
            raise TorrentError(
                f"File not found in torrent: {torrent_info.file_level1}"
            )

        # Check download overhead — skip if piece size makes it absurd
        piece_len = info.piece_length()
        file_offset = info.files().file_offset(target_idx)
        file_size = info.files().file_size(target_idx)
        first_piece = file_offset // piece_len
        last_piece = (file_offset + file_size - 1) // piece_len
        bytes_to_download = (last_piece - first_piece + 1) * piece_len
        overhead_ratio = bytes_to_download / max(file_size, 1)

        if overhead_ratio > 100:
            raise TorrentError(
                f"Piece size too large ({piece_len // 1024 // 1024}MB) — "
                f"would download {bytes_to_download // 1024 // 1024}MB for "
                f"{file_size // 1024}KB file ({overhead_ratio:.0f}x overhead)"
            )

        logger.info(
            "Found target: %s (index=%d, size=%d bytes, dl_overhead=%.0fx)",
            target_file.path,
            target_idx,
            target_file.size,
            overhead_ratio,
        )

        # Run the blocking download in a thread
        result = await asyncio.to_thread(
            self._download_selective,
            info,
            target_idx,
            target_file,
            timeout,
        )

        result.duration_s = time.monotonic() - start
        return result

    def _find_target_file(  # noqa: ANN202
        self,
        info,  # noqa: ANN001 - lt.torrent_info
        torrent_info: TorrentPath,
    ):
        """Find the target file in the torrent by matching file_level1."""
        fs = info.files()
        target_name = torrent_info.file_level1.lower()

        for i in range(fs.num_files()):
            file_path = fs.file_path(i)
            # Match by filename — file_level1 is usually the MD5 hash or MD5.ext
            parts = Path(file_path).parts
            for part in parts:
                if part.lower() == target_name or part.lower().startswith(
                    target_name + "."
                ):
                    return fs.at(i), i

        return None, None

    def _download_selective(
        self,
        info,  # noqa: ANN001 - lt.torrent_info
        target_idx: int,
        target_file,  # noqa: ANN001 - lt.file_entry
        timeout: float,
    ) -> TorrentDownloadResult:
        """Blocking selective download — runs in a thread."""
        import shutil

        tmpdir = tempfile.mkdtemp()
        try:
            return self._do_download(info, target_idx, target_file, timeout, tmpdir)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _do_download(self, info, target_idx, target_file, timeout, tmpdir):  # noqa: ANN001, ANN202
        """Run the actual download. Separated so tmpdir cleanup is robust."""
        handle = self._session.add_torrent({"ti": info, "save_path": tmpdir})

        # Add trackers for faster peer discovery
        for tracker in DEFAULT_TRACKERS:
            handle.add_tracker({"url": tracker, "tier": 0})

        # Set all files to not download, then enable only target
        num_files = info.files().num_files()
        priorities = [0] * num_files
        priorities[target_idx] = 7  # highest priority
        handle.prioritize_files(priorities)

        logger.info(
            "Downloading only: %s (%d bytes)",
            info.files().file_path(target_idx),
            target_file.size,
        )

        # Wait for download with progress logging
        start = time.monotonic()
        last_log = start
        peers = seeds = 0

        try:
            while (time.monotonic() - start) < timeout:
                status = handle.status()
                peers = status.num_peers
                seeds = status.num_seeds

                now = time.monotonic()
                if now - last_log >= 5.0:
                    logger.info(
                        "  [%ds] %s peers:%d seeds:%d dl:%dkB/s progress:%.1f%%",
                        int(now - start),
                        str(status.state),
                        peers,
                        seeds,
                        status.download_rate // 1024,
                        status.progress * 100,
                    )
                    last_log = now

                # Check if our file is complete
                if self._is_file_complete(handle, info, target_idx):
                    break

                time.sleep(0.5)
            else:
                raise TorrentError(
                    f"Torrent download timed out after {timeout}s "
                    f"(progress: {handle.status().progress * 100:.1f}%)"
                )

            # Read the downloaded file
            file_path = Path(tmpdir) / info.files().file_path(target_idx)
            if not file_path.exists():
                raise TorrentError(f"Downloaded file not found: {file_path}")

            content = file_path.read_bytes()
            logger.info(
                "Torrent download complete: %d bytes in %.1fs",
                len(content),
                time.monotonic() - start,
            )

            return TorrentDownloadResult(
                content=content,
                file_path=info.files().file_path(target_idx),
                duration_s=0,  # set by caller
                peers=peers,
                seeds=seeds,
            )

        finally:
            self._session.remove_torrent(handle)

    def _is_file_complete(
        self,
        handle,  # noqa: ANN001 - lt.torrent_handle
        info,  # noqa: ANN001 - lt.torrent_info
        file_idx: int,
    ) -> bool:
        """Check if a specific file in the torrent is fully downloaded."""
        fs = info.files()
        file_offset = fs.file_offset(file_idx)
        file_size = fs.file_size(file_idx)
        piece_length = info.piece_length()

        first_piece = file_offset // piece_length
        last_piece = (file_offset + file_size - 1) // piece_length

        status = handle.status()
        pieces = status.pieces

        for piece in range(first_piece, last_piece + 1):
            if not pieces[piece]:
                return False
        return True

    def shutdown(self) -> None:
        """Gracefully shut down the libtorrent session."""
        logger.info("Shutting down libtorrent session")
        # Pause all torrents and let alerts flush
        self._session.pause()
        self._torrent_cache.clear()


class TorrentError(Exception):
    """Torrent download failed."""
