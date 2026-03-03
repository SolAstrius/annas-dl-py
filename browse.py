"""TUI library browser — connects to annas-mcp REST API."""
import json
import os
import subprocess
import threading
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import urlencode, quote

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Static, Tree, Input, DataTable, RichLog
from textual.widgets.tree import TreeNode

MCP_BASE = os.environ.get("MCP_BASE", "https://mcp.i.ashbornlabs.com")
API_BASE = f"{MCP_BASE}/api"
API_TOKEN = os.environ.get("MCP_TOKEN", "i-solemnly-swear-i-am-up-to-no-good")
BOOKS_DIR = Path.home() / ".local" / "share" / "urn-books"


def api_get(path: str, params: dict | None = None) -> dict:
    """GET request to the API with auth."""
    url = f"{API_BASE}{path}"
    if params:
        filtered = {k: v for k, v in params.items() if v is not None}
        if filtered:
            url += "?" + urlencode(filtered)
    req = Request(url, headers={"Authorization": f"Bearer {API_TOKEN}"})
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def fetch_categories(path: str | None = None, depth: int = 1) -> str:
    """Fetch category tree as text."""
    params = {"depth": str(depth)}
    if path:
        params["path"] = path
    data = api_get("/categories", params)
    return data.get("text", "")


def fetch_library(search: str | None = None, source: str | None = None,
                  category: str | None = None, language: str | None = None,
                  sort: str | None = None, limit: int = 200, offset: int = 0) -> dict:
    """Fetch library listing as JSON."""
    return api_get("/library", {
        "search": search,
        "source": source,
        "category": category,
        "language": language,
        "sort": sort,
        "limit": str(limit),
        "offset": str(offset),
        "columns": "description,categories",
    })


def fetch_stats() -> dict:
    """Fetch library stats."""
    return api_get("/stats")


def fetch_book_info(urn: str) -> str:
    """Fetch book info + TOC."""
    # /api/books/{urn} handler prepends "urn:", so strip it from our URN
    bare = urn.removeprefix("urn:")
    # Use raw URL construction to handle colons in URN properly
    url = f"{API_BASE}/books/{quote(bare, safe='/:')}"
    req = Request(url, headers={"Authorization": f"Bearer {API_TOKEN}"})
    with urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data.get("text", "")


def parse_library_text(text: str) -> list[dict]:
    """Parse the library tool's text output into structured book dicts.

    Format:
        1. Title — Author
           id: urn:anna:HASH · pdf · 2003 · eng,grc · 11MB · indexed
           categories: Classical World > Greek Texts
    """
    books = []
    current = {}
    for line in text.split("\n"):
        line = line.rstrip()
        if not line:
            continue
        stripped = line.strip()
        # Lines starting with a number + period are book entries
        if line and line[0].isdigit() and ". " in line[:6]:
            if current and "urn" in current:
                books.append(current)
            after_num = line.split(". ", 1)[1] if ". " in line else line
            if " — " in after_num:
                title, author = after_num.rsplit(" — ", 1)
            else:
                title, author = after_num, ""
            current = {"title": title.strip(), "author": author.strip()}
        elif stripped.startswith("id: "):
            # "id: urn:anna:HASH · pdf · 2003 · eng,grc · 11MB · indexed"
            parts = stripped.removeprefix("id: ").split(" · ")
            current["urn"] = parts[0] if parts else ""
            # Remaining parts are variable: format, year, lang, size, etc.
            FORMATS = {"pdf", "epub", "djvu", "txt", "json", "xml", "mobi", "cbz", "cbr"}
            for part in parts[1:]:
                p = part.strip()
                if p in ("indexed", "not_indexed"):
                    current["indexed"] = p == "indexed"
                elif p.startswith("text:"):
                    current["text_quality"] = p.removeprefix("text:")
                elif p.startswith("★"):
                    current["rating"] = p
                elif p[-2:] in ("KB", "MB", "GB", "TB") or (p[-1:] == "B" and p[:-1].isdigit()):
                    current["size"] = p
                elif p in FORMATS:
                    current["format"] = p
                elif p.isdigit() and len(p) == 4:
                    current["year"] = p
                elif len(p) <= 10 and ("," in p or len(p) <= 3) and p not in FORMATS:
                    current["lang"] = p
        elif stripped.startswith("categories: "):
            current["category"] = stripped.removeprefix("categories: ")
        elif stripped.startswith("desc: "):
            current["desc"] = stripped.removeprefix("desc: ")
    if current and "urn" in current:
        books.append(current)
    return books


def build_tree_from_categories(text: str) -> dict:
    """Parse category text output into a nested dict for the tree widget."""
    tree = {}
    for line in text.split("\n"):
        if not line.strip():
            continue
        # Count indent level (2 spaces per level)
        stripped = line.lstrip()
        indent = (len(line) - len(stripped)) // 2
        # Parse name and count: "Category Name (42)" or "Category Name [3 subcategories ...]"
        name = stripped.split(" (")[0].split(" [")[0].strip()
        if not name or name.startswith("..."):
            continue
        # We just need the flat list of paths for the tree
        # Store as path -> name mapping
        tree.setdefault(indent, []).append(name)
    return tree


class BookBrowser(App):
    CSS = """
    #main {
        height: 1fr;
    }
    #tree-pane {
        width: 45;
        min-width: 30;
        border-right: solid $primary-background;
    }
    #tree {
        height: 1fr;
    }
    #right-pane {
        width: 1fr;
    }
    #book-table {
        height: 1fr;
    }
    #detail {
        height: 12;
        border-top: solid $primary-background;
        padding: 0 1;
        overflow-y: auto;
    }
    #search {
        dock: top;
    }
    #stats {
        dock: bottom;
        height: 1;
        padding: 0 1;
        background: $primary-background;
        color: $text;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("/", "focus_search", "Search"),
        Binding("escape", "clear_search", "Clear"),
        Binding("o", "open_book", "Open"),
        Binding("i", "show_info", "Info"),
        Binding("c", "copy_detail", "Copy"),
        Binding("w", "show_uncategorized", "Uncategorized"),
    ]

    def __init__(self):
        super().__init__()
        self.current_books: list[dict] = []
        self.stats: dict | None = None
        self.category_paths: dict[str, str] = {}  # tree node label -> full path

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Input(placeholder="Search books... (/ to focus, Esc to clear)", id="search")
        with Horizontal(id="main"):
            with Vertical(id="tree-pane"):
                tree = Tree("Library", id="tree")
                tree.root.expand()
                yield tree
            with Vertical(id="right-pane"):
                table = DataTable(id="book-table", cursor_type="row")
                yield table
                yield RichLog(id="detail", wrap=True, markup=True, auto_scroll=False)
        yield Static(" Loading...", id="stats")
        yield Footer()

    def on_mount(self):
        table = self.query_one("#book-table", DataTable)
        table.add_columns("Lang", "Title", "Author", "Year", "Fmt")
        table.fixed_columns = 1
        # Load data in background
        threading.Thread(target=self._load_initial_data, daemon=True).start()

    def _load_initial_data(self):
        try:
            self.stats = fetch_stats()
            cat_text = fetch_categories(depth=2)
            self.call_from_thread(self._populate_tree, cat_text)
            summary = self.stats.get("_summary", "")
            self.call_from_thread(self._set_stats, summary)
        except Exception as e:
            self.call_from_thread(self._set_stats, f"Error: {e}")

    def _set_stats(self, text: str):
        stats = self.query_one("#stats", Static)
        # Take just the first line for the status bar
        first_line = text.split("\n")[0] if text else "Library"
        stats.update(f" {first_line}")

    def _populate_tree(self, cat_text: str):
        tree = self.query_one("#tree", Tree)
        # Parse the hierarchical text output into tree nodes
        # The format is indented text like:
        #   Category (42)
        #     Subcategory (10)
        import re
        stack: list[tuple[int, TreeNode, str]] = [(-1, tree.root, "")]
        for line in cat_text.split("\n"):
            if not line.strip():
                continue
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if not stripped or stripped.startswith("...") or "top-level" in stripped:
                continue

            # Extract name and optional count: "Category Name (42)" or "Category Name [3 subcategories ...]"
            m = re.match(r'^(.+?)(?:\s+\((\d+)\))?(?:\s+\[.+\])?$', stripped)
            if not m:
                continue
            name = m.group(1).strip()
            count = m.group(2)
            if not name:
                continue

            # Pop stack back to parent level
            while stack and stack[-1][0] >= indent:
                stack.pop()

            parent_path = stack[-1][2] if stack else ""
            full_path = f"{parent_path} > {name}" if parent_path else name
            parent_node = stack[-1][1] if stack else tree.root

            label = f"{name} ({count})" if count else name
            child = parent_node.add(label, data={"path": full_path})
            self.category_paths[name] = full_path
            stack.append((indent, child, full_path))

    def on_tree_node_selected(self, event: Tree.NodeSelected):
        node_data = event.node.data
        if not isinstance(node_data, dict) or "path" not in node_data:
            return
        category_path = node_data["path"]
        detail = self.query_one("#detail", RichLog)
        detail.clear()
        detail.write(f"[dim]Loading {category_path}...[/dim]")
        threading.Thread(
            target=self._load_category, args=(category_path,), daemon=True
        ).start()

    def _load_category(self, path: str):
        try:
            data = fetch_library(category=path, limit=200)
            text = data.get("text", data.get("_summary", ""))
            books = parse_library_text(text)
            self.call_from_thread(self._show_books, books, path)
        except Exception as e:
            self.call_from_thread(self._show_error, str(e))

    def _show_books(self, books: list[dict], context: str = ""):
        self.current_books = books
        table = self.query_one("#book-table", DataTable)
        table.clear()
        for b in books:
            lang = b.get("lang", "?")
            title = b.get("title", "?")[:80]
            author = b.get("author", "")[:40]
            year = b.get("year", "")
            fmt = b.get("format", "")
            table.add_row(lang, title, author, year, fmt, key=b.get("urn", ""))

        detail = self.query_one("#detail", RichLog)
        detail.clear()
        detail.write(f"[bold]{len(books)}[/bold] books" + (f" in {context}" if context else ""))

    def _show_error(self, msg: str):
        detail = self.query_one("#detail", RichLog)
        detail.clear()
        detail.write(f"[bold red]Error:[/bold red] {msg}")

    def on_data_table_row_selected(self, event: DataTable.RowSelected):
        urn = str(event.row_key.value)
        book = next((b for b in self.current_books if b.get("urn") == urn), None)
        if not book:
            return

        detail = self.query_one("#detail", RichLog)
        detail.clear()
        detail.write(f"[bold]{book.get('title', '?')}[/bold]")
        if book.get("author"):
            detail.write(f"[dim]by[/dim] {book['author']}")
        parts = []
        if book.get("year"):
            parts.append(f"Year: {book['year']}")
        if book.get("lang"):
            parts.append(f"Lang: {book['lang']}")
        if book.get("format"):
            parts.append(f"Format: {book['format']}")
        if book.get("size"):
            parts.append(f"Size: {book['size']}")
        parts.append(f"URN: {urn}")
        detail.write(" · ".join(parts))
        if book.get("category"):
            detail.write(f"[green]Category:[/green] {book['category']}")

    def on_input_changed(self, event: Input.Changed):
        query = event.value.strip()
        if len(query) < 2:
            return
        # Debounce: only search on 3+ chars
        if len(query) < 3:
            return
        threading.Thread(
            target=self._search, args=(query,), daemon=True
        ).start()

    def _search(self, query: str):
        try:
            data = fetch_library(search=query, limit=50)
            text = data.get("text", data.get("_summary", ""))
            books = parse_library_text(text)
            self.call_from_thread(self._show_books, books, f"search: {query}")
        except Exception as e:
            self.call_from_thread(self._show_error, str(e))

    def _get_cursor_urn(self) -> str | None:
        table = self.query_one("#book-table", DataTable)
        if table.row_count == 0:
            return None
        try:
            row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
            return str(row_key.value)
        except Exception:
            return None

    def action_open_book(self):
        urn = self._get_cursor_urn()
        if not urn:
            return
        book = next((b for b in self.current_books if b.get("urn") == urn), None)
        if not book:
            return

        ext = book.get("format", "pdf")
        title = book.get("title", "")[:60]
        detail = self.query_one("#detail", RichLog)
        detail.clear()
        detail.write(f"[bold]{title}[/bold]")

        fname = urn.replace(":", "-") + f".{ext}"
        outpath = BOOKS_DIR / fname
        if outpath.exists():
            detail.write(f"[green]Already downloaded:[/green] {outpath}")
            subprocess.Popen(["xdg-open", str(outpath)])
            return

        url = f"{MCP_BASE}/urn/{urn}"
        detail.write(f"[dim]Downloading to {outpath.name}...[/dim]")

        def _download():
            try:
                BOOKS_DIR.mkdir(parents=True, exist_ok=True)
                tmp = outpath.with_suffix(outpath.suffix + ".part")
                req = Request(url, headers={"Authorization": f"Bearer {API_TOKEN}"})
                with urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
                    while chunk := resp.read(65536):
                        out.write(chunk)
                tmp.rename(outpath)
                subprocess.Popen(["xdg-open", str(outpath)])
                self.call_from_thread(self._open_done, str(outpath))
            except Exception as e:
                self.call_from_thread(self._show_error, str(e))

        threading.Thread(target=_download, daemon=True).start()

    def _open_done(self, path: str):
        detail = self.query_one("#detail", RichLog)
        detail.write(f"[green]Opened:[/green] {path}")

    def action_show_info(self):
        urn = self._get_cursor_urn()
        if not urn:
            return
        detail = self.query_one("#detail", RichLog)
        detail.clear()
        detail.write(f"[dim]Loading info for {urn}...[/dim]")

        def _load():
            try:
                text = fetch_book_info(urn)
                self.call_from_thread(self._show_info_text, text)
            except Exception as e:
                self.call_from_thread(self._show_error, str(e))

        threading.Thread(target=_load, daemon=True).start()

    def _show_info_text(self, text: str):
        detail = self.query_one("#detail", RichLog)
        detail.clear()
        for line in text.split("\n")[:30]:
            detail.write(line)

    def action_copy_detail(self):
        urn = self._get_cursor_urn()
        if not urn:
            return
        book = next((b for b in self.current_books if b.get("urn") == urn), None)
        if not book:
            return

        link = f"{MCP_BASE}/urn/{urn}?inline=true"
        lines = [f"**{book.get('title', '?')}**"]
        if book.get("author"):
            lines.append(f"by {book['author']}")
        parts = []
        if book.get("year"):
            parts.append(str(book["year"]))
        if book.get("lang"):
            parts.append(book["lang"])
        if book.get("format"):
            parts.append(book["format"].upper())
        if book.get("size"):
            parts.append(book["size"])
        lines.append(" · ".join(parts))
        lines.append("")
        lines.append(f"[Download]({link})")

        md = "\n".join(lines)
        try:
            subprocess.run(["wl-copy"], input=md.encode(), timeout=5)
            detail = self.query_one("#detail", RichLog)
            detail.write("[green]Copied to clipboard[/green]")
        except Exception as e:
            detail = self.query_one("#detail", RichLog)
            detail.write(f"[red]Copy failed:[/red] {e}")

    def action_show_uncategorized(self):
        detail = self.query_one("#detail", RichLog)
        detail.clear()
        detail.write("[dim]Loading uncategorized books...[/dim]")

        def _load():
            try:
                data = api_get("/library", {
                    "uncategorized": "true",
                    "limit": "200",
                    "columns": "description,categories",
                })
                text = data.get("text", data.get("_summary", ""))
                books = parse_library_text(text)
                self.call_from_thread(self._show_books, books, "uncategorized")
            except Exception as e:
                self.call_from_thread(self._show_error, str(e))

        threading.Thread(target=_load, daemon=True).start()

    def action_focus_search(self):
        self.query_one("#search", Input).focus()

    def action_clear_search(self):
        inp = self.query_one("#search", Input)
        inp.value = ""
        self.query_one("#book-table", DataTable).clear()
        self.query_one("#detail", RichLog).clear()
        self.query_one("#tree", Tree).focus()


if __name__ == "__main__":
    BookBrowser().run()
