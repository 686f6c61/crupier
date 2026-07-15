from __future__ import annotations

from collections import Counter
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
import tomllib
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
HTML_PATH = SITE / "index.html"
SITE_URL = "https://crupier.686f6c61.dev/"
SOCIAL_IMAGE_URL = f"{SITE_URL}og-crupier.png"
REQUIRED_FILES = (
    ROOT / ".dockerignore",
    ROOT / "Dockerfile.site",
    HTML_PATH,
    SITE / "styles.css",
    SITE / "app.js",
    SITE / "favicon.svg",
    SITE / "og-crupier.png",
    SITE / "robots.txt",
    SITE / "sitemap.xml",
    SITE / "social-card.html",
    SITE / "nginx.conf",
    SITE / ".nojekyll",
)
PUBLIC_MODEL_PROVIDERS = {"anthropic", "google", "ollama", "openai"}


class SiteParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.references: list[tuple[str, str]] = []
        self.tabs: list[dict[str, str]] = []
        self.scripts: list[str] = []
        self.json_ld: list[str] = []
        self.canonicals: list[str] = []
        self.meta_names: dict[str, str] = {}
        self.meta_properties: dict[str, str] = {}
        self._json_ld_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name: value or "" for name, value in attrs}
        if values.get("id"):
            self.ids.append(values["id"])
        for attribute in ("href", "src"):
            if values.get(attribute):
                self.references.append((attribute, values[attribute]))
        if values.get("role") == "tab":
            self.tabs.append(values)
        if tag == "script" and values.get("type") == "application/ld+json":
            self._json_ld_parts = []
        elif tag == "script":
            self.scripts.append(values.get("src", ""))
        if tag == "link" and values.get("rel") == "canonical":
            self.canonicals.append(values.get("href", ""))
        if tag == "meta" and values.get("name"):
            self.meta_names[values["name"]] = values.get("content", "")
        if tag == "meta" and values.get("property"):
            self.meta_properties[values["property"]] = values.get("content", "")

    def handle_data(self, data: str) -> None:
        if self._json_ld_parts is not None:
            self._json_ld_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._json_ld_parts is not None:
            self.json_ld.append("".join(self._json_ld_parts))
            self._json_ld_parts = None


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def png_dimensions(path: Path) -> tuple[int, int] | None:
    header = path.read_bytes()[:24]
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        return None
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")


def validate() -> list[str]:
    errors: list[str] = []
    for path in REQUIRED_FILES:
        if not path.is_file():
            fail(errors, f"missing required file: {path.relative_to(ROOT)}")
    if errors:
        return errors

    html = HTML_PATH.read_text(encoding="utf-8")
    css = (SITE / "styles.css").read_text(encoding="utf-8")
    js = (SITE / "app.js").read_text(encoding="utf-8")
    parser = SiteParser()
    parser.feed(html)

    duplicate_ids = sorted(name for name, count in Counter(parser.ids).items() if count > 1)
    if duplicate_ids:
        fail(errors, f"duplicate HTML ids: {', '.join(duplicate_ids)}")
    known_ids = set(parser.ids)

    for attribute, reference in parser.references:
        parsed = urlsplit(reference)
        if parsed.scheme or reference.startswith("//"):
            continue
        if parsed.fragment and not parsed.path and parsed.fragment not in known_ids:
            fail(errors, f"broken fragment: {reference}")
        if not parsed.path:
            continue
        target = (SITE / unquote(parsed.path.lstrip("/"))).resolve()
        if SITE.resolve() not in target.parents and target != SITE.resolve():
            fail(errors, f"local {attribute} escapes site/: {reference}")
        elif not target.is_file():
            fail(errors, f"missing local asset: {reference}")

    if parser.canonicals != [SITE_URL]:
        fail(errors, f"canonical URL must be exactly {SITE_URL}")
    required_named_meta = {
        "description",
        "robots",
        "twitter:card",
        "twitter:title",
        "twitter:description",
        "twitter:image",
        "twitter:image:alt",
    }
    required_property_meta = {
        "og:type",
        "og:locale",
        "og:site_name",
        "og:title",
        "og:description",
        "og:image",
        "og:image:secure_url",
        "og:image:type",
        "og:image:width",
        "og:image:height",
        "og:image:alt",
        "og:url",
    }
    for name in sorted(required_named_meta):
        if not parser.meta_names.get(name):
            fail(errors, f"missing SEO meta name: {name}")
    for prop in sorted(required_property_meta):
        if not parser.meta_properties.get(prop):
            fail(errors, f"missing Open Graph property: {prop}")
    if parser.meta_names.get("twitter:card") != "summary_large_image":
        fail(errors, "Twitter card must use summary_large_image")
    if parser.meta_names.get("twitter:image") != SOCIAL_IMAGE_URL:
        fail(errors, f"Twitter card must reference {SOCIAL_IMAGE_URL}")
    if parser.meta_properties.get("og:image") != SOCIAL_IMAGE_URL:
        fail(errors, f"Open Graph must reference {SOCIAL_IMAGE_URL}")
    if parser.meta_properties.get("og:image:secure_url") != SOCIAL_IMAGE_URL:
        fail(errors, f"Open Graph secure image must reference {SOCIAL_IMAGE_URL}")
    if parser.meta_properties.get("og:url") != SITE_URL:
        fail(errors, f"Open Graph URL must be exactly {SITE_URL}")
    if parser.meta_properties.get("og:image:width") != "1200" or parser.meta_properties.get("og:image:height") != "630":
        fail(errors, "Open Graph image metadata must be 1200x630")
    dimensions = png_dimensions(SITE / "og-crupier.png")
    if dimensions != (1200, 630):
        fail(errors, f"social image must be a 1200x630 PNG, found {dimensions}")
    if len(parser.json_ld) != 1:
        fail(errors, "site must expose exactly one JSON-LD document")
    else:
        try:
            structured_data = json.loads(parser.json_ld[0])
        except json.JSONDecodeError as error:
            fail(errors, f"invalid JSON-LD: {error}")
        else:
            if structured_data.get("@type") != "SoftwareSourceCode":
                fail(errors, "JSON-LD must describe SoftwareSourceCode")
            if structured_data.get("codeRepository") != "https://github.com/686f6c61/crupier":
                fail(errors, "JSON-LD repository does not match the public project URL")
            if structured_data.get("url") != SITE_URL or structured_data.get("image") != SOCIAL_IMAGE_URL:
                fail(errors, "JSON-LD deployment URLs do not match the canonical site")
    robots = (SITE / "robots.txt").read_text(encoding="utf-8")
    if "User-agent: *" not in robots or "Allow: /" not in robots:
        fail(errors, "robots.txt must allow indexing")
    if f"Sitemap: {SITE_URL}sitemap.xml" not in robots:
        fail(errors, "robots.txt must expose the canonical sitemap")
    try:
        sitemap = ElementTree.parse(SITE / "sitemap.xml")
    except ElementTree.ParseError as error:
        fail(errors, f"invalid sitemap.xml: {error}")
    else:
        sitemap_locations = [node.text for node in sitemap.findall("{http://www.sitemaps.org/schemas/sitemap/0.9}url/{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
        if sitemap_locations != [SITE_URL]:
            fail(errors, f"sitemap must contain only the canonical site URL, found {sitemap_locations}")
    if not parser.tabs:
        fail(errors, "site must expose keyboard-addressable tabs")
    grouped_tabs: dict[str, list[dict[str, str]]] = {}
    for tab in parser.tabs:
        for required in ("id", "aria-controls", "aria-selected", "tabindex"):
            if required not in tab:
                fail(errors, f"tab is missing {required}: {tab}")
        control = tab.get("aria-controls", "")
        grouped_tabs.setdefault(control, []).append(tab)
        if control not in known_ids:
            fail(errors, f"tab controls missing panel: {control}")
    for control, tabs in grouped_tabs.items():
        selected = [tab for tab in tabs if tab.get("aria-selected") == "true"]
        if len(selected) != 1 or selected[0].get("tabindex") != "0":
            fail(errors, f"tab group {control} must have one selected, focusable tab")

    if not parser.scripts or any(not source or urlsplit(source).scheme for source in parser.scripts):
        fail(errors, "scripts must be local external files")

    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    if f"Crupier {version}" not in html:
        fail(errors, f"site does not expose package version {version}")
    for required_text in ("Una petición", "Patrones de ejecución completos", "Qué todavía no"):
        if required_text not in html:
            fail(errors, f"missing required product statement: {required_text}")

    combined = "\n".join((html, css, js))
    if re.search(r"(?:linear|radial|conic)-gradient\s*\(", css, flags=re.IGNORECASE):
        fail(errors, "CSS gradients are outside this site's visual contract")
    if "letter-spacing: 0" not in css:
        fail(errors, "site must keep letter spacing neutral")
    model_surface = js
    unknown_providers = sorted(
        set(re.findall(r"\b([a-z][a-z0-9_-]+):[a-z][a-z0-9_.-]+", model_surface))
        - PUBLIC_MODEL_PROVIDERS
    )
    if unknown_providers:
        fail(errors, f"non-public model provider prefixes found: {', '.join(unknown_providers)}")
    secret_patterns = (
        r"\bsk-(?:proj|ant)-[A-Za-z0-9_-]{20,}",
        r"\bAIza[A-Za-z0-9_-]{20,}",
        r"\b(?:OPENAI|ANTHROPIC|GEMINI|GOOGLE|OLLAMA)_API_KEY\s*=\s*[^\s<]+",
    )
    if any(re.search(pattern, combined) for pattern in secret_patterns):
        fail(errors, "credential-like material found in static site")

    dockerfile = (ROOT / "Dockerfile.site").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    nginx = (SITE / "nginx.conf").read_text(encoding="utf-8")
    if "FROM nginx:stable-alpine" not in dockerfile or "HEALTHCHECK" not in dockerfile:
        fail(errors, "site Dockerfile must use stable Nginx and define a health check")
    if dockerignore.splitlines()[:1] != ["**"] or "!site/**" not in dockerignore:
        fail(errors, "Docker context must deny by default and include only site assets")
    for marker in ("location = /healthz", "X-Content-Type-Options", "X-Frame-Options", "try_files $uri $uri/ =404"):
        if marker not in nginx:
            fail(errors, f"Nginx configuration is missing: {marker}")

    return errors


if __name__ == "__main__":
    problems = validate()
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        raise SystemExit(1)
    print("Static site validation passed.")
