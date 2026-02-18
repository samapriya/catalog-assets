#!/usr/bin/env python3
"""
build_prism_catalogs.py

Scrape the PRISM 800m Apache directory listings and write two slim catalogs:
    assets/prism_catalog_monthly.json
    assets/prism_catalog_daily.json

Each record contains only:
    { "filename": "...", "url": "...", "size_bytes": 12345678 }

Usage:
    python build_prism_catalogs.py
    python build_prism_catalogs.py --workers 16
    python build_prism_catalogs.py --base-url http://my-mirror/800m/

Requirements:
    pip install requests beautifulsoup4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRISM_BASE_URL = "https://data.prism.oregonstate.edu/time_series/us/an/800m"
OUTPUT_DIR     = "assets"
REQUEST_TIMEOUT = 30
MAX_RETRIES     = 3
RETRY_BACKOFF   = 1.5
DEFAULT_WORKERS = 8

ALL_VARIABLES = [
    "ppt", "solslope", "soltotal", "tdmean",
    "tmax", "tmean", "tmin", "vpdmax", "vpdmin",
]

FREQUENCIES = ("monthly", "daily")

_NAV_HREFS = frozenset([
    "../", "/",
    "?C=N;O=D", "?C=N;O=A",
    "?C=M;O=D", "?C=M;O=A",
    "?C=S;O=D", "?C=S;O=A",
    "?C=D;O=D", "?C=D;O=A",
])

# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = (
    "PRISM-CatalogGen/1.0 (climate research; oregonstate.edu data)"
)


def _get(url: str) -> requests.Response | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = _SESSION.get(url, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                print(f"  [WARN] {url}: {exc}", file=sys.stderr)
                return None
            time.sleep(RETRY_BACKOFF * (2 ** (attempt - 1)))
    return None


# ---------------------------------------------------------------------------
# Apache size parser
# ---------------------------------------------------------------------------

_UNIT = {"k": 1_024, "m": 1_048_576, "g": 1_073_741_824}


def _parse_size(raw: str) -> int | None:
    s = raw.strip()
    if not s or s == "-":
        return None
    suffix = s[-1].lower()
    if suffix in _UNIT:
        try:
            return round(float(s[:-1]) * _UNIT[suffix])
        except ValueError:
            return None
    try:
        return int(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Directory scraper  →  {filename: size_bytes}
# ---------------------------------------------------------------------------

def scrape_dir(url: str) -> dict[str, int | None]:
    resp = _get(url)
    if resp is None:
        return {}

    soup   = BeautifulSoup(resp.text, "html.parser")
    result: dict[str, int | None] = {}

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href in _NAV_HREFS or not href.lower().endswith(".zip"):
            continue

        filename   = os.path.basename(href.rstrip("/"))
        size_bytes: int | None = None

        row = a.find_parent("tr")
        if row:
            cells = row.find_all("td")
            if len(cells) >= 4:
                size_bytes = _parse_size(cells[3].get_text())
            else:
                for cell in cells:
                    txt = cell.get_text().strip()
                    if txt and txt != "-":
                        candidate = _parse_size(txt)
                        if candidate:
                            size_bytes = candidate
                            break

        result[filename] = size_bytes

    return result


# ---------------------------------------------------------------------------
# Year-directory listing  →  list of year strings
# ---------------------------------------------------------------------------

def list_years(url: str) -> list[str]:
    resp = _get(url)
    if resp is None:
        return []

    soup  = BeautifulSoup(resp.text, "html.parser")
    years = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip().rstrip("/")
        if href.isdigit() and len(href) == 4:
            years.append(href)
    return sorted(years)


# ---------------------------------------------------------------------------
# Build one frequency catalog
# ---------------------------------------------------------------------------

def build_records(
    base_url: str,
    frequency: str,
    workers: int,
) -> list[dict]:
    """Return a list of {filename, url, size_bytes} dicts for one frequency."""

    tasks: list[tuple[str, str]] = []   # (year_dir_url, file_base_url)

    for variable in ALL_VARIABLES:
        freq_url = f"{base_url}/{variable}/{frequency}/"
        years    = list_years(freq_url)
        if not years:
            print(f"  [skip] no years found: {freq_url}", file=sys.stderr)
            continue
        print(f"  {variable}/{frequency}: {len(years)} year(s)")
        for year in years:
            year_url = f"{freq_url}{year}/"
            tasks.append((year_url, year_url))

    records: list[dict] = []

    def fetch_year(year_url: str) -> list[dict]:
        files = scrape_dir(year_url)
        out   = []
        for fname, sz in files.items():
            out.append({
                "filename":   fname,
                "url":        urljoin(year_url, fname),
                "size_bytes": sz,
            })
        return out

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_year, url): url for url, _ in tasks}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                records.extend(fut.result())
            except Exception as exc:
                print(f"  [ERR] {futs[fut]}: {exc}", file=sys.stderr)
            if done % 50 == 0 or done == len(tasks):
                print(f"    {frequency}: {done}/{len(tasks)} dirs done …")

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default=PRISM_BASE_URL, dest="base_url")
    p.add_argument("--workers",  type=int, default=DEFAULT_WORKERS)
    p.add_argument("--output-dir", default=OUTPUT_DIR, dest="output_dir")
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for freq in FREQUENCIES:
        print(f"\n{'='*60}")
        print(f"  Building {freq} catalog …")
        print(f"{'='*60}")

        t0      = time.monotonic()
        records = build_records(args.base_url, freq, args.workers)
        elapsed = time.monotonic() - t0

        out_path = Path(args.output_dir) / f"prism_catalog_{freq}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2, ensure_ascii=False)

        kb = out_path.stat().st_size / 1024
        print(f"\n  → {out_path}  ({len(records):,} records, {kb:,.0f} KB, {elapsed:.0f}s)")

    print("\nAll done.")


if __name__ == "__main__":
    main()
