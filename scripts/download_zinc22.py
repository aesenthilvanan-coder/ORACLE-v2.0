#!/usr/bin/env python3
"""
Download all ZINC22 in-stock SMILES via parallel tranch crawler.
Writes: data/raw/molecules/zinc20_instock.smi.gz
"""
import gzip, os, sys, time, threading, queue, logging
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("zinc22")

BASE = "https://files.docking.org/zinc22"
OUT  = Path("data/raw/molecules/zinc20_instock.smi.gz")
OUT.parent.mkdir(parents=True, exist_ok=True)

ZINC_SETS = [
    "zinc-22a","zinc-22b","zinc-22c","zinc-22d","zinc-22e","zinc-22f",
    "zinc-22g","zinc-22h","zinc-22i","zinc-22j","zinc-22k","zinc-22l",
    "zinc-22m","zinc-22n","zinc-22o","zinc-22p","zinc-22q",
]

class LinkParser(HTMLParser):
    def __init__(self): super().__init__(); self.links = []
    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for k, v in attrs:
                if (k == "href" and v not in ("/", "../", "")
                        and "?C=" not in v
                        and not v.startswith("/")):   # exclude absolute paths
                    self.links.append(v)

def fetch_text(url, retries=3):
    for i in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=30) as r:
                return r.read()
        except Exception as e:
            if i == retries - 1: raise
            time.sleep(2 ** i)

def list_dir(url):
    try:
        html = fetch_text(url).decode("utf-8", errors="ignore")
        p = LinkParser(); p.feed(html)
        return p.links
    except Exception as e:
        log.warning(f"list_dir failed: {url}: {e}")
        return []

def fetch_smi_gz(url):
    """Fetch a .g.smi.gz file and return decompressed SMILES lines."""
    try:
        data = fetch_text(url)
        if len(data) < 20:
            return []
        lines = gzip.decompress(data).decode("utf-8", errors="ignore").strip().split("\n")
        # Each line: SMILES <tab> ZINC_ID  — extract SMILES only
        smiles = []
        for ln in lines:
            parts = ln.strip().split()
            if parts and len(parts[0]) > 3:
                smiles.append(parts[0])
        return smiles
    except Exception:
        return []

def collect_smi_urls(zset):
    """Crawl one zinc-22x directory and collect all *.smi.gz URLs."""
    urls = []
    base = f"{BASE}/{zset}"
    h_dirs = [l for l in list_dir(base + "/") if l.endswith("/") and not l.startswith("..")]
    for hd in h_dirs:
        h_url = f"{base}/{hd}"
        sub_items = list_dir(h_url)
        # Direct .smi.gz files at H-level
        for item in sub_items:
            if item.endswith(".smi.gz"):
                urls.append(f"{h_url}{item}")
        # Sub-subdirectories (H04M000/ etc.)
        subdirs = [s for s in sub_items if s.endswith("/") and not s.startswith("..")]
        for sd in subdirs:
            sd_url = f"{h_url}{sd}"
            for item in list_dir(sd_url):
                if item.endswith(".smi.gz"):
                    urls.append(f"{sd_url}{item}")
    return urls

def main():
    log.info("ZINC22 parallel tranch downloader")
    log.info(f"Output: {OUT}")

    # Phase 1: collect all .smi.gz URLs
    log.info("Phase 1: crawling directory structure...")
    all_urls = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(collect_smi_urls, zs): zs for zs in ZINC_SETS}
        for fut in as_completed(futures):
            zs = futures[fut]
            try:
                urls = fut.result()
                all_urls.extend(urls)
                log.info(f"  {zs}: {len(urls)} tranch files")
            except Exception as e:
                log.warning(f"  {zs}: crawl failed: {e}")

    log.info(f"Total tranch files found: {len(all_urls):,}")

    if not all_urls:
        log.error("No tranch URLs found — aborting")
        return

    # Phase 2: download all tranch files in parallel, write to gzip output
    log.info("Phase 2: downloading SMILES...")
    n_mols = 0
    n_done = 0

    with gzip.open(OUT, "wt", compresslevel=6) as fout:
        with ThreadPoolExecutor(max_workers=32) as ex:
            futures = {ex.submit(fetch_smi_gz, url): url for url in all_urls}
            for fut in as_completed(futures):
                n_done += 1
                smiles_list = fut.result()
                for smi in smiles_list:
                    fout.write(smi + "\n")
                    n_mols += 1
                if n_done % 500 == 0:
                    log.info(f"  {n_done:,}/{len(all_urls):,} files | {n_mols:,} molecules")

    log.info(f"Done: {n_mols:,} ZINC22 molecules → {OUT}")
    log.info(f"File size: {OUT.stat().st_size / 1e6:.1f} MB")

if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    main()
