#!/usr/bin/env python
"""
Materialize Tiny-ImageNet:
- If dataset is missing, download & extract it automatically.
- Then (optionally) rearrange val/images into class folders per val_annotations.txt.
- Emits a clear JSON report.

No third-party dependencies required.
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from zipfile import ZipFile
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


DEFAULT_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def has_expected_layout(root: Path) -> bool:
    return (root / "train").is_dir() and (root / "val").is_dir() and (root / "val" / "val_annotations.txt").is_file()


def diagnose_layout(root: Path):
    return {
        "train_exists": (root / "train").is_dir(),
        "val_exists": (root / "val").is_dir(),
        "ann_exists": (root / "val" / "val_annotations.txt").is_file(),
        "train_path": str(root / "train"),
        "val_path": str(root / "val"),
        "ann_path": str(root / "val" / "val_annotations.txt"),
    }


def read_val_annotations(ann_path: Path):
    mapping = {}
    with ann_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()  # fallback: any whitespace
            if len(parts) >= 2:
                fname, cls = parts[0], parts[1]
                mapping[fname] = cls
    return mapping


def fix_validation_split(val_dir: Path, dry_run: bool = False, verbose: bool = False):
    stats = {
        "files_listed": 0,
        "moved": 0,
        "missing_in_images": 0,
        "skipped_collision": 0,
        "classes_created": 0,
    }
    images_dir = val_dir / "images"
    ann = val_dir / "val_annotations.txt"
    if not images_dir.is_dir() or not ann.is_file():
        return stats

    mapping = read_val_annotations(ann)
    stats["files_listed"] = len(mapping)
    created = set()

    for fname, cls in mapping.items():
        src = images_dir / fname
        if not src.exists():
            stats["missing_in_images"] += 1
            if verbose:
                print(f"[fix-val] Missing: {src}")
            continue

        dst_dir = val_dir / cls / "images"
        if cls not in created and not dst_dir.exists():
            if not dry_run:
                dst_dir.mkdir(parents=True, exist_ok=True)
            created.add(cls)
            stats["classes_created"] += 1

        dst = dst_dir / fname
        if dst.exists():
            stats["skipped_collision"] += 1
            if verbose:
                print(f"[fix-val] Exists, skipping: {dst}")
            continue

        if verbose:
            print(f"[fix-val] {'DRY-RUN would move' if dry_run else 'Moving'} {src} -> {dst}")
        if not dry_run:
            shutil.move(str(src), str(dst))
        stats["moved"] += 1

    if not dry_run and images_dir.exists():
        try:
            images_dir.rmdir()
        except OSError:
            pass
    return stats


def stream_download(url: str, out_path: Path, verbose: bool = False, retries: int = 3, timeout: int = 60):
    """
    Download a URL to out_path with a simple progress indicator. Standard library only.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            if verbose:
                print(f"[download] GET {url} (attempt {attempt}/{retries})")
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=timeout) as resp, out_path.open("wb") as out:
                total = resp.length or 0
                read = 0
                chunk = 1024 * 256
                last_print = time.time()
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    out.write(buf)
                    read += len(buf)
                    now = time.time()
                    if verbose and (now - last_print > 0.5):
                        if total:
                            pct = (read / total) * 100
                            print(f"[download] {read/1e6:.1f}MB / {total/1e6:.1f}MB ({pct:.1f}%)")
                        else:
                            print(f"[download] {read/1e6:.1f}MB downloaded")
                        last_print = now
            if verbose:
                print(f"[download] Saved to {out_path} ({out_path.stat().st_size/1e6:.1f}MB)")
            return True
        except (HTTPError, URLError, TimeoutError) as e:
            if attempt >= retries:
                if verbose:
                    print(f"[download] Failed: {e}")
                return False
            if verbose:
                print(f"[download] Error: {e} — retrying…")
            time.sleep(2)


def extract_zip(zip_path: Path, dest_dir: Path, verbose: bool = False):
    if verbose:
        print(f"[extract] {zip_path} -> {dest_dir}")
    with ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def locate_extracted_root(base: Path, verbose: bool = False) -> Path | None:
    """
    After extraction, find the directory that directly contains train/ and val/.
    Check base, then one level deep.
    """
    candidates = [base]
    if base.is_dir():
        for child in base.iterdir():
            if child.is_dir():
                candidates.append(child)
    for c in candidates:
        if has_expected_layout(c):
            return c
        if verbose:
            print(f"[locate] Not a valid root: {c} ({diagnose_layout(c)})")
    return None


def ensure_dataset(root: Path, url: str, auto_download: bool, verbose: bool):
    """
    Ensure the dataset exists under `root`. If not and auto_download is True,
    download to root.parent and extract.
    Returns (resolved_root: Path | None, actions: dict)
    """
    actions = {"downloaded": False, "extracted": False, "zip_path": None}

    # If user passed a parent folder, try to locate a valid root beneath it.
    if has_expected_layout(root):
        return root, actions

    # Try to find a valid child (e.g., tiny-imagenet-200 nested)
    child = locate_extracted_root(root, verbose=verbose)
    if child:
        return child, actions

    if not auto_download:
        return None, actions

    # Prepare download target
    parent = root if root.suffix.lower() == ".zip" else root.parent
    parent.mkdir(parents=True, exist_ok=True)
    zip_path = parent / "tiny-imagenet-200.zip"
    actions["zip_path"] = str(zip_path)

    ok = stream_download(url, zip_path, verbose=verbose)
    if not ok:
        return None, actions
    actions["downloaded"] = True

    # Extract to parent dir
    extract_zip(zip_path, parent, verbose=verbose)
    actions["extracted"] = True

    # Try to locate the extracted dataset
    resolved = locate_extracted_root(parent, verbose=verbose)
    if resolved is None:
        # last resort: maybe the archive extracted as ./tiny-imagenet-200/tiny-imagenet-200/...
        nested = parent / "tiny-imagenet-200"
        if nested.exists():
            resolved = locate_extracted_root(nested, verbose=verbose)

    return resolved, actions


def main():
    ap = argparse.ArgumentParser(description="Download (if needed) and materialize Tiny-ImageNet.")
    ap.add_argument("--root", required=True,
                    help="Desired dataset root (either the final folder that will contain train/ and val/, "
                         "OR a parent under which the dataset will be extracted).")
    ap.add_argument("--fix-val", action="store_true",
                    help="Rearrange val/images into per-class folders using val_annotations.txt.")
    ap.add_argument("--auto-download", action="store_true",
                    help="If dataset is missing, download and extract automatically.")
    ap.add_argument("--url", default=DEFAULT_URL, help="Download URL for tiny-imagenet-200.zip")
    ap.add_argument("--dry-run", action="store_true", help="Preview fix-val moves without changing files.")
    ap.add_argument("--verbose", "-v", action="store_true", help="Print detailed progress.")
    args = ap.parse_args()

    root = Path(args.root)
    report = {
        "dataset": "tiny-imagenet",
        "requested_root": str(root),
        "resolved_root": None,
        "auto_download": bool(args.auto_download),
        "url": args.url,
        "actions": {},
        "ok": False,
    }

    resolved, actions = ensure_dataset(root, args.url, args.auto_download, args.verbose)
    report["actions"] = actions
    if resolved is None:
        report["paths"] = diagnose_layout(root)
        report["error"] = (
            "Dataset not found and could not be downloaded/extracted automatically. "
            "Check 'paths' and 'actions.zip_path' (if any)."
        )
        print(json.dumps(report, indent=2))
        sys.exit(0)

    report["resolved_root"] = str(resolved)
    report["paths"] = diagnose_layout(resolved)
    report["ok"] = has_expected_layout(resolved)

    # Optionally fix validation split
    if args.fix_val:
        stats = fix_validation_split(resolved / "val", dry_run=args.dry_run, verbose=args.verbose)
        report["val_fix_attempted"] = True
        report["val_fix_stats"] = stats
    else:
        report["val_fix_attempted"] = False

    # Final state
    report["ok_after_fix"] = has_expected_layout(resolved)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
