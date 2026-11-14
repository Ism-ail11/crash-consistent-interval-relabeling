"""Make a GitHub-ready source archive from a verified experiment tree."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
TREES = (".github", "firmware", "har", "tools", "tests", "notebooks", "results")
TOP_LEVEL = (".gitignore", "README.md", "METHOD.md", "DATASETS.md", "Makefile", "requirements.txt")
EXCLUDE = {"__pycache__", ".ipynb_checkpoints", ".DS_Store"}


def make_zip(output):
    paths = [ROOT / name for name in TOP_LEVEL]
    for folder in TREES:
        paths.extend(p for p in (ROOT / folder).rglob("*")
                     if p.is_file() and p.suffix.lower() != ".pdf"
                     and not any(part in EXCLUDE for part in p.relative_to(ROOT).parts))
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(paths):
            name = Path("crash-consistent-har-code") / path.relative_to(ROOT)
            info = ZipInfo(name.as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=ZIP_DEFLATED, compresslevel=6)
    with ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive failed its CRC check")
        assert len(archive.namelist()) == len(paths)
    print(f"{output.name}: {len(paths)} files, {output.stat().st_size:,} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?",
                        default=ROOT.parent / "crash-consistent-har-code.zip")
    make_zip(parser.parse_args().output.resolve())
