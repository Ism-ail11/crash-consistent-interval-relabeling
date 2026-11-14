"""Download, verify and parse the two original public inertial datasets."""

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
import re
import shutil
from tempfile import NamedTemporaryFile
import urllib.request
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .model import DIM, STRIDE, VECTORS, WINDOW, quantize_samples
from .native import ROOT, features

DATASETS = ("HAPT", "MotionSense")
LABELS = {
    "HAPT": ("Walk", "Stairs up", "Stairs down", "Sit", "Stand", "Lie"),
    "MotionSense": ("Walk", "Stairs up", "Stairs down", "Sit", "Stand", "Jog"),
}
NOISE_PAIRS = {"HAPT": (1, 0, 0, 4, 3, 3), "MotionSense": (1, 0, 0, 4, 3, 0)}
MOTION_LABEL = {"wlk": 0, "ups": 1, "dws": 2, "sit": 3, "std": 4, "jog": 5}
MOTION_TEST = {11, 12, 13, 14, 15, 16}
FILES = {
    "HAPT": (
        "hapt.zip",
        "https://archive.ics.uci.edu/static/public/341/"
        "smartphone+based+recognition+of+human+activities+and+postural+transitions.zip",
        "4ac4ae064227c07045a99551876b54d204837e995e298b6488559e209b3deb09",
    ),
    "MotionSense": (
        "motionsense.zip",
        "https://raw.githubusercontent.com/mmalekzadeh/motion-sense/"
        "2a0c72755f898ae509e5cb3a1cd58f8c55db5499/data/A_DeviceMotion_data.zip",
        "bc632abddbb1e92db31daa7b740710bc8f756f90febe385e6f1dc32ff8d92285",
    ),
}
DEFAULT_DATA = ROOT / "data"


@dataclass(frozen=True)
class Segment:
    subject: int
    trial: int
    label: int
    group: int
    test: bool
    samples: np.ndarray


def dataset_file(dataset, root=DEFAULT_DATA):
    name, url, digest = FILES[dataset]
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    if not target.exists():
        with NamedTemporaryFile(dir=root, prefix=f".{name}.", suffix=".download", delete=False) as f:
            temporary = Path(f.name)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "har-reproducibility/1"})
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as f:
                shutil.copyfileobj(response, f)
            with temporary.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError(f"Checksum mismatch for {dataset}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    with target.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
            raise ValueError(f"Checksum mismatch for {target}. Remove the invalid archive and retry.")
    return target


def hapt_segments(z, only_subject=None):
    ranges = np.loadtxt(BytesIO(z.read("RawData/labels.txt")), dtype=np.int32)
    experiments = defaultdict(list)
    for exp, subject, activity, first, end in ranges:
        if 1 <= activity <= 6 and (only_subject is None or subject == only_subject):
            experiments[(int(subject), int(exp))].append((int(activity) - 1, int(first), int(end)))
    first_trial = {subject: min(e for s, e in experiments if s == subject) for subject, _ in experiments}
    for (subject, trial), labels in sorted(experiments.items()):
        suffix = f"exp{trial:02}_user{subject:02}.txt"
        acc = np.loadtxt(BytesIO(z.read(f"RawData/acc_{suffix}")))
        gyro = np.loadtxt(BytesIO(z.read(f"RawData/gyro_{suffix}")))
        imu = quantize_samples(np.column_stack((acc, gyro)))
        for group, (label, first, end) in enumerate(labels):
            # Original annotations are one-based and inclusive.
            yield Segment(subject, trial, label, group, trial != first_trial[subject], imu[first - 1:end])


def motionsense_segments(z, only_subject=None):
    pattern = re.compile(r"A_DeviceMotion_data/([a-z]+)_(\d+)/sub_(\d+)\.csv")
    axes = "xyz"
    gravity = [f"gravity.{a}" for a in axes]
    linear = [f"userAcceleration.{a}" for a in axes]
    gyro = [f"rotationRate.{a}" for a in axes]
    paths = []
    for member in z.namelist():
        match = pattern.fullmatch(member)
        if match:
            activity, trial, subject = match.groups()
            if only_subject is None or int(subject) == only_subject:
                paths.append((int(subject), int(trial), activity, member))
    for subject, trial, activity, path in sorted(paths):
        with z.open(path) as stream:
            frame = pd.read_csv(stream, usecols=gravity + linear + gyro)
        imu = quantize_samples(np.column_stack((
            frame[gravity].to_numpy() + frame[linear].to_numpy(), frame[gyro].to_numpy()
        )))
        yield Segment(subject, trial, MOTION_LABEL[activity], trial, trial in MOTION_TEST, imu)


def segments(dataset, root=DEFAULT_DATA, subject=None):
    with ZipFile(dataset_file(dataset, root)) as archive:
        reader = hapt_segments if dataset == "HAPT" else motionsense_segments
        yield from reader(archive, subject)


def windows_of(samples):
    # Anchor four consecutive, overlapping feature windows to each one-second block.
    starts = np.arange(0, len(samples) - WINDOW + 1, STRIDE)
    starts = starts[:len(starts) // VECTORS * VECTORS]
    return np.ascontiguousarray(samples[starts[:, None] + np.arange(WINDOW)])


def prepare(dataset, data_root=DEFAULT_DATA, cache=ROOT / "build" / "data"):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{dataset.lower()}-v1.npz"
    # Invalidate the derived cache when preprocessing or C feature extraction changes.
    content = [Path(__file__), ROOT / "har" / "model.py", ROOT / "firmware" / "features.c",
               ROOT / "firmware" / "har.h"]
    signature = hashlib.sha256(b"".join(path.read_bytes() for path in content)).hexdigest()
    if target.exists():
        with np.load(target) as saved:
            if saved["signature"].item() == signature:
                return {name: saved[name] for name in
                        ("raw", "subject", "trial", "label", "group", "test", "block_index")}
    columns = defaultdict(list)
    for seg in segments(dataset, data_root):
        windows = windows_of(seg.samples)
        if not len(windows):
            continue
        block_features = features(windows).reshape(-1, VECTORS, DIM)
        columns["raw"].append(block_features)
        columns["block_index"].extend(range(len(block_features)))
        for name in ("subject", "trial", "label", "group", "test"):
            columns[name].extend([getattr(seg, name)] * len(block_features))
    prepared = {key: (np.concatenate(value) if key == "raw" else np.asarray(value))
                for key, value in columns.items()}
    with NamedTemporaryFile(dir=cache, suffix=".npz", delete=False) as f:
        temporary = Path(f.name)
    try:
        np.savez_compressed(temporary, signature=signature, **prepared)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return prepared
