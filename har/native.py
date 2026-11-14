"""Typed bindings to the same C implementation compiled for Cortex-M4."""

import ctypes as C
import os
from pathlib import Path
import subprocess
from dataclasses import dataclass

import numpy as np
from numpy.ctypeslib import ndpointer

from .model import CHANNELS, CLASSES, DIM, FEATURE_LIMIT, VECTORS, WINDOW

ROOT = Path(__file__).resolve().parents[1]
SLOT_WORDS = 6
MAX_SLOTS = 128
MAX_BLOCKS = 128
SNAPSHOT_WORDS = 7 + MAX_BLOCKS // 4
CHECKPOINT_WORDS = SNAPSHOT_WORDS + SLOT_WORDS
OK, DUPLICATE, INVALID, CONFLICT, STALE, FULL, IO, CORRUPT = 0, 1, -1, -2, -3, -4, -5, -6
ERASED = 0xFFFFFFFF
SUMMARY_DTYPE = np.dtype([("sum", "<i4", (DIM,)), ("count", "<u2"),
                          ("label", "u1"), ("reserved", "u1")])
VECTORS_DTYPE = np.dtype([("vector", "<i2", (VECTORS, DIM)), ("count", "<u2"),
                          ("label", "u1"), ("reserved", "u1")])
PACKED_DTYPE = np.dtype([("sum_bits", "u1", (42,)), ("count", "u1"), ("label", "u1")])
LAYOUTS = {"summary": SUMMARY_DTYPE, "vectors": VECTORS_DTYPE, "packed": PACKED_DTYPE}


class View(C.Structure):
    _fields_ = [("blocks", C.c_void_p), ("epoch", C.c_uint32), ("size", C.c_uint16),
                ("edit_floor", C.c_uint16), ("layout", C.c_uint8), ("flags", C.c_uint8)]


class CEdit(C.Structure):
    _fields_ = [("id", C.c_uint32), ("first", C.c_uint16), ("end", C.c_uint16),
                ("from_label", C.c_uint8), ("to_label", C.c_uint8)]


class State(C.Structure):
    _fields_ = [("sum", (C.c_int32 * DIM) * CLASSES), ("count", C.c_uint32 * CLASSES),
                ("label", C.c_uint8 * MAX_BLOCKS), ("last_id", C.c_uint32),
                ("next_slot", C.c_uint32), ("generation", C.c_uint32), ("bank", C.c_uint32),
                ("last_edit_known", C.c_uint32), ("last_edit", CEdit)]


class CMetric(C.Structure):
    _fields_ = [("center", C.c_int32 * DIM), ("scale", C.c_int32 * DIM)]


@dataclass(frozen=True)
class Edit:
    id: int
    first: int
    end: int
    from_label: int
    to_label: int

    def c_value(self):
        return CEdit(self.id, self.first, self.end, self.from_label, self.to_label)


def native_library():
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    output = build / "libhar.so"
    sources = [ROOT / "firmware" / name for name in ("har.c", "features.c", "host.c")]
    dependencies = sources + [ROOT / "firmware" / "har.h"]
    if not output.exists() or any(src.stat().st_mtime_ns > output.stat().st_mtime_ns
                                  for src in dependencies):
        tmp = output.with_name(f"{output.name}.{os.getpid()}.tmp")
        subprocess.run(
            [os.environ.get("CC", "cc"), "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
             "-shared", "-fPIC", *map(str, sources), "-o", str(tmp)], check=True
        )
        tmp.replace(output)
    lib = C.CDLL(str(output))
    ptr = C.POINTER
    uint32 = ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS")
    int16 = ndpointer(dtype=np.int16, flags="C_CONTIGUOUS")
    int32 = ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    uint8 = ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
    lib.host_open.argtypes = [ptr(State), ptr(View), uint32, C.c_uint32]
    lib.host_open.restype = C.c_int
    lib.host_edit.argtypes = [ptr(State), ptr(View), uint32, C.c_uint32, CEdit,
                             C.c_int32, C.c_uint32, ptr(C.c_uint32), C.c_int]
    lib.host_edit.restype = C.c_int
    lib.host_compact.argtypes = [ptr(State), ptr(View), uint32, C.c_uint32, C.c_int32,
                                C.c_uint32, ptr(C.c_uint32)]
    lib.host_compact.restype = C.c_int
    lib.host_features.argtypes = [int16, C.c_uint32, int32]
    lib.host_features.restype = None
    lib.host_predict.argtypes = [int16, C.c_uint32, int16, uint8]
    lib.host_predict.restype = None
    lib.har_embed.argtypes = [int32, ptr(CMetric), int16]
    lib.har_embed.restype = None
    lib.har_centroids.argtypes = [ptr(State), int16, C.c_uint32, int16]
    lib.har_centroids.restype = None
    return lib


_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        _LIB = native_library()
    return _LIB


def features(windows):
    windows = np.ascontiguousarray(windows, dtype=np.int16)
    if windows.ndim != 3 or windows.shape[1:] != (WINDOW, CHANNELS):
        raise ValueError("Expected [windows, 100 samples, 6 channels]")
    output = np.empty((len(windows), DIM), dtype=np.int32)
    lib().host_features(windows, len(windows), output)
    return output


def c_embeddings(raw_features, metric):
    out = np.empty((len(raw_features), DIM), dtype=np.int16)
    fitted = CMetric(tuple(metric.center), tuple(metric.scale))
    for row, target in zip(raw_features, out, strict=True):
        lib().har_embed(row, C.byref(fitted), target)
    return out


def c_predict(embeddings, centroids):
    output = np.empty(len(embeddings), dtype=np.uint8)
    lib().host_predict(np.ascontiguousarray(embeddings), len(embeddings),
                       np.ascontiguousarray(centroids), output)
    return output


def pack_blocks(blocks, labels, layout="summary"):
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown layout {layout}")
    if blocks.ndim != 3 or blocks.shape[1:] != (VECTORS, DIM) or len(blocks) != len(labels):
        raise ValueError("Expected four feature vectors per labeled block")
    if not 0 < len(blocks) <= MAX_BLOCKS:
        raise ValueError("Calibration block count exceeds firmware capacity")
    if blocks.dtype != np.int16 or np.any(np.abs(blocks.astype(np.int32)) > FEATURE_LIMIT):
        raise ValueError("Calibration vectors must be bounded int16 embeddings")
    if np.any(labels < 0) or np.any(labels >= CLASSES):
        raise ValueError("Calibration class is out of range")
    packed = np.zeros(len(blocks), dtype=LAYOUTS[layout])
    packed["count"] = VECTORS
    packed["label"] = labels
    if layout == "summary":
        packed["sum"] = blocks.sum(axis=1)
    elif layout == "packed":
        for i, row in enumerate(blocks.sum(axis=1)):
            value = sum((int(coordinate) & 0x3FFF) << (14 * d) for d, coordinate in enumerate(row))
            packed["sum_bits"][i] = np.frombuffer(value.to_bytes(42, "little"), dtype=np.uint8)
    else:
        packed["vector"] = blocks
    return packed


class Journal:
    def __init__(self, packed, epoch=1, edit_floor=0, slots=MAX_SLOTS, words=None, *,
                 verify_crc=True, checkpoint_receipt=True, snapshot_receipt=False):
        if not 0 <= slots <= MAX_SLOTS:
            raise ValueError("Invalid journal capacity")
        if packed.dtype not in LAYOUTS.values():
            raise ValueError("Invalid calibration record format")
        self.packed = np.ascontiguousarray(packed)
        self.slots = slots
        flags = (int(not verify_crc) | (int(not checkpoint_receipt) << 1)
                 | (int(snapshot_receipt) << 2))
        layout = list(LAYOUTS.values()).index(packed.dtype)
        self.view = View(self.packed.ctypes.data, epoch, len(packed), edit_floor,
                         layout, flags)
        self.bank_words = CHECKPOINT_WORDS + slots * SLOT_WORDS
        self.words = np.full(2 * self.bank_words, ERASED, np.uint32) if words is None else words
        if len(self.words) != 2 * self.bank_words:
            raise ValueError("Flash image length does not match the journal")
        self.reboot()

    def reboot(self):
        self.state = State()
        status = lib().host_open(C.byref(self.state), C.byref(self.view), self.words, self.slots)
        if status != OK:
            raise RuntimeError(f"Journal recovery failed: {status}")

    def apply(self, edit, cut=-1, mask=0, *, snapshot=False):
        completed = C.c_uint32()
        status = lib().host_edit(C.byref(self.state), C.byref(self.view), self.words,
                                 self.slots, edit.c_value(), cut, mask, C.byref(completed), snapshot)
        return status, completed.value

    def compact(self, cut=-1, mask=0):
        completed = C.c_uint32()
        status = lib().host_compact(C.byref(self.state), C.byref(self.view), self.words,
                                    self.slots, cut, mask, C.byref(completed))
        return status, completed.value

    def slot_offset(self, slot):
        return self.state.bank * self.bank_words + CHECKPOINT_WORDS + slot * SLOT_WORDS

    def stats(self):
        return np.ctypeslib.as_array(self.state.sum).copy(), np.ctypeslib.as_array(self.state.count).copy()

    def labels(self):
        return np.ctypeslib.as_array(self.state.label)[:self.view.size].copy()

    def prototypes(self, prior, weight):
        output = np.empty((CLASSES, DIM), dtype=np.int16)
        lib().har_centroids(C.byref(self.state), np.ascontiguousarray(prior, dtype=np.int16),
                           weight, output)
        return output
