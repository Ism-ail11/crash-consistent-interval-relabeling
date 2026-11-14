"""Fixed-point feature normalization, centroids, and independent vector replay."""

from dataclasses import dataclass

import numpy as np

CLASSES = 6
DIM = 24
CHANNELS = 6
WINDOW = 100
STRIDE = 25
VECTORS = 4
FEATURE_LIMIT = 2047
PRIOR_COUNT = 8


def rounded_div(value, denominator):
    value = np.asarray(value, dtype=np.int64)
    return np.sign(value) * ((np.abs(value) + denominator // 2) // denominator)


def quantize_samples(samples):
    return np.clip(np.rint(samples * 256), -32768, 32767).astype(np.int16)


@dataclass(frozen=True)
class Metric:
    center: np.ndarray
    scale: np.ndarray

    def apply(self, features):
        scaled = rounded_div((features.astype(np.int64) - self.center) * self.scale, 4096)
        return np.clip(scaled, -FEATURE_LIMIT, FEATURE_LIMIT).astype(np.int16)


def fit_metric(features):
    """Fit on source subjects only. A floor limits weight on nearly constant axes."""
    center = np.rint(features.mean(axis=0)).astype(np.int32)
    sigma = np.maximum(features.std(axis=0), 12.0)
    scale = np.clip(np.rint(96.0 * 4096 / sigma), 1, 4096).astype(np.int32)
    return Metric(center, scale)


def statistics(vectors, labels):
    sums = np.zeros((CLASSES, DIM), dtype=np.int64)
    counts = np.bincount(labels, minlength=CLASSES).astype(np.int64)
    np.add.at(sums, labels, vectors)
    return sums.astype(np.int32), counts.astype(np.uint32)


def rebuild(blocks, block_labels):
    """Independent oracle. Rebuild from all unaggregated, quantized vectors."""
    return statistics(blocks.reshape(-1, DIM), np.repeat(block_labels, VECTORS))


def centroids(sums, counts, prior, weight):
    den = counts.astype(np.int64)[:, None] + weight
    return np.where(
        den > 0, rounded_div(sums + prior.astype(np.int64) * weight, np.maximum(den, 1)), prior
    ).astype(np.int16)


def predict(embeddings, prototypes):
    differences = embeddings.astype(np.int32)[:, None, :] - prototypes.astype(np.int32)[None, :, :]
    return np.argmin(np.sum(differences * differences, axis=2), axis=1).astype(np.uint8)


def f1_score(y, predicted):
    confusion = np.bincount(
        y.astype(np.int32) * CLASSES + predicted, minlength=CLASSES * CLASSES
    ).reshape(CLASSES, CLASSES)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = 2 * confusion.diagonal() / np.maximum(denominator, 1)
    return float(np.mean(f1)), f1, confusion
