import numpy as np


BIN_SIZE_DEG = 2
NUM_BINS = 180  # 360 / BIN_SIZE_DEG
MIN_VALID_BINS = 20        # need enough overlap to trust the comparison
ZUPT_DIST_THRESHOLD_MM = 15  # avg bin distance diff below this = "looks static"
ZUPT_CONSECUTIVE_REQUIRED = 5  # require N consecutive similar scans before triggering

def scan_to_bins(scan):
    """
    Structure the scan in bins
    """
    bins = np.full(NUM_BINS, np.nan)
    for point in scan:
        angle = point[1] % 360
        dist = point[2]
        idx = int(angle // BIN_SIZE_DEG) % NUM_BINS
        bins[idx] = dist  # last point in bin wins; fine for this coarse check
    return bins

def scan_similarity(bins_a, bins_b, percentile=85):
    """
    Find similarity between bins of two scans using a high percentile (default 85th)
    so that localized motion (e.g. a walking person or small platform translation)
    is not hidden by static background walls.
    """
    valid = ~np.isnan(bins_a) & ~np.isnan(bins_b)
    if valid.sum() < MIN_VALID_BINS:
        return None  # not enough overlapping structure to judge
    diffs = np.abs(bins_a[valid] - bins_b[valid])
    return float(np.percentile(diffs, percentile))