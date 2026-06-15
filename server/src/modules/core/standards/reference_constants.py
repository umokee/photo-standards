from __future__ import annotations

from infra.storage.file_storage import resolve_storage_path

SUPERPOINT_TORCH_WEIGHTS_PATH = resolve_storage_path("weights/superpoint_v1.pth")
LIGHTGLUE_TORCH_WEIGHTS_PATH = resolve_storage_path("weights/superpoint_lightglue.pth")
TORCH_HUB_DIR = resolve_storage_path("torch-hub")

SUPERPOINT_INPUT_STRIDE = 8

# Feature budgets are intentionally split by workload.
#
# * reference: computed once and cached for the standard image, so it can be dense;
# * photo: one-off inspection image, allowed to be heavier than realtime;
# * video: live frame path, must stay small and predictable.
SUPERPOINT_REFERENCE_MAX_SIDE: int | None = 2048
SUPERPOINT_PHOTO_MAX_SIDE: int | None = 2048
SUPERPOINT_VIDEO_MAX_SIDE: int = 640

SUPERPOINT_REFERENCE_MAX_KEYPOINTS = 4096
SUPERPOINT_PHOTO_MAX_KEYPOINTS = 2048
SUPERPOINT_VIDEO_MAX_KEYPOINTS = 512

# Grid-balanced keypoint selection keeps high-score features from collapsing
# into one textured area while keeping the extractor budget predictable.
SUPERPOINT_REFERENCE_GRID_ROWS = 12
SUPERPOINT_REFERENCE_GRID_COLS = 12
SUPERPOINT_PHOTO_GRID_ROWS = 8
SUPERPOINT_PHOTO_GRID_COLS = 8
SUPERPOINT_VIDEO_GRID_ROWS = 6
SUPERPOINT_VIDEO_GRID_COLS = 6

# Backward-compatible aliases. New code should choose the explicit profile above.
SUPERPOINT_OFFLINE_MAX_SIDE: int | None = SUPERPOINT_REFERENCE_MAX_SIDE
SUPERPOINT_REALTIME_MAX_SIDE: int = SUPERPOINT_VIDEO_MAX_SIDE

SUPERPOINT_OFFLINE_MAX_KEYPOINTS = SUPERPOINT_REFERENCE_MAX_KEYPOINTS
SUPERPOINT_REALTIME_MAX_KEYPOINTS = SUPERPOINT_VIDEO_MAX_KEYPOINTS

MIN_REFERENCE_KEYPOINTS = 100

MIN_RAW_MATCHES = 18
RANSAC_REPROJECTION_THRESHOLD = 4.0
MIN_INLIERS_FOR_ALIGNMENT = 18
MAX_LIGHTGLUE_MATCH_PAIRS = 4096

ALIGNMENT_GRID_COLS = 4
ALIGNMENT_GRID_ROWS = 4
MIN_ALIGNMENT_GRID_CELLS = 3
MAX_ALIGNMENT_CELL_FRACTION = 0.45
MAX_ALIGNMENT_MEDIAN_ERROR = 10.0

IOU_MATCH_THRESHOLD = 0.25
