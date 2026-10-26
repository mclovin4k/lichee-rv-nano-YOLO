"""Трекинг рамки между кадрами оптическим потоком (без opencv-contrib)."""

from __future__ import annotations

import cv2
import numpy as np

from .dataset import Box


def track_box(prev_gray: np.ndarray, gray: np.ndarray, box: Box,
              max_points: int = 40) -> tuple[Box | None, float]:
    """Возвращает новую рамку и качество трекинга (0..1).

    Качество — доля удержанных точек. None — объект потерян (мало точек,
    слишком большой скачок): вызывающий код оставляет прежнюю рамку.
    """
    ih, iw = prev_gray.shape[:2]
    h, w = gray.shape[:2]
    if (ih, iw) != (h, w):
        return None, 0.0
    b = box.clipped(iw, ih)
    if b.w < 8 or b.h < 8:
        return None, 0.0

    mask = np.zeros_like(prev_gray)
    x1, y1 = int(round(b.x)), int(round(b.y))
    x2, y2 = int(round(b.x2)), int(round(b.y2))
    mask[max(0, y1):min(ih, y2), max(0, x1):min(iw, x2)] = 255

    p0 = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=max_points, qualityLevel=0.01,
        minDistance=4, mask=mask, blockSize=5)
    if p0 is None or len(p0) < 4:
        return None, 0.0

    p1, status, _err = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, p0, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if p1 is None or status is None:
        return None, 0.0

    good = status.ravel() == 1
    pts0 = p0.reshape(-1, 2)[good]
    pts1 = p1.reshape(-1, 2)[good]
    inside = ((pts1[:, 0] >= 0) & (pts1[:, 0] < iw) &
              (pts1[:, 1] >= 0) & (pts1[:, 1] < ih))
    pts0, pts1 = pts0[inside], pts1[inside]
    if len(pts0) < 6:
        return None, len(pts0) / max(1.0, len(p0))

    delta = pts1 - pts0
    limit = 0.55 * max(b.w, b.h)
    keep = np.linalg.norm(delta, axis=1) < limit
    delta = delta[keep]
    pts0, pts1 = pts0[keep], pts1[keep]
    if len(delta) < 6:
        return None, len(delta) / max(1.0, len(p0))

    dx, dy = float(np.median(delta[:, 0])), float(np.median(delta[:, 1]))

    # масштаб: по разбросу удержанных точек (устойчивее площади рамки)
    spread0 = max(float(np.ptp(pts0[:, 0])), float(np.ptp(pts0[:, 1])), 1.0)
    spread1 = max(float(np.ptp(pts1[:, 0])), float(np.ptp(pts1[:, 1])), 1.0)
    scale = spread1 / spread0
    scale = max(0.85, min(1.18, scale))

    cx = b.x + b.w / 2.0 + dx
    cy = b.y + b.h / 2.0 + dy
    nw, nh = b.w * scale, b.h * scale
    new = Box(cx - nw / 2.0, cy - nh / 2.0, nw, nh, b.cls, b.conf).clipped(iw, ih)
    quality = len(delta) / max(1.0, len(p0))
    return new, quality


def track_boxes(prev_gray: np.ndarray, gray: np.ndarray,
                boxes: list[Box]) -> tuple[list[Box], list[float]]:
    """Трекает список рамок; потерянные остаются на месте с quality=0."""
    out: list[Box] = []
    quality: list[float] = []
    for b in boxes:
        nb, q = track_box(prev_gray, gray, b)
        if nb is None:
            out.append(b)
            quality.append(0.0)
        else:
            out.append(nb)
            quality.append(q)
    return out, quality
