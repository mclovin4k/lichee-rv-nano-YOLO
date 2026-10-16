#include "yolo.hpp"

#include <cstdio>

#include "vpss_det.h"

int YoloModelDetector::setup_model(const char *model_path, int num_classes, float threshold,
                                   float nms_iou) {
    release();
    if (nv_yolo8_open(&y_, model_path, num_classes, threshold, nms_iou) != 0) {
        fprintf(stderr, "Failed to setup model %s\n", model_path);
        return -1;
    }
    opened_ = true;
    return 0;
}

int YoloModelDetector::detect_bgr(const uint8_t *bgr, int width, int height, uint32_t stride,
                                  std::vector<DetBox> &out) {
    out.clear();
    if (!opened_)
        return -1;

    nv_yolo8_det_t dets[64];
    int n = 0;
    int rc = nv_yolo8_infer_bgr(&y_, bgr, stride, width, height, dets, 64, &n);
    if (rc == 0) {
        out.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; i++) {
            DetBox d;
            d.x1 = dets[i].x1;
            d.y1 = dets[i].y1;
            d.x2 = dets[i].x2;
            d.y2 = dets[i].y2;
            d.score = dets[i].score;
            d.cls = dets[i].cls;
            out.push_back(d);
        }
    }
    return rc;
}

int YoloModelDetector::detect_nv21(const uint8_t *y_plane, const uint8_t *uv_plane, int width,
                                   int height, uint32_t stride, std::vector<DetBox> &out) {
    out.clear();
    if (!opened_)
        return -1;

    nv_yolo8_det_t dets[64];
    int n = 0;
    int rc = nv_yolo8_infer(&y_, y_plane, uv_plane, stride, width, height, 0 /* NV21 */, dets, 64, &n);
    if (rc == 0) {
        out.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; i++) {
            DetBox d;
            d.x1 = dets[i].x1;
            d.y1 = dets[i].y1;
            d.x2 = dets[i].x2;
            d.y2 = dets[i].y2;
            d.score = dets[i].score;
            d.cls = dets[i].cls;
            out.push_back(d);
        }
    }
    return rc;
}

int YoloModelDetector::detect_vpss(int width, int height, int timeout_ms, std::vector<DetBox> &out) {
    out.clear();
    if (!opened_)
        return -1;

    /* The tensor is filled by the VPSS channel; here it is only copied from
       the frame into the NPU input -- the same pointer the runtime handed us
       at open(), so its command cache stays valid. */
    const size_t plane = static_cast<size_t>(y_.in_w) * static_cast<size_t>(y_.in_h);
    double prep_ms = 0.0;
    if (vpss_det_fetch(y_.in_ptr, y_.in_ptr + plane, y_.in_ptr + 2 * plane, plane, timeout_ms,
                       &prep_ms) != 0) {
        return -3;
    }

    nv_yolo8_det_t dets[64];
    int n = 0;
    int rc = nv_yolo8_infer_filled(&y_, width, height, prep_ms, dets, 64, &n);
    if (rc == 0) {
        out.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; i++) {
            DetBox d;
            d.x1 = dets[i].x1;
            d.y1 = dets[i].y1;
            d.x2 = dets[i].x2;
            d.y2 = dets[i].y2;
            d.score = dets[i].score;
            d.cls = dets[i].cls;
            out.push_back(d);
        }
    }
    return rc;
}

void YoloModelDetector::release() {
    if (opened_) {
        nv_yolo8_close(&y_);
        opened_ = false;
    }
}
