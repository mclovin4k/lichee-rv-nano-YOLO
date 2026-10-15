#pragma once

#include <cstdint>
#include <vector>

#include "yolo8.h"

struct DetBox {
    float x1, y1, x2, y2;
    float score;
    int cls;
};

/* YOLOv8 detector driven straight through libcviruntime (no TDL layer).
   Ported from the "Default Project" nano-y8 pipeline: the ~2 s
   "cvi-tpu run dmabuf timeout" wedges came from libtdl_core's pre-forward
   path, so this class drives the cvimodel itself -- preprocess, bounded
   forward (YOLO8_FORWARD_DEADLINE_MS) and DFL decode + NMS. */
class YoloModelDetector {
public:
    int setup_model(const char *model_path, int num_classes, float threshold, float nms_iou);

    /* Runs one detection on a BGR frame (3 bytes per pixel, row stride in
       bytes). Returns 0 on success (out filled), -1 on submit failure and
       -2 when the forward blew its deadline -- in that case the model has
       already been dropped and setup_model() must be called again. */
    int detect_bgr(const uint8_t *bgr, int width, int height, uint32_t stride,
                   std::vector<DetBox> &out);

    /* Same, but on an NV21 camera frame given as its luma and interleaved
       chroma planes (stride in bytes applies to both). This is the path used
       by the live pipeline: the VPSS frame goes to the NPU without any BGR
       conversion. */
    int detect_nv21(const uint8_t *y_plane, const uint8_t *uv_plane, int width, int height,
                    uint32_t stride, std::vector<DetBox> &out);

    double avg_ms() const { return y_.avg_ms; }

    void release();

private:
    nv_yolo8_t y_ = {};
    bool opened_ = false;
};
