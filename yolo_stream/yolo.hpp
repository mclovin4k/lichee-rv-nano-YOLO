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

    /* Hardware-input path: the VPSS channel (vpss_det.c) produced the int8
       letterboxed tensor. This waits for a frame, copies it into the NPU input
       and runs the usual forward/decode; width/height are the frame dimensions
       the boxes are mapped back to. Returns -3 when no frame arrived within
       timeout_ms -- a pipeline condition, not an inference failure; -1/-2 as
       the other paths. */
    int detect_vpss(int width, int height, int timeout_ms, std::vector<DetBox> &out);

    double avg_ms() const { return y_.avg_ms; }
    /* Rolling phase costs of the recent inferences (ms): preprocess (or the
       VPSS copy), forward, decode. */
    double prep_ms() const { return y_.avg_prep; }
    double fwd_ms() const { return y_.avg_fwd; }
    double dec_ms() const { return y_.avg_dec; }
    /* Model input size and quantisation, for configuring the VPSS channel. */
    int input_w() const { return y_.in_w; }
    int input_h() const { return y_.in_h; }
    float input_qscale() const { return y_.in_qscale; }
    /* Highest class score of the last inference (per-detection log line). */
    float last_max_score() const { return y_.last_max_score; }

    void release();

private:
    nv_yolo8_t y_ = {};
    bool opened_ = false;
};
