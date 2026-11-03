#ifndef NANO_YOLO8_H
#define NANO_YOLO8_H

#include <stdint.h>
#include <stddef.h>

#include "cviruntime.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Bounded wait for one forward. Comfortably above a normal inference (~40 ms)
   and below the driver's completion timeout,
   so a stall is ours to handle rather than something the process sits
   through. */
#define YOLO8_FORWARD_DEADLINE_MS 800

void nv_yolo8_set_stretch(int stretch);

typedef struct {
  float x1, y1, x2, y2;
  float score;
  int cls;
} nv_yolo8_det_t;

typedef struct {
  void *model; /* CVI_MODEL_HANDLE */
  uint8_t *in_ptr;
  size_t in_size;
  int in_w, in_h;
  uint8_t *in_rgb;

  /* The tensor descriptors must be handed back to ForwardAsync verbatim.
     Passing NULL and hoping the runtime substitutes them segfaults — that was
     the crash in the first cut of this module. */
  CVI_TENSOR *in, *out;
  int n_in, n_out;

  int8_t *out_ptr[6];
  float out_q[6];
  int out_zero[6];
  int out_ch[6], out_w[6], out_h[6];

  int num_classes;
  float threshold;
  float nms_iou;

  /* The class branches' sigmoid, precomputed per branch: the decode feeds it
     quantised int8, i.e. one of 256 values, so a table replaces the expf()
     calls that used to dominate the postprocess (8400 cells x 80 classes). */
  float sig_lut[3][256];

  /* Per-cell scratch for the class scan, sized to the largest branch. */
  int8_t *scratch_q;
  uint8_t *scratch_c;
  float *scratch_f; /* float sibling, for the fp32-output exports */
  int scratch_cells;

  /* Some exports (head_hardhat, fire_smoke, traffic_light) leave the outputs
     unquantised: fmt FP32 and the quant scale reads 1.0. The decode reads
     those branches as floats; out_fp32[i] picks the path per branch. */
  int out_fp32[6];

  /* diagnostics */
  double avg_ms, worst_ms;
  double avg_prep, avg_fwd, avg_dec; /* rolling phase costs (ms) */
  long stalls, reloads;
  float in_qscale;
  int in_zero;
  float last_max_score;
} nv_yolo8_t;

int nv_yolo8_open(nv_yolo8_t *y, const char *model_path, int num_classes, float threshold,
                  float nms_iou);
void nv_yolo8_close(nv_yolo8_t *y);

/* Opens a model, prints its full tensor signature and closes it again. Exists
   to tell the anchor-free YOLOv8 heads apart from the face/keypoint/
   classification models, which have to go back through TDL_Detection. No
   inference, no camera, so it is safe to run in a loop over every model. */
int nv_yolo8_probe_model(const char *model_path);

/* Same as nv_yolo8_infer but with the tensor already filled: VPSS has done the
   crop/resize/colour/normalise. planes = R,G,B of in_w*in_h each. */
int nv_yolo8_infer_rgb(nv_yolo8_t *y, const uint8_t *planes, int cam_w, int cam_h,
                       nv_yolo8_det_t *dets, int max_out, int *n_out);

/* Runs one detection on a camera frame given as its luma plane and its
   interleaved chroma plane. uv_swap selects NV12 (1) over NV21 (0).
   Returns 0 on success, -1 on submit failure, -2 on a forward that blew our
   deadline (y->reloads has been bumped and the model needs reopening).
   *n_out receives the number of boxes written to dets. */
int nv_yolo8_infer(nv_yolo8_t *y, const uint8_t *y_plane, const uint8_t *uv_plane, uint32_t stride,
                   int cam_w, int cam_h, int uv_swap, nv_yolo8_det_t *dets, int max_out, int *n_out);

/* Same as nv_yolo8_infer but with an interleaved BGR frame (3 bytes per pixel,
   e.g. from a VPSS channel emitting PIXEL_FORMAT_BGR_888). */
int nv_yolo8_infer_bgr(nv_yolo8_t *y, const uint8_t *bgr, uint32_t stride, int cam_w, int cam_h,
                       nv_yolo8_det_t *dets, int max_out, int *n_out);

/* Same again but with the tensor already filled by the caller (the VPSS
   hardware-input path copies its planes into y->in_ptr). prep_ms is that
   copy's cost, folded into the phase statistics; the decode geometry takes
   the camera dimensions as usual. */
int nv_yolo8_infer_filled(nv_yolo8_t *y, int cam_w, int cam_h, double prep_ms,
                          nv_yolo8_det_t *dets, int max_out, int *n_out);

#ifdef __cplusplus
}
#endif

#endif