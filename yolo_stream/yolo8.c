/* YOLOv8 detector driven straight through libcviruntime, replacing
   TDL_Detection.

   WHY: measured on this board (see CHECKPOINT.md 4.5) the ~2 s
   "cvi-tpu run dmabuf timeout" wedges come from libtdl_core's pre-forward
   path, not from the NPU or its driver. Driving the same cvimodel straight
   through cviruntime gave 18 500 consecutive inferences with zero kernel
   timeouts and a worst case of 23 ms, while the identical model through
   TDL_Detection wedged 3-4 times per 240 s at 2036 ms spikes. libtdl_core
   cannot be skipped by handing it a TENSOR_FRAME image: the public API has
   no such constructor (ImageFactory::createImage yields VPSS_FRAME or
   RAW_FRAME, and the preprocessor then rejects it), so the honest fix is to
   stop using that layer for inference at all.

   What this file owns, all of it:
     - preprocessing: camera NV12/NV21 -> RGB planar, letterboxed to the model
       input, mean/std normalised to the tensor;
     - inference: CVI_NN_ForwardAsync with a pollable wait, so a stalled
       forward can be detected and abandoned instead of freezing the process
       for the driver's fixed 2000 ms;
     - postprocessing: the standard anchor-free YOLOv8 head (3 box branches +
       3 class branches, 5040 candidates) decoded and NMS'd on the CPU.

   Output layout confirmed on the board for
   yolov8n_det_person_vehicle_384_640_INT8_cv181x:
     box_Conv   1 x 64 x 48 x 80   (stride 8)   qscale 0.131423
     cls_Conv   1 x  7 x 48 x 80               qscale 0.199447
     706_Conv   1 x 64 x 24 x 40   (stride 16)  qscale 0.109056
     724_Conv   1 x  7 x 24 x 40               qscale 0.347555
     715_Conv   1 x 64 x 12 x 20   (stride 32)  qscale 0.109056
     733_Conv   1 x  7 x 12 x 20               qscale 0.403502
   64 box channels = 4 params x 16 groups; 7 cls channels = 7 classes
   (person_vehicle). 48*80 + 24*40 + 12*20 = 5040 candidate cells. */

#include "yolo8.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "cviruntime.h"

/* ---- small helpers ------------------------------------------------------ */

/* Tensor writing: planar NCHW, RGB order, gain from the tensor's own qscale,
   no offset -- all three verified against the TDL reference implementation on
   identical frames (see tensor_from_rgb and preprocess for the evidence). */
static int g_stretch = 0;   /* 1 = fill the model rect, distorting aspect */

void nv_yolo8_set_stretch(int stretch) { g_stretch = stretch; }

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static inline int clamp_q(float v) {
  int q = (int)(v + 0.5f);
  if (q > 127) q = 127;
  if (q < -128) q = -128;
  return q;
}

/* ---- open / close ------------------------------------------------------- */

int nv_yolo8_open(nv_yolo8_t *y, const char *model_path, int num_classes, float threshold,
                  float nms_iou) {
  memset(y, 0, sizeof(*y));
  y->num_classes = num_classes;
  y->threshold = threshold;
  y->nms_iou = nms_iou;

  if (CVI_NN_RegisterModel(model_path, &y->model) != CVI_RC_SUCCESS) {
    fprintf(stderr, "yolo8: RegisterModel(%s) failed\n", model_path);
    return -1;
  }

  CVI_TENSOR *in = NULL, *out = NULL;
  int32_t n_in = 0, n_out = 0;
  if (CVI_NN_GetInputOutputTensors(y->model, &in, &n_in, &out, &n_out) != CVI_RC_SUCCESS) {
    fprintf(stderr, "yolo8: GetInputOutputTensors failed\n");
    return -1;
  }
  if (n_in != 1 || n_out != 6) {
    /* Only the 3-branch anchor-free YOLOv8 head is implemented here. Anything
       else (face, keypoint, classification) still has to go through libtdl. */
    fprintf(stderr, "yolo8: unsupported tensor layout in=%d out=%d (want 1/6)\n", n_in, n_out);
    return -1;
  }

  y->in_ptr = (uint8_t *)CVI_NN_TensorPtr(&in[0]);
  y->in_size = CVI_NN_TensorSize(&in[0]);
  y->in = in;
  y->out = out;
  y->n_in = n_in;
  y->n_out = n_out;
  if (y->in_ptr == NULL || y->in_size == 0) return -1;

  /* The input tensor's own quantisation: preprocess() fills the tensor in
     these units (q = pixel/std * qscale, no offset -- the reference config's
     mean is 0). */
  y->in_qscale = CVI_NN_TensorQuantScale(&in[0]);
  y->in_zero = (int)CVI_NN_TensorQuantZeroPoint(&in[0]);

  CVI_SHAPE s = CVI_NN_TensorShape(&in[0]);
  /* cviruntime reports NCHW as dim[0..3] */
  y->in_h = (int)s.dim[2];
  y->in_w = (int)s.dim[3];
  if (y->in_w <= 0 || y->in_h <= 0) {
    fprintf(stderr, "yolo8: bad input shape\n");
    return -1;
  }

  for (int i = 0; i < 6; i++) {
    CVI_SHAPE o = CVI_NN_TensorShape(&out[i]);
    y->out_ptr[i] = (int8_t *)CVI_NN_TensorPtr(&out[i]);
    y->out_q[i] = CVI_NN_TensorQuantScale(&out[i]);
    y->out_zero[i] = CVI_NN_TensorQuantZeroPoint(&out[i]);
    y->out_ch[i] = (int)o.dim[1];
    /* cviruntime reports NCHW, so dim[2] is height and dim[3] is width. The
       first cut had these the other way round, which transposed the whole
       decode grid without changing the number of candidates examined. */
    y->out_h[i] = (int)o.dim[2];
    y->out_w[i] = (int)o.dim[3];
    y->out_fp32[i] = (out[i].fmt == CVI_FMT_FP32);
  }

  /* Identify the head and, when the caller asked for "auto" (cnt <= 0), take
     the class count from the tensor instead of guessing. Every YOLOv8 det
     export shares this skeleton: three box branches of 4x16 DFL channels and
     three class branches of one channel per class. Anything else (SCRFD,
     keypoints, classifiers) fails here with a clear message instead of
     decoding noise. */
  if (y->out_ch[0] != 64 || y->out_ch[1] != 64 || y->out_ch[2] != 64 || y->out_ch[3] <= 0 ||
      y->out_ch[4] != y->out_ch[3] || y->out_ch[5] != y->out_ch[3]) {
    fprintf(stderr, "yolo8: not a YOLOv8 det head (box ch %d/%d/%d, cls ch %d/%d/%d)\n",
            y->out_ch[0], y->out_ch[1], y->out_ch[2], y->out_ch[3], y->out_ch[4], y->out_ch[5]);
    return -1;
  }
  if (y->out_fp32[0] != y->out_fp32[3] || y->out_fp32[1] != y->out_fp32[4] ||
      y->out_fp32[2] != y->out_fp32[5]) {
    fprintf(stderr, "yolo8: box and cls branches disagree on the output format\n");
    return -1;
  }
  if (num_classes <= 0) {
    num_classes = (int)y->out_ch[3];
    printf("yolo8: auto-detect: %d classes from the tensor\n", num_classes);
  } else if (num_classes != (int)y->out_ch[3]) {
    fprintf(stderr, "yolo8: warning: %d classes requested, tensor has %d\n", num_classes,
            (int)y->out_ch[3]);
  }
  y->num_classes = num_classes;

  /* The class-branch sigmoid: its argument is a quantised int8, so all 256
     outputs are a table built once here. The decode visits 8400 cells x 80
     classes per frame; an expf() for every one of those was the postprocess. */
  for (int b = 0; b < 3; b++) {
    const float zc = (float)y->out_zero[b + 3];
    const float qc = y->out_q[b + 3];
    for (int q = -128; q <= 127; q++)
      y->sig_lut[b][q + 128] = 1.0f / (1.0f + expf(-(((float)q - zc) * qc)));
  }

  /* Scratch for the per-cell class scan, sized to the largest branch. */
  for (int i = 0; i < 3; i++) {
    const int c = y->out_w[i] * y->out_h[i];
    if (c > y->scratch_cells) y->scratch_cells = c;
  }
  y->scratch_q = malloc((size_t)y->scratch_cells);
  y->scratch_c = malloc((size_t)y->scratch_cells);
  if (y->scratch_q == NULL || y->scratch_c == NULL) {
    fprintf(stderr, "yolo8: no memory for the decode scratch\n");
    return -1;
  }
  if (y->out_fp32[0] || y->out_fp32[1] || y->out_fp32[2]) {
    y->scratch_f = malloc((size_t)y->scratch_cells * sizeof(float));
    if (y->scratch_f == NULL) {
      fprintf(stderr, "yolo8: no memory for the float decode scratch\n");
      return -1;
    }
  }

  /* Infer the letterbox geometry for a given camera frame. */
  y->in_rgb = malloc((size_t)y->in_w * y->in_h * 3);
  if (y->in_rgb == NULL) {
    fprintf(stderr, "yolo8: no memory for %dx%dx3 staging\n", y->in_w, y->in_h);
    return -1;
  }

  printf("yolo8: %s input=%dx%d classes=%d thr=%.2f iou=%.2f\n", model_path, y->in_w, y->in_h,
         num_classes, threshold, nms_iou);
  printf("yolo8: input tensor ptr=%p size=%zu qscale=%f zero=%d\n", (void *)y->in_ptr, y->in_size,
         y->in_qscale, y->in_zero);
  for (int i = 0; i < 3; i++) {
    /* bytes per element tells int8 (1) from fp32 (4) -- the three shipped
       exports whose quant scale reads 1.0 are the fp32-output ones. */
    const size_t bz = CVI_NN_TensorSize(&out[i]);
    const size_t cz = CVI_NN_TensorSize(&out[i + 3]);
    const size_t be = (size_t)y->out_ch[i] * y->out_h[i] * y->out_w[i];
    const size_t ce = (size_t)y->out_ch[i + 3] * y->out_h[i + 3] * y->out_w[i + 3];
    printf("yolo8:   scale %d: box %dx%d esz=%.1f q=%.6f z=%d | cls %dx%d esz=%.1f q=%.6f z=%d\n", i,
           y->out_ch[i], y->out_h[i], be ? (double)bz / (double)be : -1.0, y->out_q[i],
           y->out_zero[i], y->out_ch[i + 3], y->out_h[i + 3],
           ce ? (double)cz / (double)ce : -1.0, y->out_q[i + 3], y->out_zero[i + 3]);
  }
  return 0;
}

void nv_yolo8_close(nv_yolo8_t *y) {
  if (y->model != NULL) CVI_NN_CleanupModel(y->model);
  free(y->in_rgb);
  free(y->scratch_q);
  free(y->scratch_c);
  free(y->scratch_f);
  memset(y, 0, sizeof(*y));
}

int nv_yolo8_probe_model(const char *model_path) {
  void *model = NULL;
  if (CVI_NN_RegisterModel(model_path, &model) != CVI_RC_SUCCESS || model == NULL) {
    printf("PROBE %-56s REGISTER_FAILED\n", model_path);
    return -1;
  }
  CVI_TENSOR *in = NULL, *out = NULL;
  int32_t n_in = 0, n_out = 0;
  if (CVI_NN_GetInputOutputTensors(model, &in, &n_in, &out, &n_out) != CVI_RC_SUCCESS) {
    printf("PROBE %-56s NO_TENSORS\n", model_path);
    CVI_NN_CleanupModel(model);
    return -1;
  }
  const char *base = strrchr(model_path, '/');
  base = base ? base + 1 : model_path;

  if (n_in >= 1) {
    CVI_SHAPE s = CVI_NN_TensorShape(&in[0]);
    printf("PROBE %-46s in=%d [%d %d %d %d] q=%f z=%d | out=%d :",
           base, n_in, (int)s.dim[0], (int)s.dim[1], (int)s.dim[2], (int)s.dim[3],
           CVI_NN_TensorQuantScale(&in[0]), (int)CVI_NN_TensorQuantZeroPoint(&in[0]), n_out);
  } else {
    printf("PROBE %-46s in=%d | out=%d :", base, n_in, n_out);
  }
  for (int i = 0; i < n_out; i++) {
    CVI_SHAPE o = CVI_NN_TensorShape(&out[i]);
    printf(" [%d %d %d %d]", (int)o.dim[0], (int)o.dim[1], (int)o.dim[2], (int)o.dim[3]);
  }
  /* The anchor-free YOLOv8 det head: one input, six outputs, box branches with
     4 params x 16 bins (DFL) and matching class branches. */
  int yolo8 = (n_in == 1 && n_out == 6);
  printf("  => %s\n", yolo8 ? "YOLO8_FAST" : "TDL_ONLY");
  CVI_NN_CleanupModel(model);
  return yolo8 ? 1 : 0;
}

/* ---- preprocessing ------------------------------------------------------ */

/* Camera NV12/NV21 (V/U plane order is swapped between the two) -> RGB planar
   at the model input size, letterboxed with a mid-grey border, mean/std
   normalised. Nearest-neighbour on purpose: this is a detector at 384x640,
   and a bilinear pass costs more than it buys at this resolution. */
/* Geometry of the mapping from the model rect back to the camera frame.
   preprocess() and decode() must agree on it exactly or the boxes land in the
   wrong place, so it lives in one place.

   stretch=1 fills the model rect outright, distorting the aspect ratio. That
   is what the SDK's own tooling does for these exports (a 320x320 model is fed
   a 640x640 image, and feeding it a 640x480 one is explicitly wrong). With a
   384x640 model and a 640x480 camera the ratios differ outright (0.600 vs
   1.333), so letterboxing would waste 64px of width on grey bars per side
   and hand the network a composition it was not calibrated for. */
/* Model pixels = camera pixels times k; camera = (model - pad) / k.
   Letterbox uses one uniform k with centred padding. Stretch fills the rect,
   so k differs per axis and there is no padding; getting this wrong pins all
   boxes into the top in_h rows, because the model height and the camera
   height are not the same here (384 vs 480). */
static void geom(const nv_yolo8_t *y, int cam_w, int cam_h, int *px, int *py, float *kx,
                 float *ky) {
  int rw, rh;
  if (g_stretch) {
    rw = y->in_w;
    rh = y->in_h;
    *px = 0;
    *py = 0;
    *kx = (float)y->in_w / (float)cam_w;
    *ky = (float)y->in_h / (float)cam_h;
  } else {
    float scale = (float)y->in_w / (float)cam_w;
    if ((float)y->in_h / (float)cam_h < scale) scale = (float)y->in_h / (float)cam_h;
    rw = (int)(cam_w * scale);
    rh = (int)(cam_h * scale);
    if (rw < 1) rw = 1;
    if (rh < 1) rh = 1;
    *px = (y->in_w - rw) / 2;
    *py = (y->in_h - rh) / 2;
    *kx = scale;
    *ky = scale;
  }
}

/* RGB staging buffer -> int8 tensor, in the model's own quantisation.

   The SDK does not normalise on the CPU: vpss_preprocessor.cpp hands
   mean/factor to the VPSS engine (stNormalize.mean / .factor), and
   base_model.cpp:122 pre-multiplies both by the input tensor's qscale.
   For the YOLOv8 detector configs factor is 1/255 and mean is 0, so the
   effective per-pixel gain is (1/255) * qscale -- with qscale 126.998978
   that is 0.4980355, mapping pixel 0..255 onto q 0..127. Mean is 0 there,
   so nothing is subtracted first: a -128 offset (an earlier revision had
   one) shifts every pixel half a range off the quantiser's calibration.

   Writing the raw byte instead (as the previous revision did) is wrong by
   2.008x and also wraps every pixel above 127 to a negative int8, which is
   what made this produce zero detections. qscale is read back from the
   model rather than hardcoded so other detector configs still work. */
static void tensor_from_rgb(nv_yolo8_t *y) {
  const int mw = y->in_w, mh = y->in_h;
  const float gain = (y->in_qscale > 0.001f) ? (y->in_qscale / 255.0f) : 0.4980355f;
  for (int p = 0; p < 3; p++) {
    const uint8_t *src = y->in_rgb + p;
    int8_t *dst = (int8_t *)y->in_ptr + (size_t)p * mw * mh;
    for (int i = 0, n = mw * mh; i < n; i++) {
      dst[i] = clamp_q((int)src[(size_t)i * 3] * gain);
    }
  }
}

static void preprocess(nv_yolo8_t *y, const uint8_t *y_plane, const uint8_t *uv_plane,
                       uint32_t stride, int cam_w, int cam_h, int uv_swap) {
  const int mw = y->in_w, mh = y->in_h;
  const int groups = 16; /* box channels / 4 */

  int pad_x, pad_y;
  float kx, ky;
  geom(y, cam_w, cam_h, &pad_x, &pad_y, &kx, &ky);

  /* Letterbox: uniform scale, centred, padded with 128 (neutral after
     normalisation below). */
  int rw = (g_stretch) ? mw : ((int)(cam_w * kx));
  int rh = (g_stretch) ? mh : ((int)(cam_h * kx));
  if (rw < 1) rw = 1;
  if (rh < 1) rh = 1;

  uint8_t *rgb = y->in_rgb;
  memset(rgb, 128, (size_t)mw * mh * 3);

  for (int dy = 0; dy < rh; dy++) {
    int sy = dy * cam_h / rh;
    const uint8_t *yrow = y_plane + (size_t)sy * stride;
    const uint8_t *uvrow = uv_plane + (size_t)(sy / 2) * stride;
    uint8_t *dst = rgb + ((size_t)(dy + pad_y) * mw + pad_x) * 3;
    for (int dx = 0; dx < rw; dx++) {
      int sx = dx * cam_w / rw;
      uint8_t yv = yrow[sx];
      int u = uvrow[(sx >> 1) * 2 + (uv_swap ? 0 : 1)];
      int v = uvrow[(sx >> 1) * 2 + (uv_swap ? 1 : 0)];
      const int d = u - 128;
      const int e = v - 128;
      /* Full-range BT.601 (JFIF) coefficients. The VPSS emits full-range
         NV21: raw dumps of the live sensor show Y down to 15 and up to 255,
         both outside a limited 16..235 clamp, and the model scores match the
         TDL reference pixel-for-pixel under this reading. The first revision
         used the limited-range integer formula with y - 128 (a typo of
         y - 16), which crushed everything below luma 128 to black and cost
         the detector most of its input -- that was the missing boxes. */
      int r = yv + ((91881 * e) >> 16);
      int g = yv - ((22554 * d + 46802 * e) >> 16);
      int b = yv + ((116130 * d) >> 16);
      *dst++ = (uint8_t)(r < 0 ? 0 : (r > 255 ? 255 : r));
      *dst++ = (uint8_t)(g < 0 ? 0 : (g > 255 ? 255 : g));
      *dst++ = (uint8_t)(b < 0 ? 0 : (b > 255 ? 255 : b));
    }
  }

  /* RGB planar into the int8 tensor, in the model's own quantisation. */
  tensor_from_rgb(y);
  (void)groups;


}

/* ---- postprocessing ----------------------------------------------------- */

typedef struct {
  float x1, y1, x2, y2;
  float score;
  int cls;
} det_t;

static int cmp_score(const void *a, const void *b) {
  const det_t *x = (const det_t *)a, *z = (const det_t *)b;
  if (x->score < z->score) return 1;
  if (x->score > z->score) return -1;
  return 0;
}

static float iou(const det_t *a, const det_t *b) {
  float ix1 = a->x1 > b->x1 ? a->x1 : b->x1;
  float iy1 = a->y1 > b->y1 ? a->y1 : b->y1;
  float ix2 = a->x2 < b->x2 ? a->x2 : b->x2;
  float iy2 = a->y2 < b->y2 ? a->y2 : b->y2;
  float iw = ix2 - ix1, ih = iy2 - iy1;
  if (iw <= 0 || ih <= 0) return 0;
  float inter = iw * ih;
  float ua = (a->x2 - a->x1) * (a->y2 - a->y1) + (b->x2 - b->x1) * (b->y2 - b->y1) - inter;
  return inter / (ua > 0 ? ua : 1);
}

/* Class-aware NMS, in place. */
static int nms(det_t *d, int n, float thr) {
  if (n <= 1) return n;
  qsort(d, (size_t)n, sizeof(det_t), cmp_score);
  int keep = 0;
  for (int i = 0; i < n; i++) {
    int drop = 0;
    for (int j = 0; j < keep; j++) {
      if (d[j].cls == d[i].cls && iou(&d[j], &d[i]) > thr) {
        drop = 1;
        break;
      }
    }
    if (!drop) d[keep++] = d[i];
  }
  return keep;
}

/* The 64 box channels are 4 params x 16 bins, i.e. a Distribution Focal Loss
     head, not 16 coordinate pairs. Each group holds the raw logits of one
     distance distribution; the distance is the softmax-weighted mean of the
     bin indices, in units of the cell's stride. Running a sigmoid over them
     instead (as the first cut did) yields boxes that are all roughly the same
     small size near the origin, which is exactly what the drawn output showed:
     dozens of nested rectangles bunched in one corner. */
static float dfl_dist(const int8_t *box, int stride, int idx, int bh, int bw, int gx, int gy,
                      float qb, int zb, int bins) {
  float mx = -1e30f, sum = 0.0f;
  float p[32];
  const int8_t *v = box + (size_t)idx * bins * bh * bw;
  for (int b = 0; b < bins; b++) {
    /* Raw dequantised logits, exactly as the SDK softmaxes them (yolov8.cpp:
       exp(p * qscale)). A sigmoid here compresses the distribution and pulls
       every expected distance to the middle of the bin range. */
    const float q = ((float)v[((size_t)b * bh + gy) * bw + gx] - (float)zb) * qb;
    p[b] = q;
    if (q > mx) mx = q;
  }
  for (int b = 0; b < bins; b++) {
    p[b] = expf(p[b] - mx);
    sum += p[b];
  }
  float d = 0.0f;
  for (int b = 0; b < bins; b++) d += (p[b] / sum) * (float)b;
  return d * (float)stride;
}

/* The same distribution read from an unquantised (fp32) box branch: the raw
   values already are the logits the softmax above dequantises to. */
static float dfl_dist_f(const float *box, int stride, int idx, int bh, int bw, int gx, int gy,
                        int bins) {
  float mx = -1e30f, sum = 0.0f;
  float p[32];
  const float *v = box + (size_t)idx * (size_t)bins * (size_t)bh * (size_t)bw;
  for (int b = 0; b < bins; b++) {
    const float q = v[((size_t)b * bh + gy) * bw + gx];
    p[b] = q;
    if (q > mx) mx = q;
  }
  for (int b = 0; b < bins; b++) {
    p[b] = expf(p[b] - mx);
    sum += p[b];
  }
  float d = 0.0f;
  for (int b = 0; b < bins; b++) d += (p[b] / sum) * (float)b;
  return d * (float)stride;
}

/* Shared tail of both decode paths: cell-relative distances -> one NMS
   candidate (tmp[] in decode() holds 512). */
static int push_det(det_t *tmp, int n, float cx, float cy, float l, float t, float r, float b,
                    float s, int cls, int pad_x, int pad_y, float kx, float ky) {
  if (n < 512) {
    tmp[n].x1 = (cx - l - (float)pad_x) / kx;
    tmp[n].y1 = (cy - t - (float)pad_y) / ky;
    tmp[n].x2 = (cx + r - (float)pad_x) / kx;
    tmp[n].y2 = (cy + b - (float)pad_y) / ky;
    tmp[n].score = s;
    tmp[n].cls = cls;
    n++;
  }
  return n;
}

/* Decode the 3 box + 3 cls branches into frame coordinates and NMS them. */
static int decode(nv_yolo8_t *y, int cam_w, int cam_h, nv_yolo8_det_t *out, int max_out) {
  /* Same mapping preprocess() used, via geom(). The previous revision derived
     this independently and got it subtly wrong even for the letterbox case. */
  int pad_x, pad_y;
  float kx, ky;
  geom(y, cam_w, cam_h, &pad_x, &pad_y, &kx, &ky);

  det_t tmp[512];
  int n = 0;
  /* Per-branch best raw class score, for last_max_score. */
  int bmax_q[3] = {-128, -128, -128};
  float bmax_f[3] = {-1e30f, -1e30f, -1e30f};

  for (int b = 0; b < 3; b++) {
    const int bw = y->out_w[b], bh = y->out_h[b];
    const int box_ch = y->out_ch[b], cls_ch = y->out_ch[b + 3];
    const int groups = box_ch / 4;
    /* No consistency check between the group count and the class count here:
       groups is the number of regression candidates (16 for DFL reg_max),
       cls_ch is however many classes this particular model was exported with
       (7 for person_vehicle). Requiring them to be equal skipped all three
       branches of every 7-class detector, which is why nothing was ever
       found no matter what went into the tensor. */
    const int cells = bw * bh;
    if (cells > y->scratch_cells) continue;
    uint8_t *bestc = y->scratch_c;
    memset(bestc, 0, (size_t)cells);

    if (y->out_fp32[b]) {
      /* The unquantised exports (head_hardhat, fire_smoke, traffic_light):
         the branches hold raw float logits, so the argmax compares floats
         and the sigmoid is computed after the sweep. */
      const float *box = (const float *)y->out_ptr[b];
      const float *cls = (const float *)y->out_ptr[b + 3];
      float *bestf = y->scratch_f;
      for (int i = 0; i < cells; i++) bestf[i] = -1e30f;
      for (int c = 0; c < cls_ch; c++) {
        const float *plane = cls + (size_t)c * cells;
        for (int i = 0; i < cells; i++) {
          if (plane[i] > bestf[i]) {
            bestf[i] = plane[i];
            bestc[i] = (uint8_t)c;
          }
        }
      }
      for (int i = 0; i < cells; i++) {
        const float logit = bestf[i];
        if (logit > bmax_f[b]) bmax_f[b] = logit;
        const float s = 1.0f / (1.0f + expf(-logit));
        if (s < y->threshold) continue;

        const int gx = i % bw, gy = i / bw;
        const int stride = 8 << b;
        /* cell centre in input pixels */
        const float cx = ((float)gx + 0.5f) * (float)stride;
        const float cy = ((float)gy + 0.5f) * (float)stride;
        /* One cell is one box, not one box per DFL group: the 4 distributions
           all belong to the same candidate. */
        const float l = dfl_dist_f(box, stride, 0, bh, bw, gx, gy, groups);
        const float t = dfl_dist_f(box, stride, 1, bh, bw, gx, gy, groups);
        const float r = dfl_dist_f(box, stride, 2, bh, bw, gx, gy, groups);
        const float bb = dfl_dist_f(box, stride, 3, bh, bw, gx, gy, groups);
        n = push_det(tmp, n, cx, cy, l, t, r, bb, s, (int)bestc[i], pad_x, pad_y, kx, ky);
      }
    } else {
      const int8_t *box = y->out_ptr[b];
      const int8_t *cls = y->out_ptr[b + 3];
      const float qb = y->out_q[b];
      const int zb = y->out_zero[b];

      /* Pass 1: the best class per cell, one class plane at a time. The values
         are quantised and sigmoid is monotonic, so the argmax over the raw
         bytes is the argmax over the scores. Sweeping whole planes keeps the
         reads sequential; the old per-cell shape jumped between up to 80 of
         them and ran expf() for every value. */
      int8_t *bestq = y->scratch_q;
      memset(bestq, -128, (size_t)cells);
      for (int c = 0; c < cls_ch; c++) {
        const int8_t *plane = cls + (size_t)c * cells;
        for (int i = 0; i < cells; i++) {
          if (plane[i] > bestq[i]) {
            bestq[i] = plane[i];
            bestc[i] = (uint8_t)c;
          }
        }
      }

      /* Pass 2: threshold via the LUT and DFL-decode the survivors. */
      const float *lut = y->sig_lut[b];
      for (int i = 0; i < cells; i++) {
        const int qv = bestq[i];
        if (qv > bmax_q[b]) bmax_q[b] = qv;
        const float s = lut[qv + 128];
        if (s < y->threshold) continue;

        const int gx = i % bw, gy = i / bw;
        const int stride = 8 << b;
        /* cell centre in input pixels */
        const float cx = ((float)gx + 0.5f) * (float)stride;
        const float cy = ((float)gy + 0.5f) * (float)stride;
        /* One cell is one box, not one box per DFL group: the 4 distributions
           all belong to the same candidate. */
        const float l = dfl_dist(box, stride, 0, bh, bw, gx, gy, qb, zb, groups);
        const float t = dfl_dist(box, stride, 1, bh, bw, gx, gy, qb, zb, groups);
        const float r = dfl_dist(box, stride, 2, bh, bw, gx, gy, qb, zb, groups);
        const float bb = dfl_dist(box, stride, 3, bh, bw, gx, gy, qb, zb, groups);
        n = push_det(tmp, n, cx, cy, l, t, r, bb, s, (int)bestc[i], pad_x, pad_y, kx, ky);
      }
    }
  }

  float gmax = 0.0f;
  for (int b = 0; b < 3; b++) {
    float s;
    if (y->out_fp32[b])
      s = (bmax_f[b] > -1e29f) ? 1.0f / (1.0f + expf(-bmax_f[b])) : 0.0f;
    else
      s = y->sig_lut[b][bmax_q[b] + 128];
    if (s > gmax) gmax = s;
  }

  n = nms(tmp, n, y->nms_iou);
  y->last_max_score = gmax;
  if (n > max_out) n = max_out;
  for (int i = 0; i < n; i++) {
    out[i].x1 = tmp[i].x1;
    out[i].y1 = tmp[i].y1;
    out[i].x2 = tmp[i].x2;
    out[i].y2 = tmp[i].y2;
    out[i].score = tmp[i].score;
    out[i].cls = tmp[i].cls;
  }
  return n;
}

/* ---- the public call --------------------------------------------------- */

/* Submit and decode, with the tensor already filled. Split out so the fill can
   come either from our own preprocess() or, far faster, from a VPSS prep
   channel doing crop/resize/NV12->RGB/normalise in hardware. */
static int forward_and_decode(nv_yolo8_t *y, int cam_w, int cam_h, nv_yolo8_det_t *dets,
                              int max_out, int *n_out) {

  const double t_submit = now_ms();
  void *task = NULL;
  CVI_RC frc = CVI_NN_ForwardAsync(y->model, y->in, y->n_in, y->out, y->n_out, &task);

  if (frc != CVI_RC_SUCCESS) {
    y->stalls++;
    *n_out = 0;
    return -1;
  }

  /* Pollable wait: rc 1 is the "returned" code this runtime actually reports
     on completion; AGAIN means "not ready yet" and is the only reason a
     bounded wait is possible at all. */
  int finished = 0;
  CVI_RC rc = CVI_RC_SUCCESS;
  double due = now_ms() + (double)YOLO8_FORWARD_DEADLINE_MS;
  while (now_ms() < due) {
    rc = CVI_NN_ForwardWait(y->model, task);
    if (rc == CVI_RC_AGAIN) {
      usleep(1000);
      continue;
    }
    if (rc != CVI_RC_SUCCESS && rc != 1) {
      /* The runtime gave up on this forward -- e.g. the blocking ioctl came
         back with EINTR after a watchdog nudge. Same treatment as the
         deadline: drop and re-register, a half-run task poisons the next. */
      y->stalls++;
      y->reloads++;
      CVI_NN_CleanupModel(y->model);
      y->model = NULL;
      *n_out = 0;
      return -2;
    }
    finished = 1;
    break;
  }

  if (!finished) {
    /* Our deadline expired while the driver's own would still be waiting.
       Drop the model and re-register: the measured wedges leave the NPU in a
       state where a fresh registration is what gets it going again. */
    y->stalls++;
    y->reloads++;
    CVI_NN_CleanupModel(y->model);
    y->model = NULL;
    *n_out = 0;
    return -2;
  }

  /* Re-read the output pointers: the runtime is free to rebind the output
     tensors per forward, and reading the pointers cached at open() time would
     then decode a stale or untouched buffer. */
  y->avg_fwd = y->avg_fwd * 0.9 + (now_ms() - t_submit) * 0.1;

  /* Re-read the output pointers: the runtime is free to rebind the output
     tensors per forward, and reading the pointers cached at open() time would
     then decode a stale or untouched buffer. */
  for (int i = 0; i < y->n_out; i++) y->out_ptr[i] = (int8_t *)CVI_NN_TensorPtr(&y->out[i]);

  const double t_dec = now_ms();
  *n_out = decode(y, cam_w, cam_h, dets, max_out);
  y->avg_dec = y->avg_dec * 0.9 + (now_ms() - t_dec) * 0.1;
  return 0;
}

/* Software preprocess: NV12 -> RGB planar -> int8 on the CPU. ~145 ms at
   640x480, which is what capped the fast path at 6 fps. */
int nv_yolo8_infer(nv_yolo8_t *y, const uint8_t *y_plane, const uint8_t *uv_plane, uint32_t stride,
                   int cam_w, int cam_h, int uv_swap, nv_yolo8_det_t *dets, int max_out,
                   int *n_out) {
  const double t0 = now_ms();
  preprocess(y, y_plane, uv_plane, stride, cam_w, cam_h, uv_swap);
  y->avg_prep = y->avg_prep * 0.9 + (now_ms() - t0) * 0.1;
  int rc = forward_and_decode(y, cam_w, cam_h, dets, max_out, n_out);
  y->avg_ms = y->avg_ms * 0.9 + (now_ms() - t0) * 0.1;
  return rc;
}

/* BGR variant for pipelines whose frames are already BGR (e.g. an OpenCV
   capture backend fed by a VPSS channel emitting PIXEL_FORMAT_BGR_888).
   Same letterbox geometry and normalisation as the NV12 path; the channel
   swap to RGB happens while filling the staging buffer. */
static void preprocess_bgr(nv_yolo8_t *y, const uint8_t *src, uint32_t stride, int cam_w,
                           int cam_h) {
  const int mw = y->in_w, mh = y->in_h;

  int pad_x, pad_y;
  float kx, ky;
  geom(y, cam_w, cam_h, &pad_x, &pad_y, &kx, &ky);

  int rw = (g_stretch) ? mw : ((int)(cam_w * kx));
  int rh = (g_stretch) ? mh : ((int)(cam_h * kx));
  if (rw < 1) rw = 1;
  if (rh < 1) rh = 1;

  uint8_t *rgb = y->in_rgb;
  memset(rgb, 128, (size_t)mw * mh * 3);

  for (int dy = 0; dy < rh; dy++) {
    int sy = dy * cam_h / rh;
    const uint8_t *srow = src + (size_t)sy * stride;
    uint8_t *dst = rgb + ((size_t)(dy + pad_y) * mw + pad_x) * 3;
    for (int dx = 0; dx < rw; dx++) {
      int sx = dx * cam_w / rw;
      const uint8_t *px = srow + (size_t)sx * 3;
      dst[0] = px[2]; /* R */
      dst[1] = px[1]; /* G */
      dst[2] = px[0]; /* B */
      dst += 3;
    }
  }

  tensor_from_rgb(y);
}

int nv_yolo8_infer_bgr(nv_yolo8_t *y, const uint8_t *bgr, uint32_t stride, int cam_w, int cam_h,
                       nv_yolo8_det_t *dets, int max_out, int *n_out) {
  const double t0 = now_ms();
  preprocess_bgr(y, bgr, stride, cam_w, cam_h);
  y->avg_prep = y->avg_prep * 0.9 + (now_ms() - t0) * 0.1;
  int rc = forward_and_decode(y, cam_w, cam_h, dets, max_out, n_out);
  y->avg_ms = y->avg_ms * 0.9 + (now_ms() - t0) * 0.1;
  return rc;
}

/* Hardware preprocess: a VPSS channel already emitted
   PIXEL_FORMAT_RGB_888_PLANAR at the model size with stNormalize applied, so
   the three planes drop straight into the tensor. No per-pixel CPU work. */
int nv_yolo8_infer_rgb(nv_yolo8_t *y, const uint8_t *planes, int cam_w, int cam_h,
                       nv_yolo8_det_t *dets, int max_out, int *n_out) {
  const double t0 = now_ms();
  const size_t plane = (size_t)y->in_w * y->in_h;
  for (int p = 0; p < 3; p++)
    memcpy((int8_t *)y->in_ptr + plane * p, planes + plane * p, plane);
  y->avg_prep = y->avg_prep * 0.9 + (now_ms() - t0) * 0.1;
  int rc = forward_and_decode(y, cam_w, cam_h, dets, max_out, n_out);
  y->avg_ms = y->avg_ms * 0.9 + (now_ms() - t0) * 0.1;
  return rc;
}

/* The tensor is already filled -- the VPSS hardware path copies its planes
   into y->in_ptr first -- so only the forward and the decode are timed here.
   prep_ms is that copy's cost, folded into the phase statistics. */
int nv_yolo8_infer_filled(nv_yolo8_t *y, int cam_w, int cam_h, double prep_ms,
                          nv_yolo8_det_t *dets, int max_out, int *n_out) {
  const double t0 = now_ms();
  if (prep_ms >= 0.0) y->avg_prep = y->avg_prep * 0.9 + prep_ms * 0.1;
  int rc = forward_and_decode(y, cam_w, cam_h, dets, max_out, n_out);
  y->avg_ms = y->avg_ms * 0.9 + ((now_ms() - t0) + (prep_ms > 0.0 ? prep_ms : 0.0)) * 0.1;
  return rc;
}
