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

/* Tensor writing is fixed: planar NCHW, RGB order, gain from the tensor's own
   qscale, 128 subtracted. The HWC/BGR/gain sweep that found these is over;
   they stay as the only configuration. */
static int g_bias = 128;    /* value subtracted from every channel first */
static int g_stretch = 0;   /* 1 = fill the model rect, distorting aspect */

void nv_yolo8_set_stretch(int stretch) { g_stretch = stretch; }

/* The model metadata pins down the input shape and qscale but not how the
   three planes are laid out in memory nor their order. Both were guessed; a
   wrong guess still yields a plausible-looking tensor and a network that is
   simply and consistently sure it sees nothing, so they are switchable. */


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

static inline float sigmoidf_u8(int32_t q, int32_t zero, float scale) {
  return 1.0f / (1.0f + expf(-((float)q - (float)zero) * scale));
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

  /* The input tensor's own quantisation. preprocess() currently ignores these
     and writes raw 0..255 straight into int8; if the runtime declares a
     non-zero zero point that is wrong and the network sees garbage. */
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
    printf("yolo8:   scale %d: box %dx%d cls %dx%d\n", i, y->out_ch[i], y->out_h[i],
           y->out_w[i + 3], y->out_h[i + 3]);
  }
  return 0;
}

void nv_yolo8_close(nv_yolo8_t *y) {
  if (y->model != NULL) CVI_NN_CleanupModel(y->model);
  free(y->in_rgb);
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
   that is 0.4980355, mapping pixel 0..255 onto q 0..127.

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
      dst[i] = clamp_q(((int)src[(size_t)i * 3] - g_bias) * gain);
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
      int c = yv - 128;
      int d = u - 128;
      int e = v - 128;
      /* BT.601 limited-range, the convention this SDK's YOLO models use */
      int r = (298 * c + 409 * e + 128) >> 8;
      int g = (298 * c - 100 * d - 208 * e + 128) >> 8;
      int b = (298 * c + 516 * d + 128) >> 8;
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
    float q = sigmoidf_u8(v[((size_t)b * bh + gy) * bw + gx], zb, qb);
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

/* Decode the 3 box + 3 cls branches into frame coordinates and NMS them. */
static int decode(nv_yolo8_t *y, int cam_w, int cam_h, nv_yolo8_det_t *out, int max_out) {
  /* Same mapping preprocess() used, via geom(). The previous revision derived
     this independently and got it subtly wrong even for the letterbox case. */
  int pad_x, pad_y;
  float kx, ky;
  geom(y, cam_w, cam_h, &pad_x, &pad_y, &kx, &ky);

  det_t tmp[512];
  int n = 0;
  float gmax = 0.0f;

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
    const int8_t *box = y->out_ptr[b];
    const int8_t *cls = y->out_ptr[b + 3];
    const float qb = y->out_q[b], qc = y->out_q[b + 3];
    const int zb = y->out_zero[b], zc = y->out_zero[b + 3];

    for (int gy = 0; gy < bh; gy++) {
      for (int gx = 0; gx < bw; gx++) {
        int best_cls = -1;
        float best_score = 0;
        for (int c = 0; c < cls_ch; c++) {
          float s = sigmoidf_u8(cls[((size_t)c * bh + gy) * bw + gx], zc, qc);
          if (s > gmax) gmax = s;
          if (s > best_score) {
            best_score = s;
            best_cls = c;
          }
        }
        if (best_cls < 0 || best_score < y->threshold) continue;

        /* One cell is one box, not one box per DFL group: the 4 distributions all
           belong to the same candidate. */
        {
          const int stride = 8 << b;
          /* cell centre in input pixels */
          float cx = ((float)gx + 0.5f) * (float)stride;
          float cy = ((float)gy + 0.5f) * (float)stride;
          float l = dfl_dist(box, stride, 0, bh, bw, gx, gy, qb, zb, groups);
          float t = dfl_dist(box, stride, 1, bh, bw, gx, gy, qb, zb, groups);
          float r = dfl_dist(box, stride, 2, bh, bw, gx, gy, qb, zb, groups);
          float bb = dfl_dist(box, stride, 3, bh, bw, gx, gy, qb, zb, groups);

          /* input pixels -> de-letterbox -> frame pixels */
          float x0 = cx - l;
          float y0 = cy - t;
          float x1 = cx + r;
          float y1 = cy + bb;
          x0 = (x0 - pad_x) / kx; x1 = (x1 - pad_x) / kx;
          y0 = (y0 - pad_y) / ky; y1 = (y1 - pad_y) / ky;
          if (n < (int)(sizeof(tmp) / sizeof(tmp[0]))) {
            tmp[n].x1 = x0; tmp[n].y1 = y0;
            tmp[n].x2 = x1; tmp[n].y2 = y1;
            tmp[n].score = best_score;
            tmp[n].cls = best_cls;
            n++;
          }
        }

      }
    }
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
  for (int i = 0; i < y->n_out; i++) y->out_ptr[i] = (int8_t *)CVI_NN_TensorPtr(&y->out[i]);

  *n_out = decode(y, cam_w, cam_h, dets, max_out);
  return 0;
}

/* Software preprocess: NV12 -> RGB planar -> int8 on the CPU. ~145 ms at
   640x480, which is what capped the fast path at 6 fps. */
int nv_yolo8_infer(nv_yolo8_t *y, const uint8_t *y_plane, const uint8_t *uv_plane, uint32_t stride,
                   int cam_w, int cam_h, int uv_swap, nv_yolo8_det_t *dets, int max_out,
                   int *n_out) {
  double t0 = now_ms();
  preprocess(y, y_plane, uv_plane, stride, cam_w, cam_h, uv_swap);
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
  double t0 = now_ms();
  preprocess_bgr(y, bgr, stride, cam_w, cam_h);
  int rc = forward_and_decode(y, cam_w, cam_h, dets, max_out, n_out);
  y->avg_ms = y->avg_ms * 0.9 + (now_ms() - t0) * 0.1;
  return rc;
}

/* Hardware preprocess: a VPSS channel already emitted
   PIXEL_FORMAT_RGB_888_PLANAR at the model size with stNormalize applied, so
   the three planes drop straight into the tensor. No per-pixel CPU work. */
int nv_yolo8_infer_rgb(nv_yolo8_t *y, const uint8_t *planes, int cam_w, int cam_h,
                       nv_yolo8_det_t *dets, int max_out, int *n_out) {
  double t0 = now_ms();
  const size_t plane = (size_t)y->in_w * y->in_h;
  for (int p = 0; p < 3; p++)
    memcpy((int8_t *)y->in_ptr + plane * p, planes + plane * p, plane);
  int rc = forward_and_decode(y, cam_w, cam_h, dets, max_out, n_out);
  y->avg_ms = y->avg_ms * 0.9 + (now_ms() - t0) * 0.1;
  return rc;
}
