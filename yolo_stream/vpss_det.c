/* VPSS detection channel -- see vpss_det.h.
   Layout follows the TDL reference (sophgo-tdl vpss_preprocessor.cpp): the
   channel carries the letterbox rect, the model's normalisation factor and
   an int8_c3_planar pixel format, so the NPU input is produced while the
   frame is still streaming through the video pipeline.

   The board's VPSS rejects the full TDL attribute set with ILLEGAL_PARAM on
   this middleware build, so open() walks a table of progressively simpler
   variants and keeps the first one the hardware accepts (logged). For the
   formats whose normalisation is not certain, a multiply-shift converts the
   raw pixels to the tensor's quantisation instead (a 256-entry table would
   be bit-identical but measured ~70x slower on this CPU). */

#include "vpss_det.h"

#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "cvi_sys.h"
#include "cvi_vb.h"
#include "cvi_vpss.h"
#include "linux/cvi_comm_video.h"

/* Channel 0 of this group is the capture backend's NV21 stream channel. */
#define DET_GRP 0
#define DET_CHN VPSS_CHN1

/* The board's VPSS only exists as /mnt/system/{lib,usr/lib}/libvpu.so, which
   the build-time stub does not match one for one; the capture backend already
   reaches it exactly this way, so nothing is linked directly. */
static void *vpss_lib;
static CVI_S32 (*p_SetChnAttr)(VPSS_GRP, VPSS_CHN, const VPSS_CHN_ATTR_S *);
static CVI_S32 (*p_EnableChn)(VPSS_GRP, VPSS_CHN);
static CVI_S32 (*p_DisableChn)(VPSS_GRP, VPSS_CHN);
static CVI_S32 (*p_AttachVbPool)(VPSS_GRP, VPSS_CHN, VB_POOL);
static CVI_S32 (*p_DetachVbPool)(VPSS_GRP, VPSS_CHN);
static CVI_S32 (*p_GetChnFrame)(VPSS_GRP, VPSS_CHN, VIDEO_FRAME_INFO_S *, CVI_S32);
static CVI_S32 (*p_ReleaseChnFrame)(VPSS_GRP, VPSS_CHN, const VIDEO_FRAME_INFO_S *);
static CVI_S32 (*p_SetChnScaleCoefLevel)(VPSS_GRP, VPSS_CHN, VPSS_SCALE_COEF_E); /* optional */
static CVI_S32 (*p_GetGrpAttr)(VPSS_GRP, VPSS_GRP_ATTR_S *);                     /* optional */

static VB_POOL det_pool = VB_INVALID_POOLID;
static int det_open;
static int det_need_lut; /* accepted variant needs the CPU LUT */
static float det_gain = 0.4980355f;
static unsigned det_mul = 32640; /* quant scale in 16.16: q = (v * det_mul + 32768) >> 16 */
static int det_first_frame = 1;

/* MPI fills pu8VirAddr[] only for pools it has mapped into this process; the
   blocks of a user pool attached to a channel stay unmapped (observed: NULL
   pointers). So planes are mapped on demand by physical address, exactly as
   the capture backend does for its raw NV21 frames: CVI_SYS_MmapCache plus
   CVI_SYS_IonInvalidateCache before each read (the frame was just written by
   the VPSS, not by this core), cached because the same few pool blocks
   rotate. */
#define DET_MAP_MAX 12
typedef struct {
  CVI_U64 phy;
  CVI_U32 len;
  uint8_t *va;
} det_map_t;
static det_map_t det_maps[DET_MAP_MAX];
static int det_map_count;
static float det_map_ema, det_conv_ema, det_rel_ema; /* per-fetch ms, EMA */
static long det_fetch_n;

/* The channel is drained by its own thread, not by the detector. The
   detector spends tens of milliseconds inside a forward (and the TPU driver
   can hold one for a minute); with only a couple of pool blocks the channel
   would back the whole VPSS group up, stalling the capture stream that
   shares the group -- observed as the stream dying seconds into every run.
   The drainer keeps the channel serviced at frame rate: it parks the newest
   frame in a one-slot mailbox and releases every other frame at once; the
   detector takes the parked frame (or times out) and releases it after the
   copy. One block of the pool is held between those two moments, at most. */
static pthread_mutex_t det_mx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t det_cv = PTHREAD_COND_INITIALIZER;
static pthread_t det_drain;
static int det_drain_created;
static volatile int det_drain_run;
static VIDEO_FRAME_INFO_S det_slot_fi;
static int det_slot_have;

static uint8_t *det_map_plane(CVI_U64 phy, CVI_U32 len) {
  if (phy == 0 || len == 0) return NULL;
  for (int i = 0; i < det_map_count; i++) {
    if (det_maps[i].phy == phy) {
      CVI_U32 n = (len < det_maps[i].len) ? len : det_maps[i].len;
      CVI_SYS_IonInvalidateCache(phy, det_maps[i].va, n);
      return det_maps[i].va;
    }
  }
  void *p = CVI_SYS_MmapCache(phy, len);
  if (p == NULL) return NULL;
  CVI_SYS_IonInvalidateCache(phy, p, len);
  if (det_map_count < DET_MAP_MAX) {
    det_maps[det_map_count].phy = phy;
    det_maps[det_map_count].len = len;
    det_maps[det_map_count].va = (uint8_t *)p;
    det_map_count++;
  }
  return (uint8_t *)p;
}

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* Drainer loop: never let a finished frame sit on the channel unused. */
static void *det_drain_main(void *arg) {
  (void)arg;
  while (det_drain_run) {
    VIDEO_FRAME_INFO_S fi;
    memset(&fi, 0, sizeof(fi));
    if (p_GetChnFrame(DET_GRP, DET_CHN, &fi, 200) != CVI_SUCCESS) continue;

    pthread_mutex_lock(&det_mx);
    if (det_slot_have) {
      pthread_mutex_unlock(&det_mx);
      p_ReleaseChnFrame(DET_GRP, DET_CHN, &fi); /* the parked one is not taken yet */
      continue;
    }
    det_slot_fi = fi;
    det_slot_have = 1;
    pthread_cond_broadcast(&det_cv);
    pthread_mutex_unlock(&det_mx);
    /* The frame stays held -- and its block unreused -- until a fetch takes
       it. That is one block of the pool, plus one rotating. */
  }
  return NULL;
}

#define LOAD(fp, sym)                                                 \
  do {                                                                \
    *(void **)(&fp) = dlsym(vpss_lib, sym);                           \
    if ((fp) == NULL) {                                               \
      fprintf(stderr, "vpss_det: %s missing in libvpu\n", sym);       \
      return -1;                                                      \
    }                                                                 \
  } while (0)

/* One candidate channel configuration. letterbox=1 keeps the model's
   letterbox rect, 0 fills the whole output. For int8 formats the hardware
   normalisation is what the TDL reference relies on; for uint8/rgb the
   channel may or may not apply it, so the gain LUT is used instead and the
   first frame is logged for verification. */
typedef struct {
  const char *name;
  PIXEL_FORMAT_E fmt;
  int letterbox;
  int norm;
  int bg;
  CVI_U32 depth;
  int need_lut;
} det_variant_t;

static const det_variant_t k_variants[] = {
    /* The configuration the reference preprocessor uses -- rejected here. */
    {"int8 manual norm bg d1", PIXEL_FORMAT_INT8_C3_PLANAR, 1, 1, 1, 1, 0},
    /* Which attribute is it unsatisfied with? Peel them off one by one. */
    {"int8 full norm bg d1", PIXEL_FORMAT_INT8_C3_PLANAR, 0, 1, 1, 1, 0},
    {"int8 manual norm d1", PIXEL_FORMAT_INT8_C3_PLANAR, 1, 1, 0, 1, 0},
    {"int8 manual bg d1", PIXEL_FORMAT_INT8_C3_PLANAR, 1, 0, 1, 1, 0},
    {"int8 manual d1", PIXEL_FORMAT_INT8_C3_PLANAR, 1, 0, 0, 1, 0},
    {"int8 none norm d1", PIXEL_FORMAT_INT8_C3_PLANAR, 0, 1, 0, 1, 0},
    {"int8 none d1", PIXEL_FORMAT_INT8_C3_PLANAR, 0, 0, 0, 1, 0},
    {"int8 manual norm bg d0", PIXEL_FORMAT_INT8_C3_PLANAR, 1, 1, 1, 0, 0},
    /* Fallback formats, to be quantised by the LUT on the CPU. */
    {"uint8 manual norm bg d1", PIXEL_FORMAT_UINT8_C3_PLANAR, 1, 1, 1, 1, 1},
    {"uint8 manual norm d1", PIXEL_FORMAT_UINT8_C3_PLANAR, 1, 1, 0, 1, 1},
    {"uint8 none norm d1", PIXEL_FORMAT_UINT8_C3_PLANAR, 0, 1, 0, 1, 1},
    {"rgbp manual d1", PIXEL_FORMAT_RGB_888_PLANAR, 1, 0, 0, 1, 1},
    {"rgbp none d1", PIXEL_FORMAT_RGB_888_PLANAR, 0, 0, 0, 1, 1},
};

int vpss_det_open(int model_w, int model_h, float qscale, int src_w, int src_h) {
  if (det_open) return 0;
  if (model_w <= 0 || model_h <= 0 || src_w <= 0 || src_h <= 0) return -1;

  static const char *paths[] = {"libvpu.so", "/mnt/system/lib/libvpu.so",
                                "/mnt/system/usr/lib/libvpu.so", NULL};
  for (int i = 0; vpss_lib == NULL && paths[i] != NULL; i++)
    vpss_lib = dlopen(paths[i], RTLD_NOW | RTLD_LOCAL);
  if (vpss_lib == NULL) {
    fprintf(stderr, "vpss_det: libvpu.so not loadable (%s)\n", dlerror());
    return -1;
  }

  LOAD(p_SetChnAttr, "CVI_VPSS_SetChnAttr");
  LOAD(p_EnableChn, "CVI_VPSS_EnableChn");
  LOAD(p_DisableChn, "CVI_VPSS_DisableChn");
  LOAD(p_AttachVbPool, "CVI_VPSS_AttachVbPool");
  LOAD(p_DetachVbPool, "CVI_VPSS_DetachVbPool");
  LOAD(p_GetChnFrame, "CVI_VPSS_GetChnFrame");
  LOAD(p_ReleaseChnFrame, "CVI_VPSS_ReleaseChnFrame");
  *(void **)(&p_SetChnScaleCoefLevel) = dlsym(vpss_lib, "CVI_VPSS_SetChnScaleCoefLevel");
  *(void **)(&p_GetGrpAttr) = dlsym(vpss_lib, "CVI_VPSS_GetGrpAttr");

  det_gain = (qscale > 0.001f) ? (qscale / 255.0f) : 0.4980355f;
  /* raw byte -> tensor unit is round(v * qscale / 255), computed as a 16.16
     multiply. The exact division form makes the compiler emit a real divuw
     per byte on this core (~40 cycles -> 50 ms a frame); mul+add+shift is
     bit-identical to the old float table for every v and runs at memory
     speed. The cap keeps q <= 127 whatever the model's scale is. */
  det_mul = (qscale > 0.001f) ? (unsigned)((qscale * 65536.0f) / 255.0f + 0.5f)
                             : 32640u;
  if (det_mul == 0 || det_mul > 32640u) det_mul = 32640u;

  if (p_GetGrpAttr != NULL) {
    VPSS_GRP_ATTR_S g;
    memset(&g, 0, sizeof(g));
    if (p_GetGrpAttr(DET_GRP, &g) == CVI_SUCCESS)
      printf("vpss_det: group %d input %ux%u fmt %d, src %dx%d\n", DET_GRP,
             (unsigned)g.u32MaxW, (unsigned)g.u32MaxH, (int)g.enPixelFormat, src_w, src_h);
  }

  /* The letterbox rect, computed exactly like yolo8.c's geom(): uniform
     scale, centred. decode() maps boxes back with the same numbers. */
  float scale = (float)model_w / (float)src_w;
  if ((float)model_h / (float)src_h < scale) scale = (float)model_h / (float)src_h;
  int rw = (int)(src_w * scale), rh = (int)(src_h * scale);
  if (rw < 1) rw = 1;
  if (rh < 1) rh = 1;
  const int pad_x = (model_w - rw) / 2, pad_y = (model_h - rh) / 2;

  /* Own VB pool: one block is one frame (3 planes, 1 byte per pixel). Three
     blocks: one parked in the drainer's mailbox, one being written, one in
     hand-off. */
  VB_POOL_CONFIG_S cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.u32BlkSize = (CVI_U32)model_w * (CVI_U32)model_h * 3;
  cfg.u32BlkCnt = 3;
  cfg.enRemapMode = VB_REMAP_MODE_NONE;
  det_pool = CVI_VB_CreatePool(&cfg);
  if (det_pool == VB_INVALID_POOLID) {
    fprintf(stderr, "vpss_det: CVI_VB_CreatePool(%u x2) failed\n", (unsigned)cfg.u32BlkSize);
    return -1;
  }

  const int n_variants = (int)(sizeof(k_variants) / sizeof(k_variants[0]));
  for (int vi = 0; vi < n_variants; vi++) {
    const det_variant_t *v = &k_variants[vi];

    VPSS_CHN_ATTR_S a;
    memset(&a, 0, sizeof(a));
    a.u32Width = (CVI_U32)model_w;
    a.u32Height = (CVI_U32)model_h;
    a.enVideoFormat = VIDEO_FORMAT_LINEAR;
    a.enPixelFormat = v->fmt;
    a.stFrameRate.s32SrcFrameRate = -1;
    a.stFrameRate.s32DstFrameRate = -1;
    a.bMirror = CVI_FALSE;
    a.bFlip = CVI_FALSE;
    a.u32Depth = v->depth;
    if (v->letterbox) {
      a.stAspectRatio.enMode = ASPECT_RATIO_MANUAL;
      a.stAspectRatio.stVideoRect.s32X = pad_x;
      a.stAspectRatio.stVideoRect.s32Y = pad_y;
      a.stAspectRatio.stVideoRect.u32Width = (CVI_U32)rw;
      a.stAspectRatio.stVideoRect.u32Height = (CVI_U32)rh;
    } else {
      a.stAspectRatio.enMode = ASPECT_RATIO_NONE;
    }
    if (v->bg) {
      a.stAspectRatio.bEnableBgColor = CVI_TRUE;
      /* The CPU path pads with mid-grey; same value here. */
      a.stAspectRatio.u32BgColor = 0x808080;
    }
    if (v->norm) {
      a.stNormalize.bEnable = CVI_TRUE;
      for (int i = 0; i < 3; i++) {
        a.stNormalize.factor[i] = det_gain;
        a.stNormalize.mean[i] = 0.0f;
      }
      a.stNormalize.rounding = VPSS_ROUNDING_TO_EVEN; /* TDL's choice */
    }

    CVI_S32 ret = p_SetChnAttr(DET_GRP, DET_CHN, &a);
    printf("vpss_det: variant %d \"%s\": SetChnAttr ret=0x%x\n", vi, v->name, ret);
    fflush(stdout);
    if (ret != CVI_SUCCESS) continue;

    ret = p_EnableChn(DET_GRP, DET_CHN);
    if (ret != CVI_SUCCESS) {
      printf("vpss_det: variant %d EnableChn ret=0x%x\n", vi, ret);
      fflush(stdout);
      continue;
    }
    ret = p_AttachVbPool(DET_GRP, DET_CHN, det_pool);
    if (ret != CVI_SUCCESS) {
      printf("vpss_det: variant %d AttachVbPool ret=0x%x\n", vi, ret);
      fflush(stdout);
      p_DisableChn(DET_GRP, DET_CHN);
      continue;
    }

    if (p_SetChnScaleCoefLevel != NULL)
      p_SetChnScaleCoefLevel(DET_GRP, DET_CHN, VPSS_SCALE_COEF_BILINEAR);

    det_need_lut = v->need_lut; /* static per accepted variant */
    det_drain_run = 1;
    if (pthread_create(&det_drain, NULL, det_drain_main, NULL) != 0) {
      det_drain_run = 0;
      fprintf(stderr, "vpss_det: drain thread failed to start\n");
      p_DetachVbPool(DET_GRP, DET_CHN);
      p_DisableChn(DET_GRP, DET_CHN);
      continue;
    }
    det_drain_created = 1;
    det_open = 1;
    printf("vpss_det: using \"%s\" fmt=%d (rect %d,%d %dx%d, gain %.6f%s)\n", v->name,
           (int)v->fmt, pad_x, pad_y, rw, rh, det_gain, v->need_lut ? ", CPU LUT" : "");
    fflush(stdout);
    return 0;
  }

  fprintf(stderr, "vpss_det: every channel variant was rejected\n");
  CVI_VB_DestroyPool(det_pool);
  det_pool = VB_INVALID_POOLID;
  return -1;
}

void vpss_det_close(void) {
  if (!det_open) return;
  /* Stop the drainer first: it is the channel's last regular user. */
  pthread_mutex_lock(&det_mx);
  det_drain_run = 0;
  pthread_cond_broadcast(&det_cv);
  pthread_mutex_unlock(&det_mx);
  if (det_drain_created) {
    pthread_join(det_drain, NULL);
    det_drain_created = 0;
  }
  pthread_mutex_lock(&det_mx);
  if (det_slot_have) {
    p_ReleaseChnFrame(DET_GRP, DET_CHN, &det_slot_fi);
    det_slot_have = 0;
  }
  pthread_mutex_unlock(&det_mx);
  p_DisableChn(DET_GRP, DET_CHN);
  p_DetachVbPool(DET_GRP, DET_CHN);
  for (int i = 0; i < det_map_count; i++) CVI_SYS_Munmap(det_maps[i].va, det_maps[i].len);
  det_map_count = 0;
  if (det_pool != VB_INVALID_POOLID) {
    CVI_VB_DestroyPool(det_pool);
    det_pool = VB_INVALID_POOLID;
  }
  det_open = 0;
  /* The library stays loaded: the capture backend holds its own handle. */
}

int vpss_det_fetch(uint8_t *dst_r, uint8_t *dst_g, uint8_t *dst_b, size_t dst_bytes,
                   int timeout_ms, double *prep_ms) {
  if (!det_open) return -1;

  /* Take the frame parked by the drainer, at most one frame interval old.
     Waiting holds no VPSS resources, so a detector stuck in the NPU for a
     minute costs nothing on the capture side. */
  VIDEO_FRAME_INFO_S fi;
  memset(&fi, 0, sizeof(fi));
  int got = 0;
  {
    struct timespec dl;
    clock_gettime(CLOCK_REALTIME, &dl);
    dl.tv_sec += timeout_ms / 1000;
    long ns = dl.tv_nsec + (long)(timeout_ms % 1000) * 1000000L;
    if (ns >= 1000000000L) {
      dl.tv_sec += 1;
      ns -= 1000000000L;
    }
    dl.tv_nsec = ns;
    pthread_mutex_lock(&det_mx);
    while (!det_slot_have && det_drain_run) {
      if (pthread_cond_timedwait(&det_cv, &det_mx, &dl) == ETIMEDOUT) break;
    }
    if (det_slot_have) {
      fi = det_slot_fi;
      det_slot_have = 0;
      got = 1;
    }
    pthread_mutex_unlock(&det_mx);
  }
  if (!got) return -1;

  const double t0 = now_ms();
  uint8_t *dst[3] = {dst_r, dst_g, dst_b};
  const VIDEO_FRAME_S *vf = &fi.stVFrame;
  const int w = (int)vf->u32Width;
  int rows = (int)vf->u32Height;
  if (w <= 0 || rows <= 0 || (size_t)w > dst_bytes) {
    p_ReleaseChnFrame(DET_GRP, DET_CHN, &fi);
    return -1;
  }
  if ((size_t)w * (size_t)rows > dst_bytes) rows = (int)(dst_bytes / (size_t)w);

  const int want_stats = det_first_frame;
  int vmax = 0, vmin = 255, ok = 0;
  double t_map = 0.0, t_conv = 0.0;

  /* The three planes of a frame sit back to back in its block, so the whole
     block is mapped and invalidated with one call when that holds. */
  CVI_U64 ph[3];
  CVI_U32 pl[3];
  for (int p = 0; p < 3; p++) {
    ph[p] = vf->u64PhyAddr[p];
    CVI_U32 len = vf->u32Length[p];
    if (len == 0)
      len = vf->u32Stride[p] ? vf->u32Stride[p] * (CVI_U32)rows
                             : (CVI_U32)w * (CVI_U32)rows;
    pl[p] = len;
  }
  uint8_t *base = NULL;
  if (ph[0] != 0 && ph[1] == ph[0] + pl[0] && ph[2] == ph[1] + pl[1]) {
    double s = now_ms();
    base = det_map_plane(ph[0], pl[0] + pl[1] + pl[2]);
    t_map += now_ms() - s;
  }

  for (int p = 0; p < 3; p++) {
    const uint8_t *src;
    if (base != NULL) {
      src = base + (size_t)(ph[p] - ph[0]);
    } else {
      src = (const uint8_t *)vf->pu8VirAddr[p];
      if (src == NULL) {
        double s = now_ms();
        src = det_map_plane(ph[p], pl[p]);
        t_map += now_ms() - s;
      }
    }
    if (src == NULL || dst[p] == NULL) continue;
    ok++;
    uint32_t stride = vf->u32Stride[p];
    if (stride == 0) stride = vf->u32Stride[0];
    if (stride < (uint32_t)w) stride = (uint32_t)w;
    double s = now_ms();
    for (int y = 0; y < rows; y++) {
      const uint8_t *srow = src + (size_t)y * stride;
      uint8_t *drow = dst[p] + (size_t)y * (size_t)w;
      if (det_need_lut) {
        if (want_stats) {
          for (int x = 0; x < w; x++) {
            uint8_t b = srow[x];
            if (b > vmax) vmax = b;
            if (b < vmin) vmin = b;
            drow[x] = (uint8_t)(((unsigned)b * det_mul + 32768u) >> 16);
          }
        } else {
          for (int x = 0; x < w; x++)
            drow[x] = (uint8_t)(((unsigned)srow[x] * det_mul + 32768u) >> 16);
        }
      } else {
        memcpy(drow, srow, (size_t)w);
        if (want_stats) {
          for (int x = 0; x < w; x++) {
            uint8_t b = drow[x];
            if (b > vmax) vmax = b;
            if (b < vmin) vmin = b;
          }
        }
      }
    }
    t_conv += now_ms() - s;
  }

  double s = now_ms();
  p_ReleaseChnFrame(DET_GRP, DET_CHN, &fi);
  double t_rel = now_ms() - s;

  det_map_ema += 0.05f * ((float)t_map - det_map_ema);
  det_conv_ema += 0.05f * ((float)t_conv - det_conv_ema);
  det_rel_ema += 0.05f * ((float)t_rel - det_rel_ema);
  if ((++det_fetch_n % 300) == 0) {
    printf("vpss_det: fetch timing map=%.1f conv=%.1f rel=%.1f ms (n=%ld)\n", det_map_ema,
           det_conv_ema, det_rel_ema, det_fetch_n);
    fflush(stdout);
  }

  /* One line about the very first frame: which addresses were mapped, the
     byte range (<= 127 would mean the channel normalised) and the geometry. */
  if (det_first_frame) {
    det_first_frame = 0;
    printf("vpss_det: first frame %dx%d strides %u/%u/%u phy %llx/%llx/%llx len %u/%u/%u "
           "vir0 %p planes_ok %d range %d..%d%s\n",
           w, rows, (unsigned)vf->u32Stride[0], (unsigned)vf->u32Stride[1],
           (unsigned)vf->u32Stride[2], (unsigned long long)vf->u64PhyAddr[0],
           (unsigned long long)vf->u64PhyAddr[1], (unsigned long long)vf->u64PhyAddr[2],
           (unsigned)vf->u32Length[0], (unsigned)vf->u32Length[1], (unsigned)vf->u32Length[2],
           (void *)vf->pu8VirAddr[0], ok, vmin, vmax, det_need_lut ? " (LUT)" : "");
    fflush(stdout);
  }

  if (prep_ms != NULL) *prep_ms = now_ms() - t0;
  return 0;
}
