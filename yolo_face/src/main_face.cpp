/* face_stream — MJPEG stream with face detection and recognition on LicheeRV Nano.
   Standalone application: does not touch stream_yolo/start_stream.sh.
   NV21 capture through the OpenCV-CVI backend, TDL inference (SCRFD + cviface),
   web UI on the same port 7777 (/ — page, /stream — MJPEG). */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <algorithm>
#include <mutex>
#include <vector>
#include <string>
#include <pthread.h>

#include <linux/cvi_comm_video.h>
#include <cvi_sys.h>
#include <opencv2/opencv.hpp>

#include "MJPEGWriter.h"
#include "webui.h"
#include "face_pipeline.h"
#include "vpss_face.h"

extern "C" {
#include "venc_jpeg.h"
#include "draw_nv21.h"
}

/* NV21 colors (Y, U, V). */
static const uint8_t COL_GREEN_Y = 145, COL_GREEN_U = 54, COL_GREEN_V = 34;
static const uint8_t COL_RED_Y = 76, COL_RED_U = 84, COL_RED_V = 255;
static const uint8_t COL_YELLOW_Y = 210, COL_YELLOW_U = 16, COL_YELLOW_V = 146;

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int) { g_stop = 1; }

static double mono_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static float env_float(const char *name, float def) {
    const char *e = getenv(name);
    return (e != NULL && atof(e) != 0.0) ? (float)atof(e) : def;
}

static int env_int(const char *name, int def) {
    const char *e = getenv(name);
    return (e != NULL) ? atoi(e) : def;
}

/* ---------- anti-spoofing (photo / screen) ---------- */
/* The liveness model does not work on this firmware, so spoofing is caught by
   the dynamics of the face area:
   - a real face always "breathes" (sensor noise, blinking, micro-movements) —
     the face area changes from frame to frame;
   - a photo (printed or on a screen) is static: the area does not change at all;
   - a screen flickers: the whole area changes at once and almost uniformly.
   The decision is made from the largest face in the frame. */

#define ASF_GRID 16

struct AntiSpoof {
    float face_prev[ASF_GRID * ASF_GRID];
    float glob_prev[ASF_GRID * ASF_GRID];
    int have;
    int static_run;   /* consecutive frames: face does not change */
    int screen_run;   /* consecutive frames: uniform flicker */
    int spoof;        /* verdict: 1 — spoof */
    float face_diff;
    float glob_diff;
    int active;
    int log_n;
    float period;   /* autocorrelation: high -> regular pattern */
    float chroma;   /* R/B correlation: low -> screen subpixels */
    float hf;       /* high-frequency energy (skin texture) */
    float sat;      /* mean saturation */
    float res;      /* residual after global motion compensation */
    float noise;    /* background noise (camera noise) */
    float gshift;   /* scene shift between frames, px */
};
static AntiSpoof g_asf;
static float asf_per_thr = 0.55f;    /* "pattern" threshold */
static float asf_chroma_thr = 0.75f; /* "screen" threshold */

/* "Pattern" metric: autocorrelation of the high-frequency part of luma at
   non-zero shifts. A screen (pixel grid) and a print (halftone) have a regular
   period — after a high-pass the autocorrelation at that period is high. Live
   skin (noise, pores, hair) has no such regularity: the HF signal is
   uncorrelated. */
#define ASF_AC_W 48

/* High-pass: subtract the local 3x3 mean. */
static void asf_highpass(float *v, int W, int H) {
    float tmp[ASF_AC_W * ASF_AC_W];
    memcpy(tmp, v, sizeof(tmp));
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float sum = 0.0f;
            int n = 0;
            for (int dy = -1; dy <= 1; dy++) {
                const int yy = y + dy;
                if (yy < 0 || yy >= H) continue;
                for (int dx = -1; dx <= 1; dx++) {
                    const int xx = x + dx;
                    if (xx < 0 || xx >= W) continue;
                    sum += tmp[yy * W + xx];
                    n++;
                }
            }
            v[y * W + x] = tmp[y * W + x] - (n ? sum / n : 0.0f);
        }
    }
}

/* HF energy: mean absolute neighbor difference (skin texture vs smooth
   print). */
static float asf_hf(const float *v, int W, int H) {
    double sum = 0.0;
    int n = 0;
    for (int y = 0; y < H; y++) {
        for (int x = 1; x + 1 < W; x++) {
            sum += fabsf(v[y * W + x] - v[y * W + x - 1]);
            n++;
        }
    }
    return (n > 0) ? (float)(sum / n) : 0.0f;
}

/* Mean luma of a rectangular region (for eye contrast). */
static float asf_region_mean(const uint8_t *rgb, uint32_t stride, int w, int h, int x1, int y1,
                             int x2, int y2) {
    if (x1 < 0) x1 = 0;
    if (y1 < 0) y1 = 0;
    if (x2 > w) x2 = w;
    if (y2 > h) y2 = h;
    if (x2 <= x1 || y2 <= y1) return -1.0f;
    double sum = 0.0;
    int n = 0;
    for (int y = y1; y < y2; y++) {
        const uint8_t *row = rgb + (size_t)y * stride;
        for (int x = x1; x < x2; x++) {
            const uint8_t *p = row + (size_t)x * 3;
            sum += 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
            n++;
        }
    }
    return (n > 0) ? (float)(sum / n) : -1.0f;
}

/* Mean saturation of a region (max-min per channel). */
static float asf_saturation(const uint8_t *rgb, uint32_t stride, int w, int h, int x1, int y1, int x2,
                            int y2) {
    double sum = 0.0;
    int n = 0;
    for (int y = y1; y < y2 && y < h; y += 3) {
        if (y < 0) continue;
        const uint8_t *row = rgb + (size_t)y * stride;
        for (int x = x1; x < x2 && x < w; x += 3) {
            if (x < 0) continue;
            const uint8_t *p = row + (size_t)x * 3;
            const int mx = p[0] > p[1] ? (p[0] > p[2] ? p[0] : p[2]) : (p[1] > p[2] ? p[1] : p[2]);
            const int mn = p[0] < p[1] ? (p[0] < p[2] ? p[0] : p[2]) : (p[1] < p[2] ? p[1] : p[2]);
            sum += (double)(mx - mn);
            n++;
        }
    }
    return (n > 0) ? (float)(sum / n / 255.0) : 0.0f;
}

static float asf_periodicity(const float *vin, int W, int H) {
    float v[ASF_AC_W * ASF_AC_W];
    memcpy(v, vin, sizeof(v));
    asf_highpass(v, W, H);
    float best = 0.0f;
    for (int lag = 2; lag <= 12; lag++) {
        double num = 0.0, den = 0.0;
        for (int y = 0; y < H; y++)
            for (int x = 0; x + lag < W; x++) {
                num += (double)v[y * W + x] * v[y * W + x + lag];
                den += (double)v[y * W + x] * v[y * W + x];
            }
        if (den > 1e-6) {
            const float ac = (float)(num / den);
            if (ac > best) best = ac;
        }
        num = 0.0;
        den = 0.0;
        for (int y = 0; y + lag < H; y++)
            for (int x = 0; x < W; x++) {
                num += (double)v[y * W + x] * v[(y + lag) * W + x];
                den += (double)v[y * W + x] * v[y * W + x];
            }
        if (den > 1e-6) {
            const float ac = (float)(num / den);
            if (ac > best) best = ac;
        }
    }
    return best;
}

/* "Screen" metric: on a screen the color subpixels are offset, so at sharp
   edges the R and B signals are out of phase and correlate worse than in a
   natural image (skin). */
static float asf_chroma_corr(const uint8_t *rgb, uint32_t stride, int w, int h, int x1, int y1,
                             int x2, int y2) {
    double sr = 0.0, sb = 0.0, srb = 0.0;
    for (int y = y1 + 1; y < y2 - 1 && y < h - 1; y++) {
        const uint8_t *row = rgb + (size_t)y * stride;
        const uint8_t *rowp = rgb + (size_t)(y - 1) * stride;
        const uint8_t *rown = rgb + (size_t)(y + 1) * stride;
        for (int x = x1 + 1; x < x2 - 1 && x < w - 1; x++) {
            const uint8_t *p = row + (size_t)x * 3;
            const uint8_t *pl = rowp + (size_t)x * 3;
            const uint8_t *pn = rown + (size_t)x * 3;
            const double dr = (double)p[0] - 0.5 * ((double)pl[0] + (double)pn[0]);
            const double db = (double)p[2] - 0.5 * ((double)pl[2] + (double)pn[2]);
            sr += dr * dr;
            sb += db * db;
            srb += dr * db;
        }
    }
    if (sr < 1e-6 || sb < 1e-6) return 1.0f;
    return (float)(srb / sqrt(sr * sb));
}

/* Dense 48x48 luma grid (for the periodicity metric). */
static void asf_grid_dense(const uint8_t *rgb, uint32_t stride, int w, int h, int x1, int y1, int x2,
                           int y2, float *grid, int W, int H) {
    for (int gy = 0; gy < H; gy++) {
        for (int gx = 0; gx < W; gx++) {
            const int px0 = x1 + (x2 - x1) * gx / W;
            const int py0 = y1 + (y2 - y1) * gy / H;
            const int px1 = x1 + (x2 - x1) * (gx + 1) / W;
            const int py1 = y1 + (y2 - y1) * (gy + 1) / H;
            int sw = px1 - px0, sh = py1 - py0;
            if (sw < 1) sw = 1;
            if (sh < 1) sh = 1;
            const int step_x = sw / 2 + 1;
            const int step_y = sh / 2 + 1;
            float sum = 0.0f;
            int n = 0;
            for (int y = py0; y < py1 && y < h; y += step_y) {
                if (y < 0) continue;
                const uint8_t *row = rgb + (size_t)y * stride;
                for (int x = px0; x < px1 && x < w; x += step_x) {
                    if (x < 0) continue;
                    const uint8_t *p = row + (size_t)x * 3;
                    sum += 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
                    n++;
                }
            }
            grid[gy * W + gx] = (n > 0) ? sum / n : 0.0f;
        }
    }
    /* remove the mean — autocorrelation of the DC component is not needed */
    float mean = 0.0f;
    for (int i = 0; i < W * H; i++) mean += grid[i];
    mean /= (float)(W * H);
    for (int i = 0; i < W * H; i++) grid[i] -= mean;
}

/* Mean luma grid of the region (16x16 cells). */
static void asf_grid(const uint8_t *rgb, uint32_t stride, int w, int h, int x1, int y1, int x2,
                     int y2, float *grid) {
    for (int gy = 0; gy < ASF_GRID; gy++) {
        for (int gx = 0; gx < ASF_GRID; gx++) {
            const int px0 = x1 + (x2 - x1) * gx / ASF_GRID;
            const int py0 = y1 + (y2 - y1) * gy / ASF_GRID;
            const int px1 = x1 + (x2 - x1) * (gx + 1) / ASF_GRID;
            const int py1 = y1 + (y2 - y1) * (gy + 1) / ASF_GRID;
            int sw = px1 - px0, sh = py1 - py0;
            if (sw < 1) sw = 1;
            if (sh < 1) sh = 1;
            const int step_x = sw / 3 + 1;
            const int step_y = sh / 3 + 1;
            float sum = 0.0f;
            int n = 0;
            for (int y = py0; y < py1 && y < h; y += step_y) {
                if (y < 0) continue;
                const uint8_t *row = rgb + (size_t)y * stride;
                for (int x = px0; x < px1 && x < w; x += step_x) {
                    if (x < 0) continue;
                    const uint8_t *p = row + (size_t)x * 3;
                    sum += 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
                    n++;
                }
            }
            grid[gy * ASF_GRID + gx] = (n > 0) ? sum / n : 0.0f;
        }
    }
}

static void asf_diff(const float *a, const float *b, float *mean, int *active, float *spread) {
    const int N = ASF_GRID * ASF_GRID;
    float sum = 0.0f, sum2 = 0.0f;
    int act = 0;
    for (int i = 0; i < N; i++) {
        const float d = fabsf(a[i] - b[i]);
        sum += d;
        sum2 += d * d;
        if (d > 4.0f) act++;
    }
    *mean = sum / N;
    const float var = sum2 / N - (*mean) * (*mean);
    *spread = sqrtf(var > 0.0f ? var : 0.0f);
    *active = act;
}

/* IoU of two boxes: a box is considered the same if the overlap is sufficient. */
static float box_iou(const face_box_t &a, const face_box_t &b) {
    const int x1 = a.x1 > b.x1 ? a.x1 : b.x1;
    const int y1 = a.y1 > b.y1 ? a.y1 : b.y1;
    const int x2 = a.x2 < b.x2 ? a.x2 : b.x2;
    const int y2 = a.y2 < b.y2 ? a.y2 : b.y2;
    if (x2 <= x1 || y2 <= y1) return 0.0f;
    const float inter = (float)(x2 - x1) * (float)(y2 - y1);
    const float aa = (float)(a.x2 - a.x1) * (float)(a.y2 - a.y1);
    const float bb = (float)(b.x2 - b.x1) * (float)(b.y2 - b.y1);
    if (aa <= 0.0f || bb <= 0.0f) return 0.0f;
    return inter / (aa + bb - inter);
}

/* ---------- face detection ---------- */
struct FaceState {
    std::mutex m;
    std::vector<face_box_t> boxes;
    double boxes_ms = 0;
    double infer_ms = 0;
    long errors = 0;
    long restarts = 0;
};
static FaceState g_face;

/* Face tracks: stitch together frames where the detector "blinks" (face
   present, then absent), smooth box jitter and keep the box for up to 0.5 s
   after it disappears. Only tracks found by the detector twice get into the
   overlay, so isolated garbage boxes never leak into the stream. */
struct FaceTrack {
    face_box_t box;
    double last_seen;
    int hits;
    int missed;
};
static std::vector<FaceTrack> g_tracks;

static int g_max_boxes = 8;    /* more boxes per frame — detection failure */
static int g_many_boxes = 0;
static int g_garbage_run = 0;  /* consecutive garbage frames */
static int g_pub_max = 5;      /* how many boxes to show at most */
static float g_pub_score = 0.55f;
static int g_pub_min = 48;
static float g_yaw = 0.0f, g_pitch = 0.0f;  /* head pose (from landmarks) */
static float g_face_luma = 0.0f;            /* mean face luma in the frame */
static float g_eye_contrast = 0.0f;         /* "cheek minus eye" contrast */
static float g_lex = 0.0f, g_ley = 0.0f, g_rex = 0.0f, g_rey = 0.0f; /* smoothed eyes */
static int g_eyes_ok = 0;
static face_box_t g_track;                  /* observed face (track) */
static double g_track_t = 0.0;
static int g_track_ok = 0;
static int g_seen_live = 0;                /* a confirmed face was in the frame */
static double g_seen_live_t = 0.0;
static float asf_static_thr = 0.8f;  /* "face does not change" threshold */
static int asf_run_static = 40;      /* this many consecutive frames -> spoof */
static int asf_run_screen = 5;       /* this many flicker frames -> spoof */

/* Detection lives in a separate thread: if the TPU/TDL hangs, the video stream
   must keep going. Errors in a row -> recreate TDL (model+handle). */
static volatile double g_hb = 0.0;   /* detection thread "heartbeat" */

static void *face_worker(void *arg) {
    (void)arg;
    const int period_ms = env_int("SY_FACE_PERIOD_MS", 70);
    int err_run = 0;
    while (!g_stop) {
        g_hb = mono_ms();
        VIDEO_FRAME_INFO_S rgb;
        if (vpss_face_get_rot(&rgb, 300) != 0) continue;
        face_box_t fb[16];
        const double t = mono_ms();
        const int n = face_process(&rgb, fb, 16);
        const double dt = mono_ms() - t;

        /* Watchdog: on a TPU failure (log shows `run cmdbuf failed`) TDL may
           return the same result forever — the boxes "freeze" even though a
           different person is already in the frame. If the set of boxes
           (coordinates+score) has not changed for 5 s, recreate TDL. With a
           live camera the score changes every frame, so there are no false
           triggers. */
        {
            static unsigned long long last_sig = 0;
            static double last_change = 0.0;
            unsigned long long sig = 1469598103934665603ULL;
            for (int i = 0; i < n; i++) {
                const unsigned c[4] = {(unsigned)fb[i].x1, (unsigned)fb[i].y1,
                                       (unsigned)fb[i].x2, (unsigned)fb[i].y2};
                for (int k = 0; k < 4; k++) sig = (sig ^ c[k]) * 1099511628211ULL;
                unsigned sc = 0;
                memcpy(&sc, &fb[i].score, 4);
                sig = (sig ^ sc) * 1099511628211ULL;
            }
            const double now = mono_ms();
            if (sig != last_sig) {
                last_sig = sig;
                last_change = now;
            } else if (n > 0 && now - last_change > 3000.0) {
                printf("face: detection unchanged for %.0f s, recreating TDL\n",
                       (now - last_change) / 1000.0);
                fflush(stdout);
                face_restart();
                last_change = now;
                last_sig = 0;
                g_track_ok = 0;
                g_tracks.clear();
                std::lock_guard<std::mutex> lk(g_face.m);
                g_face.boxes.clear();
                g_face.boxes_ms = 0;
            }
        }

        /* The detector "blinks" on a profile/in motion: a frame with a face, a
           frame without. We stitch detections into tracks by overlap (or
           center proximity), smooth the box and publish only confirmed
           tracks. */
        {
            const double now = mono_ms();
            if (n < 0) {
                /* inference error: leave the last boxes alone — they will
                   disappear by the freshness timeout in the stream. */
            } else if (n > g_max_boxes) {
                if (g_garbage_run < 1000) g_garbage_run++;
                if (!g_many_boxes) {
                    g_many_boxes = 1;
                    printf("face: abnormally many boxes (%d > %d), frame skipped\n", n,
                           g_max_boxes);
                    fflush(stdout);
                }
                /* A series of garbage frames (camera motion, ISP failure):
                   drop both the tracks and the old boxes — otherwise they
                   trail behind the scene. */
                if (g_garbage_run >= 2) {
                    g_tracks.clear();
                    std::lock_guard<std::mutex> lk(g_face.m);
                    g_face.boxes.clear();
                    g_face.boxes_ms = 0;
                }
            } else {
                g_many_boxes = 0;
                g_garbage_run = 0;
                /* Filter out obvious garbage: isolated low-confidence and
                   too-small boxes, as well as boxes of a strange shape. */
                face_box_t vd[16];
                int nd = 0;
                for (int i = 0; i < n && nd < 16; i++) {
                    const face_box_t &d = fb[i];
                    const int bw = d.x2 - d.x1, bh = d.y2 - d.y1;
                    if (d.score < g_pub_score) continue;
                    if (bw < g_pub_min || bh < g_pub_min) continue;
                    if (bw * 2 < bh || bh * 2 < bw) continue;
                    vd[nd++] = d;
                }
                std::vector<int> tr_det(g_tracks.size(), -1);
                std::vector<int> det_tr((size_t)nd, -1);
                for (size_t t = 0; t < g_tracks.size(); t++) {
                    int best_i = -1;
                    float best_v = 0.0f;
                    for (int i = 0; i < nd; i++) {
                        if (det_tr[i] >= 0) continue;
                        float v = box_iou(vd[i], g_tracks[t].box);
                        if (v < 0.15f) {
                            /* camera shift: the same box nearby and of similar size */
                            const int aw = vd[i].x2 - vd[i].x1;
                            const int ah = vd[i].y2 - vd[i].y1;
                            const int bw = g_tracks[t].box.x2 - g_tracks[t].box.x1;
                            const int bh = g_tracks[t].box.y2 - g_tracks[t].box.y1;
                            if (aw > 0 && ah > 0 && bw > 0 && bh > 0) {
                                const float rw = (float)aw / (float)bw;
                                const float rh = (float)ah / (float)bh;
                                if (rw > 0.65f && rw < 1.55f && rh > 0.65f && rh < 1.55f) {
                                    const float dx = 0.5f * (vd[i].x1 + vd[i].x2) -
                                                     0.5f * (g_tracks[t].box.x1 +
                                                             g_tracks[t].box.x2);
                                    const float dy = 0.5f * (vd[i].y1 + vd[i].y2) -
                                                     0.5f * (g_tracks[t].box.y1 +
                                                             g_tracks[t].box.y2);
                                    const float cr = 0.35f * (float)(aw < bw ? aw : bw);
                                    if (dx * dx + dy * dy < cr * cr) v = 0.20f;
                                }
                            }
                        }
                        if (v > best_v) {
                            best_v = v;
                            best_i = i;
                        }
                    }
                    if (best_i >= 0 && best_v > 0.0f) {
                        tr_det[t] = best_i;
                        det_tr[best_i] = (int)t;
                    }
                }
                for (size_t t = 0; t < g_tracks.size(); t++) {
                    if (tr_det[t] < 0) continue;
                    const face_box_t &d = vd[tr_det[t]];
                    face_box_t &b = g_tracks[t].box;
                    /* box smoothing: 3/4 old, 1/4 new */
                    b.x1 = (b.x1 * 3 + d.x1 + 2) / 4;
                    b.y1 = (b.y1 * 3 + d.y1 + 2) / 4;
                    b.x2 = (b.x2 * 3 + d.x2 + 2) / 4;
                    b.y2 = (b.y2 * 3 + d.y2 + 2) / 4;
                    b.score = d.score;
                    b.live = d.live;
                    b.live_ok = d.live_ok;
                    b.spoof = d.spoof;
                    b.have_lm = d.have_lm;
                    if (d.have_lm) memcpy(b.lm, d.lm, sizeof(b.lm));
                    if (d.name[0] != 0) {
                        snprintf(b.name, sizeof(b.name), "%s", d.name);
                        b.sim = d.sim;
                    } else if (d.sim >= 0.0f) {
                        b.sim = d.sim;
                    }
                    g_tracks[t].last_seen = now;
                    g_tracks[t].missed = 0;
                    if (g_tracks[t].hits < 1000000) g_tracks[t].hits++;
                }
                /* a detection miss for 2 frames — the track stops being
                   confirmed (garbage does not accumulate "hits") */
                for (size_t t = 0; t < tr_det.size(); t++) {
                    if (tr_det[t] >= 0) continue;
                    if (++g_tracks[t].missed >= 2) g_tracks[t].hits = 0;
                }
                for (int i = 0; i < nd; i++) {
                    if (det_tr[i] >= 0) continue;
                    FaceTrack t;
                    memset(&t, 0, sizeof(t));
                    t.box = vd[i];
                    t.last_seen = now;
                    t.hits = 1;
                    g_tracks.push_back(t);
                }
                /* a track lives 0.3 s after the last detection */
                for (size_t t = 0; t < g_tracks.size();) {
                    if (now - g_tracks[t].last_seen > 300.0)
                        g_tracks.erase(g_tracks.begin() + (long)t);
                    else
                        t++;
                }
                /* Publish only stable tracks (3 hits). */
                std::vector<face_box_t> pub;
                for (size_t t = 0; t < g_tracks.size(); t++)
                    if (g_tracks[t].hits >= 3) pub.push_back(g_tracks[t].box);
                std::sort(pub.begin(), pub.end(), [](const face_box_t &a, const face_box_t &b) {
                    return (long)(a.x2 - a.x1) * (a.y2 - a.y1) >
                           (long)(b.x2 - b.x1) * (b.y2 - b.y1);
                });
                /* do not publish duplicates of the same box */
                std::vector<face_box_t> uniq;
                for (size_t k = 0; k < pub.size() && (int)uniq.size() < g_pub_max; k++) {
                    bool dup = false;
                    for (size_t q = 0; q < uniq.size(); q++)
                        if (box_iou(pub[k], uniq[q]) > 0.35f) {
                            dup = true;
                            break;
                        }
                    if (!dup) uniq.push_back(pub[k]);
                }
                {
                    std::lock_guard<std::mutex> lk(g_face.m);
                    g_face.boxes = uniq;
                    g_face.boxes_ms = uniq.empty() ? 0.0 : now;
                }
                /* main track (largest) — for liveness and anti-spoofing */
                if (!uniq.empty()) {
                    g_track = uniq[0];
                    g_track_t = now;
                    g_track_ok = 1;
                }
            }
        }
        if (g_track_ok && (mono_ms() - g_track_t) > 800.0) g_track_ok = 0;
        if (!g_track_ok) face_live_update(0, 0, 0, 0, 0.0f, 0.0f, 0.0f, 0);

        /* Anti-spoofing by face dynamics (needs access to frame pixels). */
        if (g_track_ok) {
            uint32_t map_len = 0;
            void *map = vpss_face_map(&rgb, &map_len);
            if (map != NULL) {
                const VIDEO_FRAME_S &vf = rgb.stVFrame;
                /* Channel pool without mapping: read pixels through map */
                const uint8_t *rgbp = (const uint8_t *)map;
                if (rgbp != NULL) {
                    const int w = (int)vf.u32Width, h = (int)vf.u32Height;
                    const uint32_t stride = vf.u32Stride[0];
                    const face_box_t &tface = g_track;
                    float fgrid[ASF_GRID * ASF_GRID], ggrid[ASF_GRID * ASF_GRID];
                    float dgrid[ASF_AC_W * ASF_AC_W];
                    int x1 = tface.x1 < 0 ? 0 : tface.x1;
                    int y1 = tface.y1 < 0 ? 0 : tface.y1;
                    int x2 = tface.x2 >= w ? w - 1 : tface.x2;
                    int y2 = tface.y2 >= h ? h - 1 : tface.y2;
                    asf_grid(rgbp, stride, w, h, x1, y1, x2, y2, fgrid);
                    {
                        float s = 0.0f;
                        for (int k = 0; k < ASF_GRID * ASF_GRID; k++) s += fgrid[k];
                        g_face_luma = s / (float)(ASF_GRID * ASF_GRID);
                    }
                    g_asf.sat = asf_saturation(rgbp, stride, w, h, x1, y1, x2, y2);
                    asf_grid(rgbp, stride, w, h, 0, 0, w - 1, h - 1, ggrid);
                    asf_grid_dense(rgbp, stride, w, h, x1, y1, x2, y2, dgrid, ASF_AC_W, ASF_AC_W);
                    g_asf.hf = asf_hf(dgrid, ASF_AC_W, ASF_AC_W);
                    g_asf.period = asf_periodicity(dgrid, ASF_AC_W, ASF_AC_W);
                    {
                        /* inner part of the face: without hair and background */
                        const int mx = (x2 - x1) / 5, my = (y2 - y1) / 5;
                        const int ix1 = x1 + mx, iy1 = y1 + my;
                        const int ix2 = x2 - mx, iy2 = y2 - my;
                        if (ix2 - ix1 > 8 && iy2 - iy1 > 8)
                            g_asf.chroma = asf_chroma_corr(rgbp, stride, w, h, ix1, iy1, ix2, iy2);
                    }
                    if (!g_asf.have) {
                        memcpy(g_asf.face_prev, fgrid, sizeof(fgrid));
                        memcpy(g_asf.glob_prev, ggrid, sizeof(ggrid));
                        g_asf.have = 1;
                    } else {
                        float fm, gm, fspread, gspread;
                        int fact, gact;
                        asf_diff(fgrid, g_asf.face_prev, &fm, &fact, &fspread);
                        asf_diff(ggrid, g_asf.glob_prev, &gm, &gact, &gspread);
                        g_asf.face_diff = fm;
                        g_asf.glob_diff = gm;
                        g_asf.active = fact;
                        g_asf.noise = gm;
                        /* Global motion compensation: find the scene shift at
                           which the global grid matches best, and apply it to
                           the face. For a flat image the residual is ~= 0; a
                           live face retains its own motion. */
                        {
                            int best_dx = 0, best_dy = 0;
                            float best_err = 1e9f;
                            for (int dy = -3; dy <= 3; dy++) {
                                for (int dx = -3; dx <= 3; dx++) {
                                    float e = 0.0f;
                                    int cnt = 0;
                                    for (int gy = 0; gy < ASF_GRID; gy++) {
                                        const int sy = gy + dy;
                                        if (sy < 0 || sy >= ASF_GRID) continue;
                                        for (int gx = 0; gx < ASF_GRID; gx++) {
                                            const int sx = gx + dx;
                                            if (sx < 0 || sx >= ASF_GRID) continue;
                                            e += fabsf(ggrid[gy * ASF_GRID + gx] -
                                                       g_asf.glob_prev[sy * ASF_GRID + sx]);
                                            cnt++;
                                        }
                                    }
                                    if (cnt > 0) {
                                        e /= (float)cnt;
                                        if (e < best_err) {
                                            best_err = e;
                                            best_dx = dx;
                                            best_dy = dy;
                                        }
                                    }
                                }
                            }
                            g_asf.gshift = (float)(abs(best_dx) + abs(best_dy));
                            float res = 0.0f;
                            int cnt = 0;
                            for (int gy = 0; gy < ASF_GRID; gy++) {
                                const int sy = gy + best_dy;
                                if (sy < 0 || sy >= ASF_GRID) continue;
                                for (int gx = 0; gx < ASF_GRID; gx++) {
                                    const int sx = gx + best_dx;
                                    if (sx < 0 || sx >= ASF_GRID) continue;
                                    res += fabsf(fgrid[gy * ASF_GRID + gx] -
                                                 g_asf.face_prev[sy * ASF_GRID + sx]);
                                    cnt++;
                                }
                            }
                            g_asf.res = (cnt > 0) ? res / (float)cnt : fm;
                        }
                        const int NC = ASF_GRID * ASF_GRID;
                        const bool still = (fm < asf_static_thr);
                        const bool uniform = (fact > NC * 3 / 4) && (fspread < 0.4f * fm + 1.0f);
                        g_asf.static_run = still ? g_asf.static_run + 1 : 0;
                        g_asf.screen_run = uniform ? g_asf.screen_run + 1 : 0;
                        const bool scene_moves = (gm > 2.0f);
                        /* Verdict:
                           - regular pattern (print, screen pixel grid);
                           - "diverged" colors (screen subpixels);
                           - uniform flicker of the whole face (screen);
                           - the face froze while the scene moves (a pasted photo).
                           A live motionless face by itself is not considered
                           a spoof: the sensor is only slightly noisy, the face
                           may not move. */
                        /* Pattern and color fringes differ only slightly
                           between a live face and a photo — we keep them only
                           as metrics in the log. The decision is made by the
                           active check (head turn) plus the "face froze while
                           the scene is alive" and "uniform flicker" signs. */
                        if (uniform || (still && scene_moves))
                            g_asf.spoof = 1;
                        else if (g_asf.static_run == 0 && g_asf.screen_run == 0)
                            g_asf.spoof = 0;
                        memcpy(g_asf.face_prev, fgrid, sizeof(fgrid));
                        memcpy(g_asf.glob_prev, ggrid, sizeof(ggrid));
                    }
                    /* Passive liveness: luma of each eye and the nose bridge.
                       A sharp "brightening" of the eye = blink (the eyelid
                       covered the pupil). Subtracting the nose bridge removes
                       the effect of head rotation and lighting. */
                    {
                        int lvalid = 0;
                        float eye_l = 0.0f, eye_r = 0.0f, lref = 0.0f;
                        if (tface.have_lm) {
                            const float *lm = tface.lm;
                            /* smooth the eye centers: landmark jitter on a
                               small face would otherwise look like blinking */
                            if (!g_eyes_ok || fabsf(lm[0] - g_lex) > 18.0f ||
                                fabsf(lm[1] - g_ley) > 18.0f) {
                                g_lex = lm[0];
                                g_ley = lm[1];
                                g_rex = lm[2];
                                g_rey = lm[3];
                                g_eyes_ok = 1;
                            } else {
                                g_lex = 0.7f * g_lex + 0.3f * lm[0];
                                g_ley = 0.7f * g_ley + 0.3f * lm[1];
                                g_rex = 0.7f * g_rex + 0.3f * lm[2];
                                g_rey = 0.7f * g_rey + 0.3f * lm[3];
                            }
                            const float ddx = g_rex - g_lex;
                            const float ddy = g_rey - g_ley;
                            const float ld = sqrtf(ddx * ddx + ddy * ddy);
                            /* A small face (eyes closer than 26 px) cannot be
                               checked: noise exceeds the signal. Then show only
                               "Unknown", without accusing of spoofing. */
                            if (ld >= 26.0f) {
                                lref = asf_region_mean(rgbp, stride, w, h, (int)(lm[4] - 0.18f * ld),
                                                       (int)(lm[5] - 0.30f * ld),
                                                       (int)(lm[4] + 0.18f * ld),
                                                       (int)(lm[5] + 0.05f * ld));
                                eye_l = asf_region_mean(rgbp, stride, w, h,
                                                        (int)(g_lex - 0.25f * ld),
                                                        (int)(g_ley - 0.20f * ld),
                                                        (int)(g_lex + 0.25f * ld),
                                                        (int)(g_ley + 0.20f * ld));
                                eye_r = asf_region_mean(rgbp, stride, w, h,
                                                        (int)(g_rex - 0.25f * ld),
                                                        (int)(g_rey - 0.20f * ld),
                                                        (int)(g_rex + 0.25f * ld),
                                                        (int)(g_rey + 0.20f * ld));
                                if (lref > 0.0f && eye_l > 0.0f && eye_r > 0.0f) {
                                    g_eye_contrast = lref - 0.5f * (eye_l + eye_r);
                                    /* The eyes must be noticeably darker than
                                       the nose bridge: on a pale print/small
                                       face the signal is below the noise, and
                                       blinking cannot be checked there. */
                                    lvalid = (g_eye_contrast >= 12.0f) ? 1 : 0;
                                }
                            }
                        }
                        face_live_update(tface.x1, tface.y1, tface.x2, tface.y2, eye_l, eye_r,
                                         lref, lvalid);
                    }
                    if (g_asf.spoof) {
                        g_track.spoof = 1;
                        g_track.name[0] = 0;
                        if (g_asf.log_n++ < 8) {
                            printf("face: possible spoof (pattern=%.2f color=%.2f diff=%.2f "
                                   "scene=%.2f)\n",
                                   g_asf.period, g_asf.chroma, g_asf.face_diff, g_asf.glob_diff);
                            fflush(stdout);
                        }
                        face_enroll_abort();
                    }
                }
                vpss_face_unmap(map, map_len);
            }
        }
        vpss_face_release_rot(&rgb);
        {
            std::lock_guard<std::mutex> lk(g_face.m);
            g_face.infer_ms = dt;
            if (n >= 0) {
                err_run = 0;
            } else {
                g_face.errors++;
                err_run++;
            }
        }
        if (n >= 0 && dt > 5000.0) {
            printf("face: inference took %.0f ms, recreating TDL\n", dt);
            fflush(stdout);
            face_restart();
        }
        if (n == -5) {
            /* liveness hung: recreate TDL to bring the TPU back into a working
               state (the liveness model is disabled in this case). */
            printf("face: liveness hung, recreating TDL\n");
            fflush(stdout);
            face_restart();
            err_run = 0;
        }
        if (n < 0) {
            if (err_run == 1 || (err_run % 100) == 0) {
                printf("face: inference error rc=%d (consecutive %d)\n", n, err_run);
                fflush(stdout);
            }
            if (err_run == 30 || err_run == 200) {
                printf("face: recreating TDL after %d consecutive errors\n", err_run);
                fflush(stdout);
                if (face_restart() != 0) printf("face: TDL recreation failed\n");
                std::lock_guard<std::mutex> lk(g_face.m);
                g_face.restarts++;
            }
        }
        usleep((useconds_t)period_ms * 1000);
    }
    return NULL;
}

/* Watchdog: the TPU sometimes hangs so badly that TDL never returns (log shows
   `run dmabuf failed`). The detection thread is stuck at that point and the
   boxes "freeze". If there is no heartbeat for 10 s — exit with code 2, and the
   supervisor in start_face.sh will restart the application. */
static void *face_watchdog(void *arg) {
    (void)arg;
    while (!g_stop) {
        usleep(1000 * 1000);
        const double hb = g_hb;
        if (hb > 0.0 && mono_ms() - hb > 10000.0) {
            printf("face: detection thread unresponsive for %.0f s, restarting\n",
                   (mono_ms() - hb) / 1000.0);
            fflush(stdout);
            _exit(2);
        }
    }
    return NULL;
}

/* Saves a TDL RGB frame as JPEG: verifies what the detector sees and in which
   rotation. */
static void dump_rgb_frame(const VIDEO_FRAME_INFO_S *fi, const char *path) {
    const VIDEO_FRAME_S &f = fi->stVFrame;
    const int w = (int)f.u32Width, h = (int)f.u32Height;
    if (w <= 0 || h <= 0) return;
    const uint64_t base = f.u64PhyAddr[0];
    const uint32_t total = (uint32_t)w * (uint32_t)h * 3u + 4096u;
    uint8_t *v = (uint8_t *)CVI_SYS_MmapCache(base, total);
    if (v == NULL) {
        printf("dump: mmap failed\n");
        return;
    }
    cv::Mat img(h, w, CV_8UC3);
    const bool planar = (f.u64PhyAddr[1] != 0);
    for (int y = 0; y < h; y++) {
        uint8_t *dst = img.ptr<uint8_t>(y);
        if (planar) {
            const size_t off1 = (size_t)(f.u64PhyAddr[1] - base);
            const size_t off2 = (size_t)(f.u64PhyAddr[2] - base);
            const uint32_t s0 = f.u32Stride[0] ? f.u32Stride[0] : (uint32_t)w;
            const uint32_t s1 = f.u32Stride[1] ? f.u32Stride[1] : s0;
            const uint32_t s2 = f.u32Stride[2] ? f.u32Stride[2] : s0;
            memcpy(dst, v + (size_t)y * s0, (size_t)w);
            memcpy(dst + w, v + off1 + (size_t)y * s1, (size_t)w);
            memcpy(dst + 2 * w, v + off2 + (size_t)y * s2, (size_t)w);
        } else {
            memcpy(dst, v + (size_t)y * f.u32Stride[0], (size_t)w * 3);
        }
    }
    cv::imwrite(path, img);
    CVI_SYS_Munmap(v, total);
    printf("dump: %s (%dx%d)\n", path, w, h);
    fflush(stdout);
}

/* Converts box coordinates from the rotated TDL frame to the coordinates of
   the source frame (the one going to MJPEG). cw=1: ROTATION_90 in VPSS — clockwise. */
static void rot_box_to_src(int rot, int cw, int sw, int sh, int *x1, int *y1, int *x2, int *y2) {
    const int a = *x1, b = *y1, c = *x2, d = *y2;
    if (rot == 90) {
        if (cw) {
            *x1 = b; *x2 = d; *y1 = sh - 1 - c; *y2 = sh - 1 - a;
        } else {
            *x1 = sw - 1 - d; *x2 = sw - 1 - b; *y1 = a; *y2 = c;
        }
    } else if (rot == 180) {
        *x1 = sw - 1 - c; *x2 = sw - 1 - a; *y1 = sh - 1 - d; *y2 = sh - 1 - b;
    } else if (rot == 270) {
        if (cw) {
            *x1 = sw - 1 - d; *x2 = sw - 1 - b; *y1 = a; *y2 = c;
        } else {
            *x1 = b; *x2 = d; *y1 = sh - 1 - c; *y2 = sh - 1 - a;
        }
    }
}

/* Cyrillic -> Latin: the built-in overlay font only understands ASCII. */
static const char *cyr_latin(unsigned cp) {
    switch (cp) {
        case 0x410: return "A";  case 0x411: return "B";  case 0x412: return "V";
        case 0x413: return "G";  case 0x414: return "D";  case 0x415: return "E";
        case 0x401: return "Yo"; case 0x416: return "Zh"; case 0x417: return "Z";
        case 0x418: return "I";  case 0x419: return "Y";  case 0x41A: return "K";
        case 0x41B: return "L";  case 0x41C: return "M";  case 0x41D: return "N";
        case 0x41E: return "O";  case 0x41F: return "P";  case 0x420: return "R";
        case 0x421: return "S";  case 0x422: return "T";  case 0x423: return "U";
        case 0x424: return "F";  case 0x425: return "Kh"; case 0x426: return "Ts";
        case 0x427: return "Ch"; case 0x428: return "Sh"; case 0x429: return "Shch";
        case 0x42A: return "";   case 0x42B: return "Y";  case 0x42C: return "";
        case 0x42D: return "E";  case 0x42E: return "Yu"; case 0x42F: return "Ya";
        case 0x430: return "a";  case 0x431: return "b";  case 0x432: return "v";
        case 0x433: return "g";  case 0x434: return "d";  case 0x435: return "e";
        case 0x451: return "yo"; case 0x436: return "zh"; case 0x437: return "z";
        case 0x438: return "i";  case 0x439: return "y";  case 0x43A: return "k";
        case 0x43B: return "l";  case 0x43C: return "m";  case 0x43D: return "n";
        case 0x43E: return "o";  case 0x43F: return "p";  case 0x440: return "r";
        case 0x441: return "s";  case 0x442: return "t";  case 0x443: return "u";
        case 0x444: return "f";  case 0x445: return "kh"; case 0x446: return "ts";
        case 0x447: return "ch"; case 0x448: return "sh"; case 0x449: return "shch";
        case 0x44A: return "";   case 0x44B: return "y";  case 0x44C: return "";
        case 0x44D: return "e";  case 0x44E: return "yu"; case 0x44F: return "ya";
        case 0x406: return "I";  case 0x407: return "Yi"; case 0x404: return "Ye";
        case 0x490: return "G";  case 0x456: return "i";  case 0x457: return "yi";
        case 0x454: return "ye"; case 0x491: return "g";
        default: return NULL;
    }
}

static void translit(const char *in, char *out, size_t len) {
    size_t o = 0;
    const unsigned char *p = (const unsigned char *)in;
    while (*p && o + 8 < len) {
        if (*p < 0x80) {
            out[o++] = (char)*p++;
            continue;
        }
        unsigned cp = 0;
        int n = 0;
        if ((*p & 0xE0) == 0xC0) {
            cp = *p & 0x1F;
            n = 1;
        } else if ((*p & 0xF0) == 0xE0) {
            cp = *p & 0x0F;
            n = 2;
        } else {
            p++;
            continue;
        }
        p++;
        for (int i = 0; i < n && (*p & 0xC0) == 0x80; i++) {
            cp = (cp << 6) | (*p & 0x3F);
            p++;
        }
        const char *lat = cyr_latin(cp);
        if (lat != NULL) {
            while (*lat && o + 1 < len) out[o++] = *lat++;
        } else {
            out[o++] = '?';
        }
    }
    out[o] = 0;
}

int main(int argc, char *argv[]) {
    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    const int cam_w = (argc >= 2) ? atoi(argv[1]) : 960;
    const int cam_h = (argc >= 3) ? atoi(argv[2]) : 540;
    if (cam_w <= 0 || cam_h <= 0) {
        printf("usage: face_stream [width] [height]\n");
        return -1;
    }

    const char *faces_dir = getenv("SY_FACES_DIR");
    if (faces_dir == NULL || faces_dir[0] == 0) faces_dir = "/root/faces";
    const char *fd_model = getenv("SY_FD_MODEL");
    if (fd_model == NULL || fd_model[0] == 0)
        fd_model = "/root/models/face/scrfd_det_face_432_768_INT8_cv181x.cvimodel";
    const char *fr_model = getenv("SY_FR_MODEL");
    if (fr_model == NULL || fr_model[0] == 0)
        fr_model = "/root/models/face/feature_cviface_112_112_INT8_cv181x.cvimodel";
    const char *live_model = getenv("SY_LIVE_MODEL");
    if (live_model == NULL || live_model[0] == 0)
        live_model = "/root/models/face/cls_rgbliveness_256_256_INT8_cv181x.cvimodel";
    /* The liveness model hangs the call on this TDL build (it needs an RGB
       frame), so photo checking is off by default; enable it with SY_LIVE=1. */
    const char *live_env = getenv("SY_LIVE");
    const bool live_off = (live_env == NULL) || (live_env[0] != '1');

    /* 0.40 — profile faces produce a lower score, but at 0.5 they are lost. */
    const float det_thr = env_float("SY_DET_THR", 0.40f);
    const float match_thr = env_float("SY_FACE_THR", 0.40f);
    const float live_thr = env_float("SY_LIVE_THR", 0.50f);
    const int face_every = (env_int("SY_FACE_EVERY", 3) > 0) ? env_int("SY_FACE_EVERY", 3) : 3;
    g_max_boxes = env_int("SY_MAX_BOXES", 8);
    g_pub_max = env_int("SY_PUB_BOXES", 5);
    if (g_pub_max < 1) g_pub_max = 1;
    g_pub_score = env_float("SY_PUB_SCORE", 0.55f);
    g_pub_min = env_int("SY_PUB_MIN", 48);
    const int live_check_ms = env_int("SY_LIVE_CHECK_MS", 12000);
    const int live_keep_ms = env_int("SY_LIVE_KEEP_MS", 15000);
    face_live_timing(live_check_ms, live_keep_ms);
    face_live_blinks(env_int("SY_LIVE_BLINKS", 1));
    asf_static_thr = env_float("SY_ASF_STATIC", 0.8f);
    asf_per_thr = env_float("SY_ASF_PER", 0.55f);
    asf_chroma_thr = env_float("SY_ASF_CHROMA", 0.75f);
    asf_run_static = env_int("SY_ASF_RUN", 40);
    asf_run_screen = env_int("SY_ASF_SCREEN", 5);
    /* The camera in the case is rotated 90°: the frame must be rotated for the detector. */
    const int rot = env_int("SY_ROTATE", 0);
    const int rot_cw = env_int("SY_ROT_CW", 1);

    printf("face_stream: %dx%d, rotation %d, face every %d frames, thr det=%.2f match=%.2f "
           "live=%.2f blink=%d/%dms boxes=%d/%d score=%.2f%s\n",
           cam_w, cam_h, rot, face_every, det_thr, match_thr, live_thr, live_keep_ms,
           live_check_ms, g_pub_max, g_max_boxes, g_pub_score, live_off ? " (off)" : "");
    fflush(stdout);

    /* Capture: the same CVI OpenCV backend as in stream_yolo. The frame arrives
       VB-backed in NV21, and the same frame goes to TDL and the hardware JPEG. */
    setenv("OPENCV_CVI_RAW", "1", 0);
    cv::VideoCapture cap;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, cam_w);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cam_h);
    cap.open(0);
    if (!cap.isOpened()) {
        printf("face_stream: camera failed to open\n");
        return -1;
    }

    cv::Mat dummy;
    for (int i = 0; i < 5 && !g_stop; i++) {
        cap.capture(dummy);
        cap.releaseImagePtr();
    }

    /* Initialize TDL after the camera has raised its VB pools: allocating before
       the camera opens breaks CVI_VI_EnableChn (no buffers). */
    if (face_init(fd_model, fr_model, live_off ? "" : live_model, faces_dir, det_thr, match_thr,
                  live_off ? -1.0f : live_thr) != 0) {
        printf("face_stream: TDL initialization failed\n");
        return -1;
    }

    if (getenv("SY_VPSS_TEST") != NULL) vpss_face_selftest();
    face_video_ready();

    if (venc_jpeg_init(cam_w, cam_h, 75) != 0) {
        printf("face_stream: venc_jpeg init failed\n");
        return -1;
    }
    std::vector<uint8_t> jbuf(2 << 20);

    pthread_t face_thr, wd_thr;
    int face_thr_ok = 0;
    const bool probe_mode = (getenv("SY_FACE_PROBE") != NULL);
    if (vpss_face_open(cam_w, cam_h, rot) != 0) {
        printf("face_stream: VPSS RGB channel failed to open, face detection unavailable\n");
        fflush(stdout);
    } else if (!probe_mode && pthread_create(&face_thr, NULL, face_worker, NULL) == 0) {
        face_thr_ok = 1;
        pthread_create(&wd_thr, NULL, face_watchdog, NULL);
    } else if (!probe_mode) {
        printf("face_stream: detection thread failed to start\n");
        fflush(stdout);
    }
    if (getenv("SY_VPSS_TEST") != NULL) vpss_face_selftest();

    /* On-board diagnostics: several frames through TDL, without streaming. */
    if (getenv("SY_FACE_PROBE") != NULL) {
        for (int i = 0; i < 12 && !g_stop; i++) {
            std::pair<void *, void *> ptrs = cap.capture(dummy);
            VIDEO_FRAME_INFO_S *vf = (VIDEO_FRAME_INFO_S *)ptrs.second;
            if (ptrs.first == nullptr || vf == NULL) {
                cap.releaseImagePtr();
                continue;
            }
            if (i >= 5) {
                face_box_t fb[16];
                VIDEO_FRAME_INFO_S rgb;
                double t = mono_ms();
                int n = -4;
                if (vpss_face_get_rot(&rgb, 500) == 0) {
                    n = face_process(&rgb, fb, 16);
                    const char *dump = getenv("SY_DUMP");
                    if (dump != NULL && dump[0] && i == 5) dump_rgb_frame(&rgb, dump);
                    vpss_face_release_rot(&rgb);
                }
                double dt = mono_ms() - t;
                printf("probe %d: rc=%d %.0fms", i, n, dt);
                for (int k = 0; k < n; k++)
                    printf(" | [%d %d %d %d] score=%.2f sim=%.2f live=%.2f name=%s", fb[k].x1,
                           fb[k].y1, fb[k].x2, fb[k].y2, fb[k].score, fb[k].sim, fb[k].live,
                           fb[k].name);
                printf("\n");
                fflush(stdout);
            }
            cap.releaseImagePtr();
        }
        vpss_face_close();
        venc_jpeg_deinit();
        cap.release();
        face_close();
        return 0;
    }

    MJPEGWriter mjpeg(7777);
    mjpeg.setHttpHandler(web_handle);
    mjpeg.start();
    printf("face_stream: UI http://<ip>:7777/ , stream /stream\n");
    fflush(stdout);

    long frames = 0;
    double t_log = mono_ms();

    while (!g_stop) {
        std::pair<void *, void *> ptrs = cap.capture(dummy);
        VIDEO_FRAME_INFO_S *vf = (VIDEO_FRAME_INFO_S *)ptrs.second;
        if (ptrs.first == nullptr || vf == NULL || vf->stVFrame.pu8VirAddr[0] == NULL) {
            cap.releaseImagePtr();
            continue;
        }
        frames++;

        VIDEO_FRAME_S &f = vf->stVFrame;
        uint8_t *yp = f.pu8VirAddr[0];
        uint8_t *uvp = f.pu8VirAddr[1];
        const uint32_t stride = f.u32Stride[0];
        const int w = (int)f.u32Width;
        const int h = (int)f.u32Height;

        std::vector<face_box_t> local;
        double local_ms;
        double infer_ms = 0;
        long errors = 0;
        {
            std::lock_guard<std::mutex> lk(g_face.m);
            local = g_face.boxes;
            local_ms = g_face.boxes_ms;
            infer_ms = g_face.infer_ms;
            errors = g_face.errors;
        }
        const bool fresh = (mono_ms() - local_ms) < 700.0;   /* do not keep stale boxes */

        const size_t draw_n = (local.size() > (size_t)g_pub_max) ? (size_t)g_pub_max
                                                                 : local.size();
        for (size_t i = 0; fresh && i < draw_n; i++) {
            const face_box_t &b = local[i];
            int x1 = b.x1, y1 = b.y1, x2 = b.x2, y2 = b.y2;
            if (rot != 0) rot_box_to_src(rot, rot_cw, w, h, &x1, &y1, &x2, &y2);
            x1 = x1 < 0 ? 0 : x1;
            y1 = y1 < 0 ? 0 : y1;
            x2 = x2 >= w ? w - 1 : x2;
            y2 = y2 >= h ? h - 1 : y2;
            if (x2 <= x1 || y2 <= y1) continue;

            const bool spoof = (b.spoof != 0) ||
                               (b.live >= 0.0f && live_thr > 0.0f && b.live < live_thr) ||
                               (face_liveness_state() == 2);
            const bool known = (b.name[0] != 0);
            const bool live_ok = (b.live_ok != 0) && !spoof;
            const uint8_t Y = spoof ? COL_RED_Y : (known && live_ok ? COL_GREEN_Y : COL_YELLOW_Y);
            const uint8_t U = spoof ? COL_RED_U : (known && live_ok ? COL_GREEN_U : COL_YELLOW_U);
            const uint8_t V = spoof ? COL_RED_V : (known && live_ok ? COL_GREEN_V : COL_YELLOW_V);

            nv21_draw_rect(yp, uvp, stride, w, h, x1, y1, x2, y2, 3, Y, U, V);

            char label[96];
            char latin[64];
            if (known && !spoof) {
                translit(b.name, latin, sizeof(latin));
                /* Before blinking the name is provisional: "name?" in yellow.
                   After liveness confirmation — a green box and the name
                   without "?". */
                if (live_ok)
                    snprintf(label, sizeof(label), "%s %.2f", latin, b.sim);
                else
                    snprintf(label, sizeof(label), "%s? %.2f", latin, b.sim);
            } else if (spoof) {
                snprintf(label, sizeof(label), "photo?");
            } else if (b.sim >= 0.0f) {
                snprintf(label, sizeof(label), "Unknown %.2f", b.sim);
            } else {
                snprintf(label, sizeof(label), "Unknown");
            }

            const int tscale = 4;
            int twid = (int)strlen(label) * 6 * tscale;
            int tx = x1;
            if (tx + twid > w) tx = w - twid;
            if (tx < 0) tx = 0;
            int ty = y1 - 7 * tscale - 4;
            if (ty < 0) ty = y2 + 4;
            if (ty + 7 * tscale > h) ty = h - 7 * tscale;
            nv21_draw_text(yp, uvp, stride, w, h, tx, ty, label, tscale, Y, U, V);
        }

        CVI_SYS_IonFlushCache(f.u64PhyAddr[0], yp, f.u32Length[0]);
        if (uvp != NULL) {
            uint64_t uv_phy =
                f.u64PhyAddr[1] ? f.u64PhyAddr[1] : f.u64PhyAddr[0] + (uint64_t)stride * h;
            uint32_t uv_len =
                f.u32Length[1] ? f.u32Length[1] : (uint32_t)((size_t)stride * h / 2);
            CVI_SYS_IonFlushCache(uv_phy, uvp, uv_len);
        }

        uint32_t jlen = 0;
        if (venc_jpeg_encode_frame(vf, jbuf.data(), (uint32_t)jbuf.size(), &jlen) == 0) {
            mjpeg.writeJPEG(jbuf.data(), jlen);
            char pname[64];
            if (face_enroll_take_preview(pname, sizeof(pname))) {
                char path[600];
                snprintf(path, sizeof(path), "%s/%s/preview.jpg", faces_dir, pname);
                FILE *pf = fopen(path, "wb");
                if (pf != NULL) {
                    fwrite(jbuf.data(), 1, jlen, pf);
                    fclose(pf);
                    printf("face: preview %s\n", path);
                    fflush(stdout);
                }
            }
        }

        cap.releaseImagePtr();

        if ((frames % 60) == 0) {
            const double now = mono_ms();
            printf("[face] frames=%ld fps=%.1f infer=%.0fms boxes=%zu err=%ld", frames,
                   60000.0 / (now - t_log > 0 ? now - t_log : 1), infer_ms, local.size(), errors);
            for (size_t i = 0; i < local.size() && i < 4; i++) {
                printf(" [%.2f%s]", local[i].score, local[i].name[0] ? local[i].name : "?");
            }
            printf(" asf: d=%.2f res=%.2f nz=%.2f hf=%.2f sat=%.2f per=%.2f ch=%.2f eye=%.1f "
                   "lv=%d %s\n",
                   g_asf.face_diff, g_asf.res, g_asf.noise, g_asf.hf, g_asf.sat, g_asf.period,
                   g_asf.chroma, g_eye_contrast, face_liveness_state(),
                   g_asf.spoof ? "SPOOF" : "");
            fflush(stdout);
            t_log = now;
        }
    }

    printf("face_stream: stopping\n");
    fflush(stdout);
    mjpeg.stop();
    printf("face_stream: MJPEG stream stopped\n");
    fflush(stdout);
    g_stop = 1;
    if (face_thr_ok) pthread_join(face_thr, NULL);
    printf("face_stream: detection thread stopped\n");
    fflush(stdout);
    vpss_face_close();
    printf("face_stream: RGB channel closed\n");
    fflush(stdout);
    venc_jpeg_deinit();
    printf("face_stream: encoder closed\n");
    fflush(stdout);
    cap.release();
    printf("face_stream: camera closed\n");
    fflush(stdout);
    face_close();
    printf("face_stream: TDL closed\n");
    fflush(stdout);
    return 0;
}
