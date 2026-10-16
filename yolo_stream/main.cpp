/* LicheeRV Nano MJPEG stream with YOLOv8 on the NPU.

   The pipeline is the "two frames" scheme the opencv-mobile CVI backend was
   designed for: the VPSS hands out one NV21 frame (VIDEO_FRAME_INFO_S) per
   capture, the NPU preprocess reads that frame's planes directly and the same
   buffer goes to the hardware JPEG encoder. Nothing is converted to BGR and
   no cv::Mat copy exists on the hot path.

   The detector runs on its own thread. The known platform defect (cvi-tpu
   "run dmabuf timeout") blocks a forward for ~60 s once it fires; with the
   split, the capture/encode/HTTP thread keeps streaming at capture rate.
   The stream never shows boxes for a scene that has moved on: results whose
   frame went stale while the driver held the forward are dropped, boxes older
   than SY_BOX_TTL_MS are not drawn ("no det Ns" appears instead), and the
   detector re-registers the model by itself after each stall (see yolo8.c).
   So the wedge costs boxes for a while, not the stream. */

#include "MJPEGWriter.h"
#include "draw_nv21.h"
#include "venc_jpeg.h"
#include "vpss_det.h"
#include "yolo.hpp"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include <cvi_sys.h>
#include <linux/cvi_comm_video.h>

/* Overlay palette: sample_vi_detection draws blue boxes/labels; in full-range
   NV21 that blue is (Y, U, V) = (41, 240, 110). */
#define OVERLAY_Y 41
#define OVERLAY_U 240
#define OVERLAY_V 110

static volatile sig_atomic_t interrupted = 0;

static void interrupt_handler(int signum) {
    (void)signum;
    interrupted = 1;
}

static double mono_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* The platform's TPU driver periodically blocks a forward in an ioctl, and
   neither signals (the wait is not interruptible) nor pulsing the TPU reset
   lines shortens it: the driver's own timeout restores it after ~60 s
   ("run dmabuf timeout" in dmesg), and the detector then re-registers the
   model on its own. The stream thread only watches the silence, reports it,
   and -- when asked via SY_WEDGE_REBOOT=1 -- reboots if a stall outlives the
   usual recovery window by a wide margin. */

/* ---- shared state between the capture/encode thread and the detector ---- */

/* The detector works on a private copy of one NV21 frame (luma plane then
   chroma plane, same stride). The capture thread only drops a copy in while
   the detector is idle, so a slow inference costs frames, never latency. */
struct Stage {
    std::mutex m;
    std::condition_variable cv;
    std::vector<uint8_t> buf;
    uint32_t stride = 0;
    int w = 0, h = 0;
    double t_copy = 0;   /* mono_ms() at the copy: results older than the
                            freshness window are dropped, not published */
    bool fresh = false;  /* a frame is waiting */
    bool busy = false;   /* the detector is running */
    bool quit = false;
};

/* Latest published boxes, written by the detector, read by the capture
   thread once per frame. */
struct BoxSlot {
    std::mutex m;
    std::vector<DetBox> boxes;
    long generation = 0;
    double last_pub_ms = 0;  /* when the boxes were published (0 = never) */
};

int main(int argc, char *argv[]) {
    signal(SIGINT, interrupt_handler);
    signal(SIGTERM, interrupt_handler);

    if (argc < 4) {
        printf("Usage: ./stream_yolo <model_file_path> <class_cnt|auto> <width> [height]\n");
        return 0;
    }

    const char *model_path = argv[1];
    /* "auto" (or a non-number) lets the model's class branches decide how
       many classes it has -- the menu script uses this for every model. */
    const int class_cnt = (strcmp(argv[2], "auto") == 0) ? 0 : atoi(argv[2]);
    /* Stream geometry -- one size for the VPSS frame, the NPU letterbox and
       the box overlay, e.g. "80 960 540" (the reference sample's frame). */
    const int cam_w = atoi(argv[3]);
    const int cam_h = (argc >= 5) ? atoi(argv[4]) : cam_w;
    if (cam_w <= 0 || cam_h <= 0) {
        printf("bad frame size %dx%d\n", cam_w, cam_h);
        return -1;
    }

    /* Offline detector check: run one 640x640 NV21 frame from a file through
       the same infer path the camera feeds, then exit. Used to compare the
       decode against the reference implementation on identical pixels. */
    const char *test_nv21 = getenv("SY_TEST_NV21");
    if (test_nv21 != NULL) {
        const uint32_t tw = (uint32_t)cam_w, th = (uint32_t)cam_h;
        const size_t y_len = (size_t)tw * th;
        FILE *tf = fopen(test_nv21, "rb");
        if (tf == NULL) {
            printf("test: cannot open %s\n", test_nv21);
            return -1;
        }
        std::vector<uint8_t> tbuf(y_len + y_len / 2);
        size_t got = fread(tbuf.data(), 1, tbuf.size(), tf);
        fclose(tf);
        printf("test: read %zu of %zu bytes\n", got, tbuf.size());
        if (got < tbuf.size()) return -1;

        YoloModelDetector tdet;
        if (tdet.setup_model(model_path, class_cnt, 0.5, 0.5) < 0) return -1;
        for (int rep = 0; rep < 3; rep++) {
            std::vector<DetBox> out;
            int rc = tdet.detect_nv21(tbuf.data(), tbuf.data() + y_len, (int)tw, (int)th, tw, out);
            printf("test: rc=%d max=%.3f n=%zu", rc, tdet.last_max_score(), out.size());
            for (size_t i = 0; i < out.size(); i++)
                printf(" | %d:%.3f [%.1f %.1f %.1f %.1f]", out[i].cls, out[i].score, out[i].x1,
                       out[i].y1, out[i].x2, out[i].y2);
            printf("\n");
            fflush(stdout);
        }
        tdet.release();
        return 0;
    }

    /* Raw NV21 capture mode: the VPSS channel emits the sensor frame as NV21
       and the backend maps both planes for us. Must be set before open(),
       the channel format is chosen there. */
    setenv("OPENCV_CVI_RAW", "1", 0);

    cv::VideoCapture cap;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, cam_w);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cam_h);
    cap.open(0);

    if (!cap.isOpened()) {
        printf("Failed to open video capture device.\n");
        return -1;
    }

    /* Score threshold: 0.5 by default; the TDL reference samples run 0.2-0.5,
       a lower threshold also shows weak detections. */
    double det_thr = 0.5;
    {
        const char *e = getenv("SY_DET_THR");
        if (e != NULL && atof(e) > 0.0) det_thr = atof(e);
    }

    YoloModelDetector detector;
    if (detector.setup_model(model_path, class_cnt, det_thr, 0.5) < 0) {
        printf("Failed to setup model %s\n", model_path);
        return -1;
    }

    /* Detection input path. The VPSS can hand the detector a ready int8
       letterboxed tensor (resize, colour conversion and normalisation all in
       hardware), which removes the ~145 ms CPU preprocess that capped the
       detector near 3 fps. The channel is added to the capture pipeline's
       group, so the camera must already be open. SY_DET_VPSS=0 pins the CPU
       path; should the channel stop delivering frames, the detector falls
       back to it by itself at runtime. */
    std::atomic<bool> det_vpss_mode(false);
    {
        const char *e = getenv("SY_DET_VPSS");
        const bool want = !(e != NULL && e[0] == '0');
        if (want && vpss_det_open(detector.input_w(), detector.input_h(), detector.input_qscale(),
                                  cam_w, cam_h) == 0) {
            det_vpss_mode.store(true);
        } else if (want) {
            printf("det: VPSS input unavailable, using the CPU NV21 preprocess\n");
            fflush(stdout);
        }
    }

    // Warmup: let AE/AWB settle, drop first frames (no inference yet).
    cv::Mat dummy;  // the wrapper wants a Mat; in raw mode it is not filled
    for (int i = 0; i < 5 && !interrupted; i++) {
        cap.capture(dummy);
        cap.releaseImagePtr();
    }

    // Hardware JPEG for the stream: cv::imencode on this CPU starves the TPU.
    if (venc_jpeg_init(cam_w, cam_h, 75) != 0) {
        printf("venc_jpeg init failed\n");
        return -1;
    }
    std::vector<uint8_t> jbuf(2 << 20);

    Stage st;
    BoxSlot slot;

    /* Experiment knobs (env): all default to the full pipeline. They exist to
       bisect the platform's TPU wedge against the stream load without a
       rebuild: SY_STREAM_FPS throttles the capture/encode loop, SY_ENC_EVERY
       encodes every N-th frame, SY_DRAW=0 skips drawing, SY_DET=0 skips the
       detector thread entirely. Freshness: SY_DET_MAX_AGE_MS drops results
       whose frame is older than that at publish time (the wedged forward
       returns seconds to minutes late); SY_BOX_TTL_MS hides drawn boxes once
       they are that old (0 = keep drawing them forever). */
    int stream_fps = 0;
    int enc_every = 1;
    int enc_on = 1;
    int draw_on = 1;
    int det_on = 1;
    int wedge_ms = 0;
    int wedge_reboot = 0;
    int det_max_age_ms = 2000;
    int box_ttl_ms = 4000;
    {
        const char *e = getenv("SY_STREAM_FPS");
        if (e != NULL) stream_fps = atoi(e);
        e = getenv("SY_ENC_EVERY");
        if (e != NULL && atoi(e) >= 1) enc_every = atoi(e);
        e = getenv("SY_ENC");
        if (e != NULL && e[0] == '0') enc_on = 0;
        e = getenv("SY_DRAW");
        if (e != NULL && e[0] == '0') draw_on = 0;
        e = getenv("SY_DET");
        if (e != NULL && e[0] == '0') det_on = 0;
        e = getenv("SY_WEDGE_MS");
        if (e != NULL && atoi(e) > 0) wedge_ms = atoi(e);
        e = getenv("SY_WEDGE_REBOOT");
        if (e != NULL && e[0] == '1') wedge_reboot = 1;
        e = getenv("SY_DET_MAX_AGE_MS");
        if (e != NULL && atoi(e) > 0) det_max_age_ms = atoi(e);
        e = getenv("SY_BOX_TTL_MS");
        if (e != NULL && atoi(e) >= 0) box_ttl_ms = atoi(e);
    }
    const double frame_period_ms = (stream_fps > 0) ? 1000.0 / stream_fps : 0.0;
    double next_frame_at = 0;
    printf("knobs: fps=%d enc_every=%d enc=%d draw=%d det=%d wedge_ms=%d wedge_reboot=%d thr=%.2f "
           "det_max_age=%dms box_ttl=%dms\n",
           stream_fps, enc_every, enc_on, draw_on, det_on, wedge_ms, wedge_reboot, det_thr,
           det_max_age_ms, box_ttl_ms);

    std::thread det_thread;
    if (det_on) {
        det_thread = std::thread([&]() {
        long stalls = 0;
        /* One result, one place for what to do with it: reopen after a stall,
           publish fresh boxes, or reject a result the driver held for so long
           that its scene is gone. */
        auto handle = [&](int rc, std::vector<DetBox> &out, double frame_age_ms) {
            if (rc == -2) {
                stalls++;
                printf("detector stall #%ld, reopening model\n", stalls);
                fflush(stdout);
                detector.setup_model(model_path, class_cnt, det_thr, 0.5);
            } else if (rc == 0 && frame_age_ms <= (double)det_max_age_ms) {
                /* Per-detection log line: class, score and box for every
                   published frame; handy for comparing against the TDL
                   reference sample on the same scene. */
                char dbg[512];
                int off = snprintf(dbg, sizeof(dbg), "det: n=%d max=%.3f", (int)out.size(),
                                   detector.last_max_score());
                for (size_t i = 0; i < out.size() && i < 6 && off < (int)sizeof(dbg) - 48; i++)
                    off += snprintf(dbg + off, sizeof(dbg) - (size_t)off,
                                    " | %d:%.2f [%.0f %.0f %.0f %.0f]", out[i].cls, out[i].score,
                                    out[i].x1, out[i].y1, out[i].x2, out[i].y2);
                printf("%s\n", dbg);
                fflush(stdout);
                std::lock_guard<std::mutex> bl(slot.m);
                slot.boxes = out;
                slot.generation++;
                slot.last_pub_ms = mono_ms();
            } else if (rc == 0) {
                /* The driver held the forward through the whole wedge; the
                   frame it returned is seconds to minutes old. Publishing its
                   boxes would draw on the current scene a result of a scene
                   that is gone. */
                printf("det: dropped stale result (frame age %.1fs)\n", frame_age_ms / 1000.0);
                fflush(stdout);
            }
        };

        int vpss_quiet = 0;
        while (true) {
            std::vector<DetBox> out;
            double frame_age_ms = 0;
            int rc;

            if (det_vpss_mode.load()) {
                /* The capture thread drives the group and the VPSS writes the
                   model input; the detector only takes the newest frame. A
                   slow or stalled forward no longer loses the pipeline: the
                   next frame is already waiting. */
                const double t_got = mono_ms();
                rc = detector.detect_vpss(cam_w, cam_h, 200, out);
                if (rc == -3) {
                    /* No frame at all: either a pipeline lull, or the channel
                       is not being serviced. After ~5 s of silence give up on
                       it and go back to the CPU preprocess for good. */
                    std::lock_guard<std::mutex> lk(st.m);
                    if (st.quit) break;
                    if (++vpss_quiet >= 25) {
                        vpss_quiet = 0;
                        printf("det: no VPSS frames for 5s, falling back to the CPU preprocess\n");
                        fflush(stdout);
                        det_vpss_mode.store(false);
                    }
                    continue;
                }
                vpss_quiet = 0;
                frame_age_ms = mono_ms() - t_got;
            } else {
                std::unique_lock<std::mutex> lk(st.m);
                st.cv.wait(lk, [&] { return st.fresh || st.quit; });
                if (st.quit) break;
                st.busy = true;
                st.fresh = false;
                const double t_copy = st.t_copy;
                lk.unlock();

                rc = detector.detect_nv21(st.buf.data(), st.buf.data() + (size_t)st.stride * st.h,
                                          st.w, st.h, st.stride, out);
                frame_age_ms = mono_ms() - t_copy;

                lk.lock();
                st.busy = false;
                lk.unlock();
            }

            handle(rc, out, frame_age_ms);
        }
        });
    }

    MJPEGWriter mjpeg(7777);
    mjpeg.start();
    char clsbuf[32];
    if (class_cnt > 0)
        snprintf(clsbuf, sizeof(clsbuf), "%d classes", class_cnt);
    else
        snprintf(clsbuf, sizeof(clsbuf), "auto classes");
    printf("MJPEG on :7777, model %s (%s), raw NV21 pipeline %dx%d\n", model_path, clsbuf, cam_w,
           cam_h);
    fflush(stdout);

    int fails = 0;
    long frames = 0;
    long last_gen = -1;
    double last_gen_t = 0;
    long last_gen_seen = -1;
    double last_progress_ms = mono_ms();
    int stall_logged = 0;
    double t0 = (double)cv::getTickCount();
    while (!interrupted) {
        std::pair<void *, void *> ptrs = cap.capture(dummy);
        VIDEO_FRAME_INFO_S *vf = (VIDEO_FRAME_INFO_S *)ptrs.second;
        if (ptrs.first == nullptr || vf == NULL || vf->stVFrame.pu8VirAddr[0] == NULL) {
            if (++fails > 30) {
                printf("Too many empty frames, exiting\n");
                break;
            }
            cap.releaseImagePtr();
            continue;
        }
        fails = 0;
        frames++;

        VIDEO_FRAME_S &f = vf->stVFrame;
        uint8_t *yp = f.pu8VirAddr[0];
        uint8_t *uvp = f.pu8VirAddr[1];
        const uint32_t stride = f.u32Stride[0];
        const int w = (int)f.u32Width;
        const int h = (int)f.u32Height;

        // One-shot raw NV21 dump for offline analysis (SY_DUMP_NV21=<path>).
        static const char *dump_path = getenv("SY_DUMP_NV21");
        static bool dumped = false;
        if (dump_path != NULL && !dumped) {
            FILE *df = fopen(dump_path, "wb");
            if (df != NULL) {
                const size_t ylen = (size_t)f.u32Stride[0] * h;
                const size_t uvlen = (size_t)f.u32Stride[1] * (h / 2);
                fwrite(yp, 1, ylen, df);
                if (uvp != NULL) fwrite(uvp, 1, uvlen, df);
                fclose(df);
                printf("dump: wrote %s y=%zu uv=%zu stride=%u/%u w=%d h=%d\n", dump_path, ylen,
                       uvlen, f.u32Stride[0], f.u32Stride[1], w, h);
                fflush(stdout);
            }
            dumped = true;
        }

        // Hand a copy of the frame to the detector, but only when it is idle.
        // (Not in hardware-input mode: the detector reads the VPSS channel.)
        if (!det_vpss_mode.load()) {
            std::unique_lock<std::mutex> lk(st.m, std::try_to_lock);
            if (lk.owns_lock() && !st.busy && !st.fresh) {
                const size_t y_len = (size_t)stride * h;
                if (st.buf.size() < y_len + y_len / 2) st.buf.resize(y_len + y_len / 2);
                uint8_t *dst = st.buf.data();
                memcpy(dst, yp, y_len);
                if (uvp != NULL)
                    memcpy(dst + y_len, uvp, y_len / 2);
                else
                    memset(dst + y_len, 128, y_len / 2);
                st.stride = stride;
                st.w = w;
                st.h = h;
                st.t_copy = mono_ms();
                st.fresh = true;
                st.cv.notify_one();
            }
        }

        // Redraw the latest known boxes: green in NV21 means Y=145, U=54, V=34.
        std::vector<DetBox> boxes;
        {
            std::lock_guard<std::mutex> bl(slot.m);
            boxes = slot.boxes;
            if (slot.generation != last_gen_seen) {
                last_gen_seen = slot.generation;
                last_progress_ms = mono_ms();
            }
        }
        /* Watchdog: the detector publishes a generation per completed forward,
           so a long silence means the TPU driver wedged it -- the platform's
           known defect, which its own driver timeout clears after ~60 s. Just
           report the silence; reboot only if explicitly configured and the
           stall drags on far past that window. */
        const double idle_ms = mono_ms() - last_progress_ms;
        if (det_on && wedge_ms > 0) {
            if (idle_ms <= (double)wedge_ms) {
                stall_logged = 0;
            } else if (!stall_logged) {
                stall_logged = 1;
                printf("watchdog: detector stalled (idle %dms) -- waiting for the driver's recovery\n",
                       (int)idle_ms);
                fflush(stdout);
            } else if (wedge_reboot && idle_ms > (double)wedge_ms * 3.0) {
                printf("watchdog: detector idle %dms, rebooting\n", (int)idle_ms);
                fflush(stdout);
                system("reboot");
                sleep(5);
            }
        }

        if (enc_on && ((frames % enc_every) == 0)) {
            if (draw_on) {
                /* Boxes older than the TTL are not drawn: during a TPU wedge the
                   scene keeps moving while the detector is silent, and a frozen
                   box would point where its object no longer is. The HUD line
                   tells the viewer that the detector, not the stream, waits. */
                const bool boxes_fresh = (box_ttl_ms <= 0) || (idle_ms < (double)box_ttl_ms);
                if (boxes_fresh) {
                    for (size_t i = 0; i < boxes.size(); i++) {
                        const DetBox &d = boxes[i];
                        nv21_draw_rect(yp, uvp, stride, w, h, (int)d.x1, (int)d.y1, (int)d.x2,
                                       (int)d.y2, 3, OVERLAY_Y, OVERLAY_U, OVERLAY_V);
                        char label[64];
                        snprintf(label, sizeof(label), "class:%d score:%.2f", d.cls, d.score);
                        const int tscale = 4;
                        int twid = (int)strlen(label) * 6 * tscale;
                        int tx = (int)d.x1;
                        if (tx + twid > w) tx = w - twid;
                        if (tx < 0) tx = 0;
                        int ty = (int)d.y1 - 7 * tscale - 4;
                        if (ty < 0) ty = (int)d.y2 + 4;
                        if (ty + 7 * tscale > h) ty = h - 7 * tscale;
                        nv21_draw_text(yp, uvp, stride, w, h, tx, ty, label, tscale, OVERLAY_Y, OVERLAY_U,
                                       OVERLAY_V);
                    }
                } else if (det_on) {
                    char hud[48];
                    snprintf(hud, sizeof(hud), "no det %.0fs", idle_ms / 1000.0);
                    nv21_draw_text(yp, uvp, stride, w, h, 8, 34, hud, 4, OVERLAY_Y, OVERLAY_U, OVERLAY_V);
                }

                // The codec reads the physical buffer: make the drawn pixels visible.
                CVI_SYS_IonFlushCache(f.u64PhyAddr[0], yp, f.u32Length[0]);
                if (uvp != NULL) {
                    uint64_t uv_phy = f.u64PhyAddr[1] ? f.u64PhyAddr[1]
                                                      : f.u64PhyAddr[0] + (uint64_t)stride * h;
                    uint32_t uv_len = f.u32Length[1] ? f.u32Length[1]
                                                     : (uint32_t)((size_t)stride * h / 2);
                    CVI_SYS_IonFlushCache(uv_phy, uvp, uv_len);
                }
            }

            // Hardware JPEG straight from the same frame -> MJPEG.
            uint32_t jlen = 0;
            if (venc_jpeg_encode_frame(vf, jbuf.data(), (uint32_t)jbuf.size(), &jlen) == 0)
                mjpeg.writeJPEG(jbuf.data(), jlen);
        }

        cap.releaseImagePtr();

        if ((frames % 30) == 0) {
            double now_t = (double)cv::getTickCount();
            double dt = (now_t - t0) / cv::getTickFrequency();
            double gen_age = -1;
            {
                std::lock_guard<std::mutex> bl(slot.m);
                if (slot.generation != last_gen) {
                    last_gen = slot.generation;
                    last_gen_t = now_t;
                }
                if (last_gen_t > 0) { gen_age = (now_t - last_gen_t) / cv::getTickFrequency(); }
            }
            printf("[stream] frames=%ld fps=%.1f det=%.1fms (prep=%.1f fwd=%.1f dec=%.1f) boxes=%zu "
                   "det_age=%.1fs\n",
                   frames, 30.0 / (dt > 0 ? dt : 1), detector.avg_ms(), detector.prep_ms(),
                   detector.fwd_ms(), detector.dec_ms(), boxes.size(), gen_age);
            fflush(stdout);
            t0 = now_t;
        }

        if (frame_period_ms > 0.0) {
            double t = mono_ms();
            double slack = next_frame_at - t;
            if (slack > 0.5) usleep((useconds_t)(slack * 1000.0));
            next_frame_at = mono_ms() + frame_period_ms;
        }
    }

    printf("Stopping...\n");
    fflush(stdout);

    bool det_stuck;
    {
        std::lock_guard<std::mutex> lk(st.m);
        det_stuck = st.busy;
    }
    if (det_stuck && det_on) {
        /* The forward is blocked in the driver and no signal reaches it:
           joining would hang. Tear the camera pipeline down first (the next
           open reclaims the stale VPSS group), then leave -- the driver
           timeout releases the thread by itself. */
        printf("detector thread stuck in the kernel, exiting without join\n");
        fflush(stdout);
        vpss_det_close();
        cap.release();
        _exit(0);
    }

    {
        std::lock_guard<std::mutex> lk(st.m);
        st.quit = true;
        st.cv.notify_all();
    }
    if (det_thread.joinable()) det_thread.join();

    /* The detection channel goes first: it lives on the capture group, which
       cap.release() tears down. */
    vpss_det_close();

    venc_jpeg_deinit();
    cap.release();
    detector.release();

    return 0;
}
