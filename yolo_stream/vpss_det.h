/* Hardware detection input: a second channel on the capture pipeline's VPSS
   group (group 0, channel 1 -- channel 0 belongs to the OpenCV CVI capture
   backend). The channel scales the sensor frame into the model's letterbox
   rect, converts NV21 -> RGB, normalises (pixel * factor) and quantises to
   int8, all in hardware, so the detector receives the exact tensor the NPU
   input wants without the ~145 ms CPU preprocess.

   vpss_det_open() must be called after the camera is open: the group it
   attaches to is created there. The VPSS functions are reached through
   dlopen, the same way the capture backend loads them, so the build gains no
   new link-time dependency. */
#ifndef VPSS_DET_H
#define VPSS_DET_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* model_w/model_h: NPU input size. qscale: the input tensor's quant scale --
   the per-pixel gain becomes qscale/255, matching the reference TDL VPSS
   preprocessor. src_w/src_h: frame size the detector's boxes are mapped back
   to; the letterbox rect is computed from it exactly as yolo8.c's geom()
   does, so the decode geometry stays identical to the CPU path.
   Returns 0 on success. */
int vpss_det_open(int model_w, int model_h, float qscale, int src_w, int src_h);
void vpss_det_close(void);

/* Waits up to timeout_ms for the frame parked by the channel's own drainer
   thread, copies its three planes (R, G, B) into dst_* and releases the
   frame before returning: the drainer services the channel at frame rate by
   itself, so a slow or stalled detector cannot starve the capture path that
   shares the VPSS group. *prep_ms (optional) receives the copy time.
   Returns 0 on success, -1 on timeout/failure. */
int vpss_det_fetch(uint8_t *dst_r, uint8_t *dst_g, uint8_t *dst_b, size_t dst_bytes,
                   int timeout_ms, double *prep_ms);

#ifdef __cplusplus
}
#endif

#endif
