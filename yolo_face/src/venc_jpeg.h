#ifndef NV_VENC_JPEG_H
#define NV_VENC_JPEG_H

#include <stdint.h>
#include <stddef.h>

#include <linux/cvi_comm_video.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Hardware JPEG (VENC PT_JPEG) fed straight from a VPSS frame.

   The encode path never converts or copies the frame on the CPU: the same
   VIDEO_FRAME_INFO_S captured for inference (PIXEL_FORMAT_NV21, VB-backed,
   physical addresses) is handed to the encoder as-is. Recording the frame is
   the capture backend's job; the caller must only flush the CPU caches for
   any region it has drawn into before calling encode_frame().

   init() creates the VENC channel. encode_frame() sends the frame, waits for
   the JPEG bitstream and copies it out; it returns only after VENC has
   consumed the frame, so the caller may release the frame immediately after.
   Returns 0 on success, <0 when the frame was skipped (busy or error) -- a
   skipped frame must never stall the caller. */
int venc_jpeg_init(int width, int height, int quality);
int venc_jpeg_encode_frame(const VIDEO_FRAME_INFO_S *vf, uint8_t *out, uint32_t cap, uint32_t *len);
void venc_jpeg_deinit(void);

#ifdef __cplusplus
}
#endif

#endif
