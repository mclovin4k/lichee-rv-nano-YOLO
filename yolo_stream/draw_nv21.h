#ifndef NV_DRAW_NV21_H
#define NV_DRAW_NV21_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Box and text helpers for an NV21 frame given as its luma plane and its
   interleaved chroma plane (may be NULL to draw luma only). stride is the
   row stride of both planes. Coordinates are clamped to the frame. */
void nv21_draw_rect(uint8_t *yp, uint8_t *uvp, uint32_t stride, int w, int h, int x0, int y0,
                    int x1, int y1, int thickness, uint8_t Y, uint8_t U, uint8_t V);
void nv21_draw_text(uint8_t *yp, uint8_t *uvp, uint32_t stride, int w, int h, int x, int y,
                    const char *text, int scale, uint8_t Y, uint8_t U, uint8_t V);

#ifdef __cplusplus
}
#endif

#endif
