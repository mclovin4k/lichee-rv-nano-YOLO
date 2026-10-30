#ifndef VPSS_FACE_H
#define VPSS_FACE_H

#include <linux/cvi_comm_video.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Additional VPSS channel (group 0, channel 1) with RGB_888 — TDL accepts a
   frame in this format. The channel is opened after the camera has already
   brought up group 0. 0 — success. */
int vpss_face_open(int src_w, int src_h, int rot);

/* Number of a free VPSS group (-1 if it cannot be determined): groups 0..2
   are created by the system on this firmware, TDL must take the next one. */
int vpss_face_available_grp(void);

/* 1 — the group can be created (free), 0 — busy. */
int vpss_face_group_free(int grp);

/* Destroy the group (required after TDL DestroyHandle). 0 — success. */
int vpss_face_destroy_grp(int grp);

/* Stop the group and channel 0 (otherwise DestroyGrp hangs). */
int vpss_face_stop_grp(int grp);

/* Get the freshest frame (0 — success). The frame is held until release. */
int vpss_face_get(VIDEO_FRAME_INFO_S *fi, int timeout_ms);
void vpss_face_release(VIDEO_FRAME_INFO_S *fi);

/* Access frame pixels (mapping into CPU memory). NULL — failed. */
void *vpss_face_map(const VIDEO_FRAME_INFO_S *fi, uint32_t *len);
void vpss_face_unmap(void *p, uint32_t len);

/* Same, but the frame is rotated by SY_ROTATE degrees (in software) — for TDL. */
int vpss_face_get_rot(VIDEO_FRAME_INFO_S *fi, int timeout_ms);
void vpss_face_release_rot(VIDEO_FRAME_INFO_S *fi);

void vpss_face_close(void);

/* Diagnostics (SY_VPSS_TEST): which VPSS groups can be created. */
int vpss_face_selftest(void);

#ifdef __cplusplus
}
#endif

#endif
