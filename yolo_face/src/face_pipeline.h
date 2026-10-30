#ifndef FACE_PIPELINE_H
#define FACE_PIPELINE_H
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int x1, y1, x2, y2;
    float score;      /* detector confidence */
    float sim;        /* cosine to the best gallery match; <0 — no match */
    float live;       /* liveness score; <0 — model disabled */
    int spoof;        /* 1 — looks like a photo/screen: do not show the name */
    int live_ok;      /* 1 — blink confirmed (the box may be drawn green) */
    int have_lm;      /* 1 — landmarks are present */
    float lm[10];     /* 5 points: left eye, right eye, nose, mouth left/right */
    char name[64];    /* gallery name (UTF-8) or empty */
} face_box_t;

/* ---------- passive liveness check (blinking) ---------- */
/* A person blinks naturally, nothing needs to be done. A photo never blinks.
   The confirmation is not permanent: without new blinks the face goes back to
   "checking", so replacing a live face with a photo does not inherit the status. */
void face_live_update(int x1, int y1, int x2, int y2, float eye_l, float eye_r, float ref,
                        int valid);
void face_live_timing(int check_ms, int keep_ms); /* check timings (env) */
void face_live_blinks(int n);                     /* how many blinks confirm liveness */
int face_liveness_ok(void);               /* 1 — a live face has been seen */
int face_liveness_state(void);            /* 0 — checking, 1 — person, 2 — photo */
int face_liveness_face_ok(int x1, int y1, int x2, int y2, float luma); /* same face? */
void face_liveness_reset(void);           /* reset (the face disappeared) */
int face_liveness_json(char *buf, int len);
int face_liveness_prompt(char *buf, int len); /* empty string (no overlay needed) */

/* Initialize TDL and load the gallery. 0 — success. */
/* Recreate TDL (after a TPU hang). 0 — success. */
int face_restart(void);

int face_init(const char *fd_model, const char *fr_model, const char *live_model,
              const char *faces_dir, float det_thr, float match_thr, float live_thr);

/* Inference on a frame (VIDEO_FRAME_INFO_S *). Returns the number of faces or <0. */
int face_process(void *frame_info, face_box_t *out, int max);

/* Called after video start: allocates a dedicated VB pool for TDL. */
int face_video_ready(void);

/* Gallery. */
int face_gallery_reload(void);
int face_gallery_json(char *buf, int len);
int face_enroll_begin(const char *name);
int face_enroll_status(char *buf, int len);
int face_delete(const char *name);

/* Reset enrollment (the face turned out to be a spoof). */
void face_enroll_abort(void);

/* For the preview: take the "enrollment just finished" flag. */
int face_enroll_take_preview(char *name, int len);

void face_close(void);

#ifdef __cplusplus
}
#endif
#endif
