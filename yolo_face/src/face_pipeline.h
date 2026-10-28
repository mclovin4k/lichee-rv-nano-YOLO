#ifndef FACE_PIPELINE_H
#define FACE_PIPELINE_H
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int x1, y1, x2, y2;
    float score;      /* уверенность детектора */
    float sim;        /* косинус к лучшему в галерее; <0 — нет совпадения */
    float live;       /* liveness score; <0 — модель отключена */
    int spoof;        /* 1 — похоже на фото/экран: имя не показываем */
    int live_ok;      /* 1 — моргание подтверждено (рамку можно рисовать зелёной) */
    int have_lm;      /* 1 — есть ландмарки */
    float lm[10];     /* 5 точек: левый глаз, правый глаз, нос, рот левая/правая */
    char name[64];    /* имя из галереи (UTF-8) или пусто */
} face_box_t;

/* ---------- пассивная проверка живости (моргание) ---------- */
/* Человек моргает сам, ничего делать не нужно. Фотография не моргает.
   Подтверждение не вечное: без новых морганий лицо снова переводится в
   «проверяю», поэтому подмена живого лица фотографией не наследует статус. */
void face_live_update(int x1, int y1, int x2, int y2, float eye_l, float eye_r, float ref,
                        int valid);
void face_live_timing(int check_ms, int keep_ms); /* тайминги проверки (env) */
void face_live_blinks(int n);                     /* сколько морганий подтверждают */
int face_liveness_ok(void);               /* 1 — видели живое лицо */
int face_liveness_state(void);            /* 0 — проверяем, 1 — человек, 2 — фото */
int face_liveness_face_ok(int x1, int y1, int x2, int y2, float luma); /* то же лицо? */
void face_liveness_reset(void);           /* сброс (лицо пропало) */
int face_liveness_json(char *buf, int len);
int face_liveness_prompt(char *buf, int len); /* пустая строка (оверлей не нужен) */

/* Инициализация TDL и загрузка галереи. 0 — успех. */
/* Пересоздать TDL (после зависания TPU). 0 — успех. */
int face_restart(void);

int face_init(const char *fd_model, const char *fr_model, const char *live_model,
              const char *faces_dir, float det_thr, float match_thr, float live_thr);

/* Инференс по кадру (VIDEO_FRAME_INFO_S *). Возвращает число лиц или <0. */
int face_process(void *frame_info, face_box_t *out, int max);

/* Вызывается после старта видео: выделяет TDL собственный VB-пул. */
int face_video_ready(void);

/* Галерея. */
int face_gallery_reload(void);
int face_gallery_json(char *buf, int len);
int face_enroll_begin(const char *name);
int face_enroll_status(char *buf, int len);
int face_delete(const char *name);

/* Сбросить регистрацию (лицо оказалось подделкой). */
void face_enroll_abort(void);

/* Для превью: забрать флаг «регистрация только что завершилась». */
int face_enroll_take_preview(char *name, int len);

void face_close(void);

#ifdef __cplusplus
}
#endif
#endif
