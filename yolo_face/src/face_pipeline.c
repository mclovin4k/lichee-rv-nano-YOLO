/* Распознавание лиц на cv181x через TDL SDK (SCRFD + cviface + RGB liveness).
   Галерея: <faces_dir>/<имя>/feature.bin  (заголовок "CVF1", uint32 dim, float[dim]). */
#include "face_pipeline.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>
#include <dlfcn.h>
#include <linux/cvi_comm_video.h>
#include "cvi_vb.h"
#include "cvi_tdl.h"
#include "vpss_face.h"

#define MAX_PEOPLE 64
#define MAX_DIM 1024

/* TDL грузим через dlopen: libcvi_tdl.so собрана с shared OpenCV 3.2, а
   приложение — со статическим OpenCV 4.13 (CVI-бэкенд камеры). Прямая
   линковка тянет скрытые символы статической сборки в DSO и падает. */
typedef CVI_S32 (*tdl_create2_t)(cvitdl_handle_t *, const VPSS_GRP, const CVI_U8);
typedef CVI_S32 (*tdl_open_t)(cvitdl_handle_t, CVI_TDL_SUPPORTED_MODEL_E, const char *);
typedef CVI_S32 (*tdl_thr_t)(cvitdl_handle_t, CVI_TDL_SUPPORTED_MODEL_E, float);
typedef CVI_S32 (*tdl_scrfd_t)(const cvitdl_handle_t, VIDEO_FRAME_INFO_S *, cvtdl_face_t *);
typedef CVI_S32 (*tdl_det_t)(const cvitdl_handle_t, VIDEO_FRAME_INFO_S *, CVI_TDL_SUPPORTED_MODEL_E,
                             cvtdl_face_t *);
typedef CVI_S32 (*tdl_rec_t)(const cvitdl_handle_t, VIDEO_FRAME_INFO_S *, cvtdl_face_t *);
typedef CVI_S32 (*tdl_live_t)(const cvitdl_handle_t, VIDEO_FRAME_INFO_S *, VIDEO_FRAME_INFO_S *,
                              cvtdl_face_t *, cvtdl_face_t *);
typedef CVI_S32 (*tdl_free_face_t)(cvtdl_face_t *);
typedef CVI_S32 (*tdl_destroy_t)(cvitdl_handle_t);
typedef CVI_S32 (*tdl_vbpool_t)(cvitdl_handle_t, uint32_t, VB_POOL);
typedef CVI_S32 (*tdl_timeout_t)(cvitdl_handle_t, uint32_t);

static void *g_tdl_lib = NULL;
static tdl_create2_t p_create2 = NULL;
static tdl_open_t p_open = NULL;
static tdl_thr_t p_set_thr = NULL;
static tdl_scrfd_t p_scrfd = NULL;
static tdl_det_t p_det = NULL;
static tdl_rec_t p_rec = NULL;
static tdl_live_t p_live = NULL;
static tdl_free_face_t p_free_face = NULL;
static tdl_destroy_t p_destroy = NULL;
static tdl_vbpool_t p_set_vbpool = NULL;
static tdl_timeout_t p_set_timeout = NULL;
static VB_POOL g_tdl_pool = VB_INVALID_POOLID;

/* libvpu поднимаем в глобальную область до libcvi_tdl.so: TDL использует
   GDC-символы, которых нет в NEEDED.  В поставке рядом лежит патченная
   middleware-сборка libvpu_tdl.so; VPSS при этом продолжает работать с
   системной libvpu.so.  Путь можно переопределить через SY_VPU_LIB.
   RTLD_LAZY: у libvpu есть необязательные ISP-зависимости. */
static void vpu_preload(void) {
    const char *env = getenv("SY_VPU_LIB");
    if (env != NULL && env[0] != '\0' &&
        dlopen(env, RTLD_LAZY | RTLD_GLOBAL) != NULL)
        return;
    if (dlopen("libvpu_tdl.so", RTLD_LAZY | RTLD_GLOBAL) != NULL) return;
    if (dlopen("libvpu.so", RTLD_LAZY | RTLD_GLOBAL) != NULL) return;
    dlopen("/mnt/system/usr/lib/libvpu.so", RTLD_LAZY | RTLD_GLOBAL);
}

static int tdl_load(void) {
    if (g_tdl_lib != NULL) return 0;
    vpu_preload();

    const char *env = getenv("SY_TDL_LIB");
    const char *path = (env != NULL && env[0]) ? env : "libcvi_tdl.so";
    g_tdl_lib = dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
    if (g_tdl_lib == NULL && path != NULL && path[0] != '\0') {
        dlerror();
        g_tdl_lib = dlopen("libcvi_tdl.so", RTLD_LAZY | RTLD_GLOBAL);
    }
    if (g_tdl_lib == NULL) {
        printf("faces: не смог загрузить libcvi_tdl.so: %s\n", dlerror());
        return -1;
    }
#define TDL_SYM(var, name)                                  \
    do {                                                    \
        *(void **)(&var) = dlsym(g_tdl_lib, name);          \
        if (var == NULL) {                                  \
            printf("faces: в libcvi_tdl.so нет %s\n", name); \
            return -1;                                      \
        }                                                   \
    } while (0)
    TDL_SYM(p_create2, "CVI_TDL_CreateHandle2");
    TDL_SYM(p_open, "CVI_TDL_OpenModel");
    TDL_SYM(p_set_thr, "CVI_TDL_SetModelThreshold");
    /* Две сборки TDL: старая (ScrFDFace) и новая (FaceDetection). */
    *(void **)(&p_scrfd) = dlsym(g_tdl_lib, "CVI_TDL_ScrFDFace");
    if (p_scrfd == NULL) TDL_SYM(p_det, "CVI_TDL_FaceDetection");
    TDL_SYM(p_rec, "CVI_TDL_FaceRecognition");
    *(void **)(&p_live) = dlsym(g_tdl_lib, "CVI_TDL_Liveness"); /* может отсутствовать */
    TDL_SYM(p_free_face, "CVI_TDL_FreeFace");
    TDL_SYM(p_destroy, "CVI_TDL_DestroyHandle");
    *(void **)(&p_set_vbpool) = dlsym(g_tdl_lib, "CVI_TDL_SetVBPool");
    *(void **)(&p_set_timeout) = dlsym(g_tdl_lib, "CVI_TDL_SetVpssTimeout");
#undef TDL_SYM
    printf("faces: TDL %s загружен\n", path);
    fflush(stdout);
    return 0;
}

typedef struct {
    char name[64];
    float feat[MAX_DIM];
    int dim;
} person_t;

static cvitdl_handle_t g_tdl = NULL;
static person_t g_people[MAX_PEOPLE];
static int g_npeople = 0;
static char g_dir[256] = "/root/faces";
static float g_match_thr = 0.40f;
static float g_live_thr = -1.0f;   /* <0 — liveness не проверяем */
static int g_live_on = 0;

/* Параметры инициализации — чтобы пересоздать TDL после зависания TPU. */
static VPSS_GRP g_tdl_grp = 0;
static char g_fd_model[256] = "";
static char g_fr_model[256] = "";
static char g_live_model[256] = "";
static float g_det_thr = 0.5f;

enum { EN_IDLE = 0, EN_COLLECT, EN_DONE, EN_ERR };
static int g_en_state = EN_IDLE;
static char g_en_name[64];
static char g_en_msg[128];
static int g_en_need = 12;
static int g_en_got = 0;
static int g_en_dim = 0;
static float g_en_sum[MAX_DIM];
static double g_en_t0 = 0;
static int g_en_preview = 0;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* ---------- имя/пути ---------- */
static int name_ok(const char *s) {
    if (s == NULL || s[0] == '\0' || strlen(s) >= sizeof(((person_t *)0)->name)) return 0;
    if (strstr(s, "..") != NULL) return 0;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (*p == '/' || *p == '\\' || *p < 0x20) return 0;
    }
    return 1;
}

static void person_path(const char *name, char *buf, size_t len) {
    snprintf(buf, len, "%s/%s", g_dir, name);
}

/* ---------- пассивная проверка живости ---------- */
/* Человек моргает сам: при моргании веко на 1-2 кадра прикрывает зрачок, и
   яркость глаза относительно переносицы резко меняется. Смотрим производную
   «глаз минус переносица» — она слабо зависит от поворота головы и освещения.
   Фотография не моргает никогда. Порог изменения адаптивный: оцениваем шум
   кадров, поэтому дрожание фото в руке не превращается в «моргание». Итог
   проверки не вечен: без новых морганий статус «протухает» (keep), иначе
   подмена живого лица фотографией наследовала бы зелёную рамку. */
#define LIVE_CHECK 0
#define LIVE_OK 1
#define LIVE_PHOTO 2

static int g_lv_state = LIVE_CHECK;
static int g_lv_box[4] = {0, 0, 0, 0};
static double g_lv_t0 = 0.0;          /* начало текущей проверки */
static double g_lv_seen_t = 0.0;      /* когда лицо последний раз было в кадре */
static double g_lv_last_blink = 0.0;  /* время последнего принятого моргания */
static int g_lv_blinks = 0;           /* морганий в текущей проверке */
static float g_lv_eye_prev[2] = {0.0f, 0.0f};
static float g_lv_ref_prev = 0.0f;
static int g_lv_prev_ok = 0;
static double g_lv_valid_t = 0.0;     /* последний кадр с валидными глазами */
static double g_lv_update_t = 0.0;    /* прошлое обновление (для dt) */
static double g_lv_valid_ms = 0.0;    /* накопленное время с валидными глазами */
static float g_lv_noise = 1.0f;       /* оценка шума |dv| */
static int g_lv_close = 0;            /* кадров с закрытием глаза */
static int g_lv_open = 0;             /* было закрытие, ждём открытия */
static float g_lv_peak = 0.0f;        /* максимум dv в текущем закрытии */
static char g_lv_msg[160] = "ждём лицо в кадре";
static int g_lv_dbg = 0;

/* Тайминги: check — сколько валидных миллисекунд ждать моргания (таймер
   перезапускается после каждого моргания), keep — сколько живёт подтверждение
   без новых морганий. */
static int g_lv_check_ms = 12000;
static int g_lv_keep_ms = 15000;
static int g_lv_blinks_need = 1;   /* достаточно одного уверенного моргания */

void face_live_timing(int check_ms, int keep_ms) {
    if (check_ms > 0) g_lv_check_ms = check_ms;
    if (keep_ms > 0) g_lv_keep_ms = keep_ms;
}

void face_live_blinks(int n) {
    if (n >= 1 && n <= 10) g_lv_blinks_need = n;
}

static int lv_iou(int ax1, int ay1, int ax2, int ay2, int bx1, int by1, int bx2, int by2) {
    const int x1 = ax1 > bx1 ? ax1 : bx1;
    const int y1 = ay1 > by1 ? ay1 : by1;
    const int x2 = ax2 < bx2 ? ax2 : bx2;
    const int y2 = ay2 < by2 ? ay2 : by2;
    if (x2 <= x1 || y2 <= y1) return 0;
    const int inter = (x2 - x1) * (y2 - y1);
    const int aa = (ax2 - ax1) * (ay2 - ay1);
    const int bb = (bx2 - bx1) * (by2 - by1);
    if (aa <= 0 || bb <= 0) return 0;
    return (inter * 100) / (aa + bb - inter);
}

static int lv_same_face(int x1, int y1, int x2, int y2) {
    if (g_lv_box[2] <= g_lv_box[0]) return 0;
    if (lv_iou(x1, y1, x2, y2, g_lv_box[0], g_lv_box[1], g_lv_box[2], g_lv_box[3]) >= 10) return 1;
    const float cx = (x1 + x2) * 0.5f, cy = (y1 + y2) * 0.5f;
    const float ox = (g_lv_box[0] + g_lv_box[2]) * 0.5f;
    const float oy = (g_lv_box[1] + g_lv_box[3]) * 0.5f;
    const float fw = (x2 - x1) > 0 ? (x2 - x1) : 1.0f;
    const float ddx = cx - ox, ddy = cy - oy;
    return (ddx * ddx + ddy * ddy) < 2.25f * fw * fw;
}

static void lv_clear_prev(void) {
    g_lv_prev_ok = 0;
    g_lv_close = 0;
    g_lv_open = 0;
    g_lv_peak = 0.0f;
}

static void lv_start(int x1, int y1, int x2, int y2) {
    g_lv_state = LIVE_CHECK;
    g_lv_blinks = 0;
    lv_clear_prev();
    g_lv_t0 = now_ms();
    g_lv_valid_t = now_ms();
    g_lv_update_t = now_ms();
    g_lv_valid_ms = 0.0;
    g_lv_last_blink = 0.0;
    g_lv_box[0] = x1;
    g_lv_box[1] = y1;
    g_lv_box[2] = x2;
    g_lv_box[3] = y2;
    snprintf(g_lv_msg, sizeof(g_lv_msg), "моргните: проверяю лицо");
}

void face_live_update(int x1, int y1, int x2, int y2, float eye_l, float eye_r, float ref,
                      int valid) {
    static int dbg_init = 0;
    if (!dbg_init) {
        g_lv_dbg = (getenv("SY_LIVE_DEBUG") != NULL);
        dbg_init = 1;
    }
    const double now = now_ms();

    /* лицо пропало из кадра */
    if (x2 <= x1 || y2 <= y1) {
        if (g_lv_state != LIVE_CHECK && now - g_lv_seen_t > 3000.0) {
            g_lv_state = LIVE_CHECK;
            g_lv_blinks = 0;
            g_lv_valid_ms = 0.0;
            lv_clear_prev();
            snprintf(g_lv_msg, sizeof(g_lv_msg), "ждём лицо в кадре");
        }
        return;
    }

    if (!lv_same_face(x1, y1, x2, y2)) {
        lv_start(x1, y1, x2, y2);
    } else if (g_lv_box[2] > g_lv_box[0]) {
        /* Плавный дрейф и изменение масштаба не должны сбрасывать проверку:
           подтягиваем опорную рамку к текущей. */
        g_lv_box[0] = (g_lv_box[0] * 3 + x1 + 2) / 4;
        g_lv_box[1] = (g_lv_box[1] * 3 + y1 + 2) / 4;
        g_lv_box[2] = (g_lv_box[2] * 3 + x2 + 2) / 4;
        g_lv_box[3] = (g_lv_box[3] * 3 + y2 + 2) / 4;
    }
    g_lv_seen_t = now;

    /* Подтверждение не вечно: нет морганий keep мс — снова проверяем. Так
       фотография, подсунутая вместо живого лица, не наследует зелёную рамку. */
    if (g_lv_state == LIVE_OK && now - g_lv_last_blink > (double)g_lv_keep_ms) {
        g_lv_state = LIVE_CHECK;
        g_lv_blinks = 0;
        g_lv_valid_ms = 0.0;
        g_lv_t0 = now;
        lv_clear_prev();
        snprintf(g_lv_msg, sizeof(g_lv_msg), "моргните: подтвердите лицо");
        printf("live: подтверждение устарело, снова проверяю\n");
        fflush(stdout);
    }

    if (!valid || eye_l <= 0.0f || eye_r <= 0.0f || ref <= 0.0f) {
        /* Глаза не найдены (мелкое лицо, тёмные очки): моргание не проверить,
           но и обвинять в подделке не за что. Короткие пропуски не сбрасывают
           историю, иначе шум детектора рвёт проверку. */
        if (now - g_lv_valid_t > 1500.0) lv_clear_prev();
        if (g_lv_state == LIVE_CHECK && g_lv_valid_ms == 0.0 && now - g_lv_t0 > 3000.0)
            snprintf(g_lv_msg, sizeof(g_lv_msg), "не вижу глаза — подойдите ближе");
        return;
    }

    /* В проверке копим только время, когда глаза реально видны: иначе «фото»
       получало бы и живое лицо, которое стоит слишком далеко. */
    const double dt = (g_lv_update_t > 0.0) ? (now - g_lv_update_t) : 0.0;
    g_lv_update_t = now;
    g_lv_valid_t = now;
    if (g_lv_state == LIVE_CHECK) {
        g_lv_valid_ms += (dt > 200.0) ? 200.0 : dt;
        if (g_lv_valid_ms >= (double)g_lv_check_ms && !g_lv_open) {
            g_lv_state = LIVE_PHOTO;
            g_lv_blinks = 0;
            snprintf(g_lv_msg, sizeof(g_lv_msg), "фото: моргания нет");
            printf("live: фото (моргания нет)\n");
            fflush(stdout);
            return;
        }
    }

    if (!g_lv_prev_ok) {
        g_lv_eye_prev[0] = eye_l;
        g_lv_eye_prev[1] = eye_r;
        g_lv_ref_prev = ref;
        g_lv_prev_ok = 1;
        return;
    }
    /* Изменение глаза относительно переносицы за кадр. Берём среднее по двум
       глазам (шум одиночного глаза раньше выглядел морганием) и требуем, чтобы
       закрывались оба века. */
    const float dref = ref - g_lv_ref_prev;
    const float dl = (eye_l - g_lv_eye_prev[0]) - dref;
    const float dr = (eye_r - g_lv_eye_prev[1]) - dref;
    const float dv = 0.5f * (dl + dr);
    g_lv_eye_prev[0] = eye_l;
    g_lv_eye_prev[1] = eye_r;
    g_lv_ref_prev = ref;

    /* Порог адаптивный: считаем текущий шум |dv|, чтобы дрожание бумажного
       фото в руке не превращалось в «моргание». */
    float thr_c = 3.0f * g_lv_noise + 4.0f;   /* глаз стал ярче (веко закрылось) */
    if (thr_c < 6.0f) thr_c = 6.0f;
    if (thr_c > 25.0f) thr_c = 25.0f;
    float thr_o = 2.0f * g_lv_noise + 3.0f;   /* глаз снова открылся */
    if (thr_o < 4.0f) thr_o = 4.0f;
    if (thr_o > 18.0f) thr_o = 18.0f;

    if (g_lv_dbg && (g_lv_open || fabsf(dv) > 2.5f)) {
        printf("live: dbg dl=%.1f dr=%.1f dv=%.1f noise=%.1f thr_c=%.1f thr_o=%.1f st=%d\n",
               dl, dr, dv, g_lv_noise, thr_c, thr_o, g_lv_state);
        fflush(stdout);
    }

    const int close_ok =
        (dv > thr_c) &&
        (((dl > 0.3f * thr_c) && (dr > 0.3f * thr_c)) ||
         ((dl > 1.6f * thr_c) && (dr > -0.1f * thr_c)) ||
         ((dr > 1.6f * thr_c) && (dl > -0.1f * thr_c)));
    const int open_ok =
        (dv < -thr_o) &&
        (((dl < -0.3f * thr_o) && (dr < -0.3f * thr_o)) ||
         ((dl < -1.6f * thr_o) && (dr < 0.1f * thr_o)) ||
         ((dr < -1.6f * thr_o) && (dl < 0.1f * thr_o)));

    if (!g_lv_open) {
        if (close_ok && now - g_lv_last_blink > 250.0) {
            g_lv_open = 1;
            g_lv_close = 0;
            g_lv_peak = dv;
        } else {
            g_lv_noise = 0.9f * g_lv_noise + 0.1f * fabsf(dv);
            if (g_lv_noise < 0.5f) g_lv_noise = 0.5f;
            if (g_lv_noise > 20.0f) g_lv_noise = 20.0f;
        }
    } else {
        g_lv_close++;
        if (dv > g_lv_peak) g_lv_peak = dv;
        if (open_ok && g_lv_close >= 1 && g_lv_close <= 12) {
            /* закрытие и открытие уложились в ~0.7 с — это моргание */
            g_lv_open = 0;
            g_lv_close = 0;
            g_lv_last_blink = now;
            g_lv_valid_ms = 0.0;
            g_lv_blinks++;
            if (g_lv_dbg) {
                printf("live: dbg моргание #%d (пик %.1f)\n", g_lv_blinks, g_lv_peak);
                fflush(stdout);
            }
            if (g_lv_state == LIVE_OK) {
                /* уже подтверждено: моргание просто продлевает статус */
                snprintf(g_lv_msg, sizeof(g_lv_msg), "человек: моргание");
            } else if (g_lv_blinks >= g_lv_blinks_need) {
                g_lv_state = LIVE_OK;
                snprintf(g_lv_msg, sizeof(g_lv_msg), "человек: моргание");
                printf("live: подтверждено (моргание)\n");
                fflush(stdout);
            } else {
                snprintf(g_lv_msg, sizeof(g_lv_msg), "ещё раз моргните");
            }
        } else if (g_lv_close > 12) {
            g_lv_open = 0;   /* закрытие без открытия — не моргание */
            g_lv_close = 0;
        }
    }
}

int face_liveness_ok(void) { return g_lv_state == LIVE_OK ? 1 : 0; }

int face_liveness_state(void) { return g_lv_state; }

int face_liveness_face_ok(int x1, int y1, int x2, int y2, float luma) {
    (void)luma;
    if (g_lv_state != LIVE_OK) return 0;
    return lv_same_face(x1, y1, x2, y2);
}

void face_liveness_reset(void) {
    g_lv_state = LIVE_CHECK;
    g_lv_blinks = 0;
    g_lv_valid_ms = 0.0;
    g_lv_last_blink = 0.0;
    lv_clear_prev();
    g_lv_box[0] = g_lv_box[1] = g_lv_box[2] = g_lv_box[3] = 0;
    snprintf(g_lv_msg, sizeof(g_lv_msg), "ждём лицо в кадре");
}

int face_liveness_prompt(char *buf, int len) { return snprintf(buf, len, ""); }

int face_liveness_json(char *buf, int len) {
    static const char *st[] = {"check", "ok", "photo"};
    const int st_i = (g_lv_state >= 0 && g_lv_state <= 2) ? g_lv_state : 0;
    return snprintf(buf, len, "{\"state\":\"%s\",\"msg\":\"%s\"}", st[st_i], g_lv_msg);
}

/* ---------- галерея ---------- */
static void feat_normalize(float *v, int dim) {
    double n = 0.0;
    for (int i = 0; i < dim; i++) n += (double)v[i] * v[i];
    n = sqrt(n);
    if (n > 1e-9) {
        float inv = (float)(1.0 / n);
        for (int i = 0; i < dim; i++) v[i] *= inv;
    }
}

int face_gallery_reload(void) {
    DIR *d = opendir(g_dir);
    g_npeople = 0;
    if (d == NULL) return 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.') continue;
        if (!name_ok(e->d_name)) continue;
        if (g_npeople >= MAX_PEOPLE) break;
        char path[512];
        snprintf(path, sizeof(path), "%s/%s/feature.bin", g_dir, e->d_name);
        FILE *f = fopen(path, "rb");
        if (f == NULL) continue;
        char magic[4];
        uint32_t dim = 0;
        if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "CVF1", 4) != 0 ||
            fread(&dim, sizeof(dim), 1, f) != 1 || dim == 0 || dim > MAX_DIM) {
            fclose(f);
            continue;
        }
        person_t *p = &g_people[g_npeople];
        snprintf(p->name, sizeof(p->name), "%s", e->d_name);
        p->dim = (int)dim;
        if (fread(p->feat, sizeof(float), dim, f) != dim) {
            fclose(f);
            continue;
        }
        fclose(f);
        feat_normalize(p->feat, p->dim);
        g_npeople++;
    }
    closedir(d);
    printf("faces: галерея %d чел. из %s\n", g_npeople, g_dir);
    fflush(stdout);
    return g_npeople;
}

int face_gallery_json(char *buf, int len) {
    int off = snprintf(buf, len, "{\"people\":[");
    for (int i = 0; i < g_npeople && off < len - 4; i++)
        off += snprintf(buf + off, len - off, "%s\"%s\"", i ? "," : "", g_people[i].name);
    snprintf(buf + off, len - off, "]}");
    return 0;
}

static void json_escape(const char *s, char *out, int len) {
    int j = 0;
    for (; *s && j < len - 2; s++) {
        if (*s == '"' || *s == '\\') out[j++] = '\\';
        out[j++] = *s;
    }
    out[j] = 0;
}

int face_delete(const char *name) {
    if (!name_ok(name)) return -1;
    char path[512], file[600];
    person_path(name, path, sizeof(path));
    snprintf(file, sizeof(file), "%s/feature.bin", path);
    unlink(file);
    snprintf(file, sizeof(file), "%s/preview.jpg", path);
    unlink(file);
    snprintf(file, sizeof(file), "%s/meta.json", path);
    unlink(file);
    rmdir(path);
    face_gallery_reload();
    return 0;
}

/* ---------- регистрация ---------- */
int face_enroll_begin(const char *name) {
    if (!name_ok(name)) return -1;
    if (g_en_state == EN_COLLECT) return -2; /* уже идёт */

    snprintf(g_en_name, sizeof(g_en_name), "%s", name);
    g_en_state = EN_COLLECT;
    g_en_got = 0;
    g_en_dim = 0;
    g_en_preview = 0;
    g_en_t0 = now_ms();
    memset(g_en_sum, 0, sizeof(g_en_sum));
    const char *e = getenv("SY_ENROLL_N");
    g_en_need = (e != NULL) ? atoi(e) : 12;
    if (g_en_need < 3) g_en_need = 3;
    if (g_en_need > 40) g_en_need = 40;
    snprintf(g_en_msg, sizeof(g_en_msg),
             face_liveness_ok() ? "смотрите в камеру" : "проверяю лицо — смотрите в камеру");
    printf("faces: регистрация «%s», нужно кадров: %d\n", g_en_name, g_en_need);
    fflush(stdout);
    return 0;
}

int face_enroll_status(char *buf, int len) {
    if (g_en_state == EN_COLLECT && now_ms() - g_en_t0 > 25000.0) {
        g_en_state = EN_ERR;
        snprintf(g_en_msg, sizeof(g_en_msg),
                 "не удалось: подойдите ближе и моргните");
    }
    const char *st = (g_en_state == EN_COLLECT) ? "collecting"
                     : (g_en_state == EN_DONE) ? "done"
                     : (g_en_state == EN_ERR) ? "error" : "idle";
    int progress = (g_en_need > 0) ? (100 * g_en_got / g_en_need) : 0;
    char nm[160], ms[256];
    json_escape(g_en_name, nm, sizeof(nm));
    json_escape(g_en_msg, ms, sizeof(ms));
    snprintf(buf, len, "{\"state\":\"%s\",\"progress\":%d,\"name\":\"%s\",\"msg\":\"%s\"}", st,
             progress, nm, ms);
    return 0;
}

/* Сбросить регистрацию (например, если лицо оказалось подделкой). */
void face_enroll_abort(void) {
    if (g_en_state == EN_COLLECT) {
        g_en_state = EN_IDLE;
        g_en_got = 0;
        g_en_name[0] = 0;
        snprintf(g_en_msg, sizeof(g_en_msg), "регистрация прервана: похоже на фото");
        printf("faces: регистрация прервана (лицо не меняется — фото?)\n");
        fflush(stdout);
    }
}

int face_enroll_take_preview(char *name, int len) {
    if (!g_en_preview) return 0;
    g_en_preview = 0;
    snprintf(name, len, "%s", g_en_name);
    return 1;
}

static void enroll_finish(void) {
    /* средний вектор -> L2-норма -> в файл */
    float avg[MAX_DIM];
    for (int i = 0; i < g_en_dim; i++) avg[i] = g_en_sum[i] / (float)g_en_got;
    feat_normalize(avg, g_en_dim);

    char path[512], file[600];
    strcpy(path, g_dir);
    mkdir(path, 0755);
    person_path(g_en_name, path, sizeof(path));
    mkdir(path, 0755);
    snprintf(file, sizeof(file), "%s/feature.bin", path);
    FILE *f = fopen(file, "wb");
    if (f == NULL) {
        g_en_state = EN_ERR;
        snprintf(g_en_msg, sizeof(g_en_msg), "не смог записать %s", file);
        return;
    }
    uint32_t dim = (uint32_t)g_en_dim;
    fwrite("CVF1", 1, 4, f);
    fwrite(&dim, sizeof(dim), 1, f);
    fwrite(avg, sizeof(float), dim, f);
    fclose(f);

    snprintf(file, sizeof(file), "%s/meta.json", path);
    f = fopen(file, "w");
    if (f != NULL) {
        fprintf(f, "{\"name\":\"%s\",\"frames\":%d,\"dim\":%u,\"time\":%ld}\n", g_en_name,
                g_en_got, dim, (long)time(NULL));
        fclose(f);
    }
    g_en_state = EN_DONE;
    g_en_preview = 1;
    snprintf(g_en_msg, sizeof(g_en_msg), "готово: %d кадров", g_en_got);
    printf("faces: «%s» сохранён (%d кадров, dim=%u)\n", g_en_name, g_en_got, dim);
    fflush(stdout);
    face_gallery_reload();
}

/* ---------- liveness ---------- */
/* CVI_TDL_Liveness(handle, rgbFrame, irFrame, rgb_face, ir_face): вторым кадром
   ожидается ИК-кадр, у нас его нет — передаём NULL (RGB-модель).
   На этой прошивке вызов умеет зависать на TPU, поэтому делаем его в отдельном
   потоке со сторожевым таймером: нет ответа за 2 с -> модель отключаем. */
typedef struct {
    cvitdl_handle_t h;
    VIDEO_FRAME_INFO_S frame;
    cvtdl_face_t faces;
    volatile int done;
    CVI_S32 rc;
} live_ctx_t;

static void *live_thread(void *arg) {
    live_ctx_t *c = (live_ctx_t *)arg;
    c->rc = p_live(c->h, &c->frame, NULL, &c->faces, NULL);
    c->done = 1;
    return NULL;
}

/* 0 — оценка получена, -2 — модель зависла (отключаем), прочее — ошибка. */
static int liveness_score(cvitdl_handle_t h, VIDEO_FRAME_INFO_S *frame, cvtdl_face_t *faces) {
    live_ctx_t *c = (live_ctx_t *)calloc(1, sizeof(live_ctx_t));
    if (c == NULL) return -1;
    c->h = h;
    c->frame = *frame;
    c->faces = *faces;
    c->rc = -1;
    pthread_t th;
    if (pthread_create(&th, NULL, live_thread, c) != 0) {
        free(c);
        return -1;
    }
    int rc = -2;
    for (int i = 0; i < 40; i++) {
        if (c->done) {
            rc = (c->rc == CVI_SUCCESS) ? 0 : -3;
            break;
        }
        usleep(50000);
    }
    if (rc == 0) *faces = c->faces;
    if (c->done) {
        pthread_join(th, NULL);
        free(c);
    } else {
        /* Поток завис внутри TDL: ctx не трогаем (утечка осознанная), поток
           умрёт вместе с процессом. */
        pthread_detach(th);
        g_live_on = 0;
        g_live_thr = -1.0f;
        printf("faces: liveness не ответил за 2 с — модель отключена\n");
        fflush(stdout);
    }
    return rc;
}

/* ---------- TDL ---------- */

/* Освобождение TDL в отдельном потоке: DestroyHandle/StopGrp могут зависнуть.
   Порядок важен: сначала останавливаем и уничтожаем VPSS-группу TDL (она же
   держит наш VB-пул), иначе память ION не возвращается в систему. */
static volatile int g_cleanup_done = 0;

static void *tdl_destroy_thread(void *arg) {
    cvitdl_handle_t h = (cvitdl_handle_t)arg;
    const int grp = (int)g_tdl_grp;
    printf("faces: очистка: стоп группы...\n");
    fflush(stdout);
    const int s1 = vpss_face_stop_grp(grp);
    printf("faces: очистка: DestroyGrp...\n");
    fflush(stdout);
    const int s2 = vpss_face_destroy_grp(grp);
    printf("faces: стоп группы rc=0x%x, DestroyGrp(%d) rc=0x%x\n", (unsigned)s1, grp,
           (unsigned)s2);
    fflush(stdout);
    /* Сначала handle: он держит ссылки на блоки пула, и DestroyPool без
       этого виснет. */
    printf("faces: очистка: DestroyHandle...\n");
    fflush(stdout);
    p_destroy(h);
    /* VB-пул не трогаем: DestroyPool виснет, пока на блоках висят ссылки TDL.
       Пул освободит ядро при выходе процесса. */
    printf("faces: очистка: готово\n");
    fflush(stdout);
    g_cleanup_done = 1;
    return NULL;
}

/* Создаёт handle TDL и открывает модели. Отдельно от face_init, чтобы можно
   было пересоздать TDL после зависания TPU (face_restart). */
static int tdl_open_handle(void) {
    /* На этой прошивке группы 0..2 созданы системой: CreateGrp на них даёт
       0xc0068004 (EXIST) и TDL падает с CVI_TDL_ERR_INIT_VPSS. Поэтому берём
       первую свободную группу. */
    const char *grp_env = getenv("SY_TDL_GRP");
    VPSS_GRP tdl_grp;
    if (grp_env != NULL && atoi(grp_env) > 0) {
        tdl_grp = (VPSS_GRP)atoi(grp_env);
    } else {
        /* Группы 0..2 на этой прошивке созданы системой (CreateGrp -> EXIST),
           а группы после аварийных завершений могут остаться занятыми. Поэтому
           берём первую, которую реально можно создать. */
        int g = 0;
        for (int i = 1; i < 16; i++) {
            if (vpss_face_group_free(i)) {
                g = i;
                break;
            }
        }
        tdl_grp = (g > 0) ? (VPSS_GRP)g : (VPSS_GRP)3;
    }
    g_tdl_grp = tdl_grp;
    printf("faces: TDL VPSS группа %d\n", (int)tdl_grp);
    fflush(stdout);
    if (p_create2(&g_tdl, tdl_grp, 0) != CVI_TDL_SUCCESS) {
        printf("faces: CVI_TDL_CreateHandle2 не удался\n");
        fflush(stdout);
        return -1;
    }
    if (p_open(g_tdl, CVI_TDL_SUPPORTED_MODEL_SCRFDFACE, g_fd_model) != CVI_SUCCESS) {
        printf("faces: не открылась модель детекции %s\n", g_fd_model);
        fflush(stdout);
        return -2;
    }
    if (p_open(g_tdl, CVI_TDL_SUPPORTED_MODEL_FACERECOGNITION, g_fr_model) != CVI_SUCCESS) {
        printf("faces: не открылась модель распознавания %s\n", g_fr_model);
        fflush(stdout);
        return -3;
    }
    if (g_live_model[0] && p_live != NULL) {
        /* Модель RGB-liveness; ИК-кадра у нас нет — вызов идёт с irFrame=NULL. */
        if (p_open(g_tdl, CVI_TDL_SUPPORTED_MODEL_LIVENESS, g_live_model) == CVI_SUCCESS) {
            g_live_on = 1;
        } else {
            printf("faces: liveness-модель не открылась, проверка фото отключена\n");
        }
    } else if (p_live == NULL) {
        printf("faces: в этой сборке TDL нет RGB-liveness, проверка фото отключена\n");
    }
    p_set_thr(g_tdl, CVI_TDL_SUPPORTED_MODEL_SCRFDFACE, g_det_thr);
    printf("faces: TDL готов (fd=%s, fr=%s, live=%s, thr=%.2f/%.2f%.2f)\n", g_fd_model, g_fr_model,
           g_live_on ? g_live_model : "off", g_det_thr, g_match_thr, g_live_thr);
    fflush(stdout);
    return 0;
}

/* VB-пул, на котором TDL делает свою препроцессинг-преобразование. */
static int tdl_pool_create(void) {
    VB_POOL_CONFIG_S cfg;
    memset(&cfg, 0, sizeof(cfg));
    /* Блок с запасом под самый большой вход модели (960x540x3); TDL создаёт
       на нём каналы своего VPSS. */
    cfg.u32BlkSize = 960u * 540u * 3u + (64u * 1024u);
    cfg.u32BlkCnt = 3;
    cfg.enRemapMode = VB_REMAP_MODE_NONE;
    g_tdl_pool = CVI_VB_CreatePool(&cfg);
    if (g_tdl_pool == VB_INVALID_POOLID) {
        printf("faces: не удалось создать VB-пул для TDL\n");
        fflush(stdout);
        return -1;
    }
    CVI_S32 rc = (p_set_vbpool != NULL) ? p_set_vbpool(g_tdl, 0, g_tdl_pool) : -1;
    if (p_set_timeout != NULL) p_set_timeout(g_tdl, 1000);
    printf("faces: TDL VB pool id=%d rc=0x%x\n", g_tdl_pool, (unsigned)rc);
    fflush(stdout);
    return (rc == CVI_SUCCESS) ? 0 : -1;
}

int face_init(const char *fd_model, const char *fr_model, const char *live_model,
              const char *faces_dir, float det_thr, float match_thr, float live_thr) {
    if (faces_dir != NULL && faces_dir[0]) snprintf(g_dir, sizeof(g_dir), "%s", faces_dir);
    g_match_thr = match_thr;
    g_live_thr = live_thr;
    if (fd_model != NULL && fd_model[0]) snprintf(g_fd_model, sizeof(g_fd_model), "%s", fd_model);
    if (fr_model != NULL && fr_model[0]) snprintf(g_fr_model, sizeof(g_fr_model), "%s", fr_model);
    if (live_model != NULL && live_model[0])
        snprintf(g_live_model, sizeof(g_live_model), "%s", live_model);
    else
        g_live_model[0] = '\0';
    g_det_thr = det_thr;

    if (tdl_load() != 0) return -1;
    if (tdl_open_handle() != 0) return -1;
    face_gallery_reload();
    return 0;
}

/* Пересоздаёт TDL целиком: после зависания TPU старый handle больше не
   работает (инференс падает мгновенно), помогает только пересоздание. */
/* Освобождение TDL. Вызовы StopGrp/DestroyHandle умеют виснуть на этой
   прошивке, поэтому делаем их в отдельном потоке и ждём не дольше 3 с:
   лучше потерять VB-пул, чем не дать приложению завершиться. */
static void tdl_release(void) {
    if (g_tdl == NULL) return;
    g_cleanup_done = 0;
    pthread_t th;
    const int cr = pthread_create(&th, NULL, tdl_destroy_thread, g_tdl);
    if (cr == 0) {
        pthread_detach(th);
        for (int i = 0; i < 60 && !g_cleanup_done; i++) usleep(50000);
    } else {
        p_destroy(g_tdl);
    }
    g_tdl = NULL;
}

int face_restart(void) {
    printf("faces: пересоздаю TDL\n");
    fflush(stdout);
    tdl_release();
    g_live_on = 0;
    if (tdl_open_handle() != 0) return -1;
    /* Тот же VB-пул переиспользуем: новый создавать не нужно, а старый не
       удалить (DestroyPool с висящими ссылками TDL виснет). */
    if (g_tdl_pool != VB_INVALID_POOLID) {
        CVI_S32 rc = (p_set_vbpool != NULL) ? p_set_vbpool(g_tdl, 0, g_tdl_pool) : -1;
        if (p_set_timeout != NULL) p_set_timeout(g_tdl, 1000);
        printf("faces: TDL VB pool id=%d rc=0x%x\n", g_tdl_pool, (unsigned)rc);
        fflush(stdout);
    }
    return 0;
}

static float cosine_int8(const int8_t *a, int na, const float *b, int nb) {
    if (na != nb || na <= 0) return -1.0f;
    double dot = 0.0, aa = 0.0, bb = 0.0;
    for (int i = 0; i < na; i++) {
        double x = (double)a[i], y = (double)b[i];
        dot += x * y;
        aa += x * x;
        bb += y * y;
    }
    if (aa < 1e-9 || bb < 1e-9) return -1.0f;
    return (float)(dot / sqrt(aa * bb));
}

static float cosine_float(const float *a, int na, const float *b, int nb) {
    if (na != nb || na <= 0) return -1.0f;
    double dot = 0.0, aa = 0.0, bb = 0.0;
    for (int i = 0; i < na; i++) {
        dot += (double)a[i] * b[i];
        aa += (double)a[i] * a[i];
        bb += (double)b[i] * b[i];
    }
    if (aa < 1e-9 || bb < 1e-9) return -1.0f;
    return (float)(dot / sqrt(aa * bb));
}

int face_video_ready(void) {
    if (g_tdl == NULL) return -1;
    if (g_tdl_pool != VB_INVALID_POOLID) return 0;
    return tdl_pool_create();
}

static float feature_similarity(const cvtdl_feature_t *f, const float *g, int gn) {
    if (f->ptr == NULL || f->size == 0 || gn <= 0) return -1.0f;
    if (f->type == TYPE_INT8)
        return cosine_int8(f->ptr, (int)f->size, g, gn);
    if (f->type == TYPE_FLOAT)
        return cosine_float((const float *)f->ptr, (int)f->size, g, gn);
    return -1.0f;
}

/* Копирует фичу в float-вектор; возвращает размерность или 0. */
static int feature_to_float(const cvtdl_feature_t *f, float *out, int max) {
    if (f->ptr == NULL || f->size == 0 || (int)f->size > max) return 0;
    if (f->type == TYPE_INT8) {
        for (uint32_t i = 0; i < f->size; i++) out[i] = (float)f->ptr[i];
    } else if (f->type == TYPE_FLOAT) {
        memcpy(out, f->ptr, (size_t)f->size * sizeof(float));
    } else {
        return 0;
    }
    return (int)f->size;
}

/* Распознавание (cviface) — самая тяжёлая модель после детекции. Гоняем её
   раз в 3 кадра, а на промежуточных берём имя из кэша по IoU: нагрузка на TPU
   ниже, подпись не мигает. */
static face_box_t g_rec_boxes[8];
static int g_rec_n = 0;
static int g_frame_no = 0;

int face_process(void *frame_info, face_box_t *out, int max) {
    if (g_tdl == NULL || frame_info == NULL) return -1;
    VIDEO_FRAME_INFO_S *vf = (VIDEO_FRAME_INFO_S *)frame_info;
    cvtdl_face_t faces;
    memset(&faces, 0, sizeof(faces));

    CVI_S32 rc_det = (p_scrfd != NULL)
                         ? p_scrfd(g_tdl, vf, &faces)
                         : p_det(g_tdl, vf, CVI_TDL_SUPPORTED_MODEL_SCRFDFACE, &faces);
    if (rc_det != CVI_SUCCESS) {
        static int logged = 0;
        if (logged++ < 5) {
            printf("faces: ScrFDFace rc=0x%x frame=%ux%u fmt=%d stride=%u vir=%p phy=0x%llx\n",
                   (unsigned)rc_det, (unsigned)vf->stVFrame.u32Width,
                   (unsigned)vf->stVFrame.u32Height, (int)vf->stVFrame.enPixelFormat,
                   (unsigned)vf->stVFrame.u32Stride[0], (void *)vf->stVFrame.pu8VirAddr[0],
                   (unsigned long long)vf->stVFrame.u64PhyAddr[0]);
            fflush(stdout);
        }
        return -2;
    }
    /* Пока живость не подтверждена, лицо не опознаём: фотография не моргает,
       значит пассивная проверка её отсечёт. */
    const int live_ok = face_liveness_ok();
    const int is_photo = (face_liveness_state() == LIVE_PHOTO);
    const int do_rec = ((g_frame_no++ % 3) == 0);
    if (faces.size > 0 && do_rec) {
        if (p_rec(g_tdl, vf, &faces) != CVI_SUCCESS) {
            p_free_face(&faces);
            return -3;
        }
        if (g_live_on) {
            const int lr = liveness_score(g_tdl, vf, &faces);
            if (lr == -2) {
                /* Модель зависла: TDL надо пересоздать (этим занимается поток
                   детекции в main_face.cpp). */
                p_free_face(&faces);
                return -5;
            }
            if (lr != 0) {
                printf("faces: liveness не выполнился, проверка фото отключена\n");
                fflush(stdout);
                g_live_on = 0;
                g_live_thr = -1.0f;
            }
        }
    }

    /* bbox приходит в системе координат кадра; TDL может вернуть её в своих
       width/height (после внутреннего ресайза). */
    float sx = 1.0f, sy = 1.0f;
    const uint32_t fw = vf->stVFrame.u32Width, fh = vf->stVFrame.u32Height;
    if (faces.width > 0 && faces.width != fw) sx = (float)fw / (float)faces.width;
    if (faces.height > 0 && faces.height != fh) sy = (float)fh / (float)faces.height;

    /* Крупнейшее лицо — кандидат для регистрации. */
    int enroll_idx = -1;
    float enroll_area = 0.0f;
    if (g_en_state == EN_COLLECT && !is_photo) {
        for (uint32_t i = 0; i < faces.size; i++) {
            cvtdl_bbox_t *bb = &faces.info[i].bbox;
            float a = (bb->x2 - bb->x1) * (bb->y2 - bb->y1);
            if (a > enroll_area) {
                enroll_area = a;
                enroll_idx = (int)i;
            }
        }
    }

    int n = 0;
    for (uint32_t i = 0; i < faces.size && n < max; i++) {
        cvtdl_face_info_t *fi = &faces.info[i];
        face_box_t b;
        memset(&b, 0, sizeof(b));
        b.x1 = (int)(fi->bbox.x1 * sx);
        b.y1 = (int)(fi->bbox.y1 * sy);
        b.x2 = (int)(fi->bbox.x2 * sx);
        b.y2 = (int)(fi->bbox.y2 * sy);
        b.score = fi->bbox.score;
        b.sim = -1.0f;
        b.have_lm = 0;
        if (fi->pts.x != NULL && fi->pts.y != NULL && fi->pts.size >= 5) {
            for (int k = 0; k < 5; k++) {
                b.lm[k * 2 + 0] = fi->pts.x[k] * sx;
                b.lm[k * 2 + 1] = fi->pts.y[k] * sy;
            }
            b.have_lm = 1;
        }
        b.live = (fi->liveness_score >= 0.0f) ? fi->liveness_score : -1.0f;
        b.live_ok = live_ok;
        if (!do_rec) {
            /* кадр без распознавания: имя с прошлого кадра, если рамка та же */
            int best_iou = 0, ci = -1;
            for (int c = 0; c < g_rec_n; c++) {
                const int io = lv_iou(b.x1, b.y1, b.x2, b.y2, g_rec_boxes[c].x1,
                                      g_rec_boxes[c].y1, g_rec_boxes[c].x2, g_rec_boxes[c].y2);
                if (io > best_iou) {
                    best_iou = io;
                    ci = c;
                }
            }
            if (ci >= 0 && best_iou >= 30) {
                b.sim = g_rec_boxes[ci].sim;
                snprintf(b.name, sizeof(b.name), "%s", g_rec_boxes[ci].name);
            }
        }
        if (fi->feature.ptr != NULL && fi->feature.size > 0) {
            float best = -1.0f;
            int best_i = -1;
            for (int p = 0; p < g_npeople; p++) {
                float s = feature_similarity(&fi->feature, g_people[p].feat, g_people[p].dim);
                if (s > best) {
                    best = s;
                    best_i = p;
                }
            }
            b.sim = best;
            const int model_ok = (g_live_thr < 0.0f) || !g_live_on || (b.live >= g_live_thr);
            if (best_i >= 0 && best >= g_match_thr && model_ok && !is_photo)
                snprintf(b.name, sizeof(b.name), "%s", g_people[best_i].name);
            /* Имя показываем и до моргания — жёлтой рамкой «имя?»; зелёной
               рамка станет, когда b.live_ok подтвердится морганием. Фотография
               (is_photo) имя не получает: рисуется «photo?». */
        }
        out[n++] = b;

        /* Сбор кадров для регистрации: только самое крупное лицо. */
        if (g_en_state == EN_COLLECT && (int)i == enroll_idx) {
            int bw = b.x2 - b.x1, bh = b.y2 - b.y1;
            int live_ok = (g_live_thr < 0.0f) || !g_live_on || (b.live >= g_live_thr);
            if (is_photo) live_ok = 0;
            float vec[MAX_DIM];
            int dim = feature_to_float(&fi->feature, vec, MAX_DIM);
            if (bw >= 60 && bh >= 60 && dim > 0 && live_ok) {
                feat_normalize(vec, dim);
                if (g_en_got == 0) g_en_dim = dim;
                if (g_en_dim == dim) {
                    for (int k = 0; k < dim; k++) g_en_sum[k] += vec[k];
                    g_en_got++;
                    if (g_en_got >= g_en_need) enroll_finish();
                }
            }
        }
    }
    if (do_rec) {
        g_rec_n = (n < 8) ? n : 8;
        for (int i = 0; i < g_rec_n; i++) g_rec_boxes[i] = out[i];
    }
    p_free_face(&faces);
    return n;
}

void face_close(void) {
    tdl_release();
    g_tdl_pool = VB_INVALID_POOLID; /* освободит ядро при выходе процесса */
}
