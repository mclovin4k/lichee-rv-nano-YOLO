#ifndef VPSS_FACE_H
#define VPSS_FACE_H

#include <linux/cvi_comm_video.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Дополнительный канал VPSS (группа 0, канал 1) с RGB_888 — в этом формате
   TDL принимает кадр. Канал открывается после того, как камера уже подняла
   группу 0. 0 — успех. */
int vpss_face_open(int src_w, int src_h, int rot);

/* Номер свободной VPSS-группы (-1, если не определить): группы 0..2 на
   этой прошивке созданы системой, TDL должен брать следующую. */
int vpss_face_available_grp(void);

/* 1 — группу можно создать (свободна), 0 — занята. */
int vpss_face_group_free(int grp);

/* Уничтожить группу (нужно после DestroyHandle TDL). 0 — успех. */
int vpss_face_destroy_grp(int grp);

/* Остановить группу и канал 0 (иначе DestroyGrp виснет). */
int vpss_face_stop_grp(int grp);

/* Забрать самый свежий кадр (0 — успех). Кадр удерживается до release. */
int vpss_face_get(VIDEO_FRAME_INFO_S *fi, int timeout_ms);
void vpss_face_release(VIDEO_FRAME_INFO_S *fi);

/* Доступ к пикселям кадра (mapping в память CPU). NULL — не удалось. */
void *vpss_face_map(const VIDEO_FRAME_INFO_S *fi, uint32_t *len);
void vpss_face_unmap(void *p, uint32_t len);

/* То же, но кадр повёрнут на SY_ROTATE градусов (в софте) — для TDL. */
int vpss_face_get_rot(VIDEO_FRAME_INFO_S *fi, int timeout_ms);
void vpss_face_release_rot(VIDEO_FRAME_INFO_S *fi);

void vpss_face_close(void);

/* Диагностика (SY_VPSS_TEST): какие VPSS-группы можно создать. */
int vpss_face_selftest(void);

#ifdef __cplusplus
}
#endif

#endif
