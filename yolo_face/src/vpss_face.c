/* RGB-канал VPSS для TDL: группа 0, канал 1 (канал 0 занят потоком NV21).
   Поворот канала VPSS на этой прошивке запрещён (ILLEGAL_PARAM), поэтому кадр
   для TDL поворачивается в софте в отдельном VB-пуле.

   Устройство повторяет рабочий vpss_det.c: собственный VB-пул (3 блока по
   кадру), отдельный поток-дренажёр, который всегда забирает кадры из канала
   и держит самый свежий в однокадровом «почтовом ящике». Без дренажёра
   заполненная очередь канала тормозит всю группу VPSS и вместе с ней поток. */

#include "vpss_face.h"

#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <stdint.h>

#include "cvi_sys.h"
#include "cvi_vb.h"
#include "cvi_vpss.h"

#define FACE_GRP 0
#define FACE_CHN VPSS_CHN1

static void *vpss_lib;
static CVI_S32 (*p_SetChnAttr)(VPSS_GRP, VPSS_CHN, const VPSS_CHN_ATTR_S *);
static CVI_S32 (*p_EnableChn)(VPSS_GRP, VPSS_CHN);
static CVI_S32 (*p_DisableChn)(VPSS_GRP, VPSS_CHN);
static CVI_S32 (*p_AttachVbPool)(VPSS_GRP, VPSS_CHN, VB_POOL);
static CVI_S32 (*p_DetachVbPool)(VPSS_GRP, VPSS_CHN);
static CVI_S32 (*p_GetChnFrame)(VPSS_GRP, VPSS_CHN, VIDEO_FRAME_INFO_S *, CVI_S32);
static CVI_S32 (*p_ReleaseChnFrame)(VPSS_GRP, VPSS_CHN, const VIDEO_FRAME_INFO_S *);
static CVI_S32 (*p_SetChnRotation)(VPSS_GRP, VPSS_CHN, ROTATION_E);

static VB_POOL face_pool = VB_INVALID_POOLID;
static int face_open_flag = 0;

/* VPSS/VI работают с системной libvpu.so (она согласована с драйвером этой
   прошивки).  Патченная middleware-сборка нужна только TDL, её грузит
   face_pipeline.c через SY_VPU_LIB.  Ищем libvpu обычным порядком:
   LD_LIBRARY_PATH, затем системные каталоги. */
static void *vpu_dlopen(int flags) {
    void *lib = dlopen("libvpu.so", flags);
    if (lib == NULL) lib = dlopen("/mnt/system/usr/lib/libvpu.so", flags);
    if (lib == NULL) lib = dlopen("/mnt/system/lib/libvpu.so", flags);
    return lib;
}

/* Пул под повёрнутый кадр (TDL принимает кадр с физическим адресом). */
static VB_POOL rot_pool = VB_INVALID_POOLID;
static uint32_t rot_blk_size = 0;
static int g_rot_angle = 0;
static int g_src_w = 0, g_src_h = 0;
static int g_dst_w = 0, g_dst_h = 0;
static uint32_t g_dst_stride = 0;

static pthread_mutex_t slot_mx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t slot_cv = PTHREAD_COND_INITIALIZER;
static pthread_t drain_thread;
static volatile int drain_run = 0;
static VIDEO_FRAME_INFO_S slot_fi;
static int slot_have = 0;

static void *drain_main(void *arg) {
    (void)arg;
    int logged = 0;
    while (drain_run) {
        VIDEO_FRAME_INFO_S fi;
        memset(&fi, 0, sizeof(fi));
        if (p_GetChnFrame(FACE_GRP, FACE_CHN, &fi, 200) != CVI_SUCCESS) continue;

        if (!logged) {
            logged = 1;
            printf("vpss_face: кадр %ux%u fmt=%d stride=%u vir=%p\n", fi.stVFrame.u32Width,
                   fi.stVFrame.u32Height, (int)fi.stVFrame.enPixelFormat,
                   fi.stVFrame.u32Stride[0], fi.stVFrame.pu8VirAddr[0]);
            fflush(stdout);
        }

        pthread_mutex_lock(&slot_mx);
        if (slot_have) {
            pthread_mutex_unlock(&slot_mx);
            p_ReleaseChnFrame(FACE_GRP, FACE_CHN, &fi);
            continue;
        }
        slot_fi = fi;
        slot_have = 1;
        pthread_cond_broadcast(&slot_cv);
        pthread_mutex_unlock(&slot_mx);
        {
            static int n_store = 0;
            if (++n_store <= 3) {
                printf("vpss_face: кадр %d в слоте\n", n_store);
                fflush(stdout);
            }
        }
    }
    return NULL;
}

#define LOAD(fp, sym)                                            \
    do {                                                         \
        *(void **)(&fp) = dlsym(vpss_lib, sym);                  \
        if ((fp) == NULL) {                                      \
            fprintf(stderr, "vpss_face: нет %s в libvpu\n", sym); \
            return -1;                                           \
        }                                                        \
    } while (0)

int vpss_face_open(int src_w_arg, int src_h_arg, int rot) {
    if (face_open_flag) return 0;
    if (src_w_arg <= 0 || src_h_arg <= 0) return -1;
    const int src_w = src_w_arg, src_h = src_h_arg;

    vpss_lib = vpu_dlopen(RTLD_NOW | RTLD_LOCAL);
    if (vpss_lib == NULL) {
        fprintf(stderr, "vpss_face: libvpu.so не загрузилась (%s)\n", dlerror());
        return -1;
    }

    LOAD(p_SetChnAttr, "CVI_VPSS_SetChnAttr");
    LOAD(p_EnableChn, "CVI_VPSS_EnableChn");
    LOAD(p_DisableChn, "CVI_VPSS_DisableChn");
    LOAD(p_AttachVbPool, "CVI_VPSS_AttachVbPool");
    LOAD(p_DetachVbPool, "CVI_VPSS_DetachVbPool");
    LOAD(p_GetChnFrame, "CVI_VPSS_GetChnFrame");
    LOAD(p_ReleaseChnFrame, "CVI_VPSS_ReleaseChnFrame");
    LOAD(p_SetChnRotation, "CVI_VPSS_SetChnRotation");

    VB_POOL_CONFIG_S cfg;
    memset(&cfg, 0, sizeof(cfg));
    /* +64КБ на выравнивание строк (повёрнутый кадр шире исходного). */
    cfg.u32BlkSize = (CVI_U32)src_w * (CVI_U32)src_h * 3 + (64u * 1024u);
    cfg.u32BlkCnt = 3;
    cfg.enRemapMode = VB_REMAP_MODE_NONE;
    face_pool = CVI_VB_CreatePool(&cfg);
    if (face_pool == VB_INVALID_POOLID) {
        fprintf(stderr, "vpss_face: CVI_VB_CreatePool(%u x3) не удался\n",
                (unsigned)cfg.u32BlkSize);
        return -1;
    }

    /* Перебираем варианты: эта сборка VPSS отвергает часть полей, а поворот
       принимает не во всех последовательностях вызовов.
       бит0 — глубина очереди канала, бит1 — поворот до EnableChn,
       бит2 — в атрибутах канала уже перевёрнутые размеры. */
    int ok = 0;
    for (int v = 0; v < 2 && !ok; v++) {
        const int depth_var = (v & 1);

        VPSS_CHN_ATTR_S a;
        memset(&a, 0, sizeof(a));
        a.u32Width = (CVI_U32)src_w;
        a.u32Height = (CVI_U32)src_h;
        a.enVideoFormat = VIDEO_FORMAT_LINEAR;
        a.enPixelFormat = PIXEL_FORMAT_RGB_888;
        a.stFrameRate.s32SrcFrameRate = -1;
        a.stFrameRate.s32DstFrameRate = -1;
        a.bMirror = CVI_FALSE;
        a.bFlip = CVI_FALSE;
        a.u32Depth = (depth_var == 0) ? 2 : 1;
        a.stAspectRatio.enMode = ASPECT_RATIO_NONE;

        CVI_S32 ret = p_SetChnAttr(FACE_GRP, FACE_CHN, &a);
        printf("vpss_face: вариант %d (depth=%d) SetChnAttr ret=0x%x\n", v, depth_var, ret);
        fflush(stdout);
        if (ret != CVI_SUCCESS) continue;

        if (p_EnableChn(FACE_GRP, FACE_CHN) != CVI_SUCCESS) {
            printf("vpss_face: вариант %d EnableChn не удался\n", v);
            fflush(stdout);
            continue;
        }
        if (p_AttachVbPool(FACE_GRP, FACE_CHN, face_pool) != CVI_SUCCESS) {
            printf("vpss_face: вариант %d AttachVbPool не удался\n", v);
            fflush(stdout);
            p_DisableChn(FACE_GRP, FACE_CHN);
            continue;
        }
        ok = 1;
    }
    if (!ok) {
        CVI_VB_DestroyPool(face_pool);
        face_pool = VB_INVALID_POOLID;
        return -1;
    }

    drain_run = 1;
    if (pthread_create(&drain_thread, NULL, drain_main, NULL) != 0) {
        drain_run = 0;
        p_DetachVbPool(FACE_GRP, FACE_CHN);
        p_DisableChn(FACE_GRP, FACE_CHN);
        CVI_VB_DestroyPool(face_pool);
        face_pool = VB_INVALID_POOLID;
        return -1;
    }
    face_open_flag = 1;
    printf("vpss_face: канал RGB %dx%d открыт\n", src_w, src_h);
    fflush(stdout);

    if (rot != 0) {
        g_rot_angle = rot;
        g_src_w = src_w_arg;
        g_src_h = src_h_arg;
        if (rot == 180) {
            g_dst_w = src_w_arg;
            g_dst_h = src_h_arg;
        } else {
            g_dst_w = src_h_arg;
            g_dst_h = src_w_arg;
        }
        g_dst_stride = (uint32_t)(((g_dst_w * 3 + 31) / 32) * 32);
        rot_blk_size = g_dst_stride * (uint32_t)g_dst_h + 4096u;

        VB_POOL_CONFIG_S rcfg;
        memset(&rcfg, 0, sizeof(rcfg));
        rcfg.u32BlkSize = rot_blk_size;
        rcfg.u32BlkCnt = 2;
        rcfg.enRemapMode = VB_REMAP_MODE_NONE;
        rot_pool = CVI_VB_CreatePool(&rcfg);
        if (rot_pool == VB_INVALID_POOLID) {
            printf("vpss_face: не удалось создать пул под повёрнутый кадр\n");
            fflush(stdout);
            g_rot_angle = 0;
        } else {
            printf("vpss_face: поворот %d -> %dx%d stride=%u\n", rot, g_dst_w, g_dst_h, g_dst_stride);
            fflush(stdout);
        }
    }
    return 0;
}

int vpss_face_get(VIDEO_FRAME_INFO_S *fi, int timeout_ms) {
    if (!face_open_flag || fi == NULL) return -1;
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += timeout_ms / 1000;
    ts.tv_nsec += (long)(timeout_ms % 1000) * 1000000L;
    if (ts.tv_nsec >= 1000000000L) {
        ts.tv_sec++;
        ts.tv_nsec -= 1000000000L;
    }

    pthread_mutex_lock(&slot_mx);
    while (!slot_have && drain_run) {
        int rc = pthread_cond_timedwait(&slot_cv, &slot_mx, &ts);
        if (rc == ETIMEDOUT) break;
    }
    int ok = slot_have;
    if (ok) {
        *fi = slot_fi;
        slot_have = 0;
    }
    pthread_mutex_unlock(&slot_mx);
    if (!ok) {
        static int n_to = 0;
        if (++n_to <= 3) {
            printf("vpss_face: get: таймаут (%d)\n", n_to);
            fflush(stdout);
        }
    }
    return ok ? 0 : -1;
}

/* Отображает кадр в память CPU для анализа пикселей (антиподделка). */
void *vpss_face_map(const VIDEO_FRAME_INFO_S *fi, uint32_t *len) {
    if (fi == NULL) return NULL;
    const VIDEO_FRAME_S *sf = &fi->stVFrame;
    const uint32_t l = sf->u32Length[0] ? sf->u32Length[0] : sf->u32Stride[0] * sf->u32Height;
    void *v = CVI_SYS_MmapCache(sf->u64PhyAddr[0], l);
    if (v == NULL) return NULL;
    CVI_SYS_IonInvalidateCache(sf->u64PhyAddr[0], (void *)v, l);
    if (len != NULL) *len = l;
    return v;
}

void vpss_face_unmap(void *p, uint32_t len) {
    if (p != NULL) CVI_SYS_Munmap(p, len);
}

/* Поворот на 90° по часовой: dst(x,y) = src(y, H-1-x). */
static void rgb_rot_cw(const uint8_t *src, uint32_t sstride, int sw, int sh, uint8_t *dst,
                       uint32_t dstride) {
    for (int y = 0; y < sw; y++) {
        uint8_t *drow = dst + (size_t)y * dstride;
        const uint8_t *col = src + (size_t)y * 3;
        for (int x = 0; x < sh; x++) {
            const uint8_t *p = col + (size_t)(sh - 1 - x) * sstride;
            drow[x * 3 + 0] = p[0];
            drow[x * 3 + 1] = p[1];
            drow[x * 3 + 2] = p[2];
        }
    }
}

/* Поворот на 90° против часовой: dst(x,y) = src(W-1-y, x). */
static void rgb_rot_ccw(const uint8_t *src, uint32_t sstride, int sw, int sh, uint8_t *dst,
                        uint32_t dstride) {
    for (int y = 0; y < sw; y++) {
        uint8_t *drow = dst + (size_t)y * dstride;
        const uint8_t *col = src + (size_t)(sw - 1 - y) * 3;
        for (int x = 0; x < sh; x++) {
            const uint8_t *p = col + (size_t)x * sstride;
            drow[x * 3 + 0] = p[0];
            drow[x * 3 + 1] = p[1];
            drow[x * 3 + 2] = p[2];
        }
    }
}

/* Поворот на 180°. */
static void rgb_rot_180(const uint8_t *src, uint32_t sstride, int sw, int sh, uint8_t *dst,
                        uint32_t dstride) {
    for (int y = 0; y < sh; y++) {
        uint8_t *drow = dst + (size_t)y * dstride;
        const uint8_t *srow = src + (size_t)(sh - 1 - y) * sstride;
        for (int x = 0; x < sw; x++) {
            const uint8_t *p = srow + (size_t)(sw - 1 - x) * 3;
            drow[x * 3 + 0] = p[0];
            drow[x * 3 + 1] = p[1];
            drow[x * 3 + 2] = p[2];
        }
    }
}

/* Кадр для TDL: повёрнутый (если задан поворот) в собственном VB-пуле.
   Кадр держится до vpss_face_release_rot(). */
int vpss_face_get_rot(VIDEO_FRAME_INFO_S *fi, int timeout_ms) {
    if (!face_open_flag || fi == NULL) return -1;
    VIDEO_FRAME_INFO_S src;
    if (vpss_face_get(&src, timeout_ms) != 0) return -1;
    if (g_rot_angle == 0) {
        *fi = src;
        return 0;
    }

    const VIDEO_FRAME_S *sf = &src.stVFrame;
    const uint32_t slen = sf->u32Length[0] ? sf->u32Length[0] : sf->u32Stride[0] * sf->u32Height;
    uint8_t *sptr = sf->pu8VirAddr[0];
    void *mapped = NULL;
    if (sptr == NULL) {
        mapped = CVI_SYS_MmapCache(sf->u64PhyAddr[0], slen);
        if (mapped == NULL) {
            vpss_face_release(&src);
            return -1;
        }
        sptr = (uint8_t *)mapped;
    } else {
        CVI_SYS_IonInvalidateCache(sf->u64PhyAddr[0], sptr, slen);
    }

    VB_BLK blk = CVI_VB_GetBlock(rot_pool, rot_blk_size);
    if (blk == (VB_BLK)-1) {
        printf("vpss_face: VB_GetBlock не удался (pool=%d size=%u)\n", (int)rot_pool,
               (unsigned)rot_blk_size);
        fflush(stdout);
        if (mapped != NULL) CVI_SYS_Munmap(mapped, slen);
        vpss_face_release(&src);
        return -1;
    }
    const uint64_t dphy = CVI_VB_Handle2PhysAddr(blk);
    void *dvir = CVI_SYS_MmapCache(dphy, rot_blk_size);
    if (dvir == NULL) {
        printf("vpss_face: mmap блока поворота не удался (phy=0x%llx)\n",
               (unsigned long long)dphy);
        fflush(stdout);
        CVI_VB_ReleaseBlock(blk);
        if (mapped != NULL) CVI_SYS_Munmap(mapped, slen);
        vpss_face_release(&src);
        return -1;
    }
    uint8_t *dptr = (uint8_t *)dvir;
    const uint32_t sstride = sf->u32Stride[0];
    if (g_rot_angle == 90)
        rgb_rot_cw(sptr, sstride, g_src_w, g_src_h, dptr, g_dst_stride);
    else if (g_rot_angle == 180)
        rgb_rot_180(sptr, sstride, g_src_w, g_src_h, dptr, g_dst_stride);
    else
        rgb_rot_ccw(sptr, sstride, g_src_w, g_src_h, dptr, g_dst_stride);

    CVI_SYS_IonFlushCache(dphy, dptr, g_dst_stride * (uint32_t)g_dst_h);

    memset(fi, 0, sizeof(*fi));
    fi->stVFrame.u64PhyAddr[0] = dphy;
    fi->stVFrame.pu8VirAddr[0] = dptr;
    fi->stVFrame.u32Width = (CVI_U32)g_dst_w;
    fi->stVFrame.u32Height = (CVI_U32)g_dst_h;
    fi->stVFrame.u32Stride[0] = g_dst_stride;
    fi->stVFrame.u32Length[0] = g_dst_stride * (uint32_t)g_dst_h;
    fi->stVFrame.enPixelFormat = PIXEL_FORMAT_RGB_888;
    fi->stVFrame.enVideoFormat = VIDEO_FORMAT_LINEAR;
    fi->u32PoolId = rot_pool;
    fi->stVFrame.pPrivateData = (void *)(uintptr_t)blk;

    if (mapped != NULL) CVI_SYS_Munmap(mapped, slen);
    vpss_face_release(&src);
    return 0;
}

void vpss_face_release_rot(VIDEO_FRAME_INFO_S *fi) {
    if (fi == NULL) return;
    VB_BLK blk = (VB_BLK)(uintptr_t)fi->stVFrame.pPrivateData;
    if (blk != (VB_BLK)-1 && blk != 0) {
        if (fi->stVFrame.pu8VirAddr[0] != NULL) {
            CVI_SYS_Munmap(fi->stVFrame.pu8VirAddr[0], rot_blk_size);
            fi->stVFrame.pu8VirAddr[0] = NULL;
        }
        CVI_VB_ReleaseBlock(blk);
        fi->stVFrame.pPrivateData = NULL;
    }
}

void vpss_face_release(VIDEO_FRAME_INFO_S *fi) {
    if (face_open_flag && fi != NULL) p_ReleaseChnFrame(FACE_GRP, FACE_CHN, fi);
}

int vpss_face_selftest(void) {
    void *lib = vpss_lib;
    if (lib == NULL) lib = vpu_dlopen(RTLD_NOW | RTLD_LOCAL);
    if (lib == NULL) {
        printf("selftest: libvpu не загрузилась\n");
        return -1;
    }
    typedef VPSS_GRP (*fn_avail_t)(void);
    typedef CVI_S32 (*fn_create_t)(VPSS_GRP, const VPSS_GRP_ATTR_S *);
    typedef CVI_S32 (*fn_destroy_t)(VPSS_GRP);
    fn_avail_t p_avail = (fn_avail_t)dlsym(lib, "CVI_VPSS_GetAvailableGrp");
    fn_create_t p_create = (fn_create_t)dlsym(lib, "CVI_VPSS_CreateGrp");
    fn_destroy_t p_destroy_grp = (fn_destroy_t)dlsym(lib, "CVI_VPSS_DestroyGrp");
    printf("selftest: available grp=%d\n", (p_avail != NULL) ? (int)p_avail() : -999);
    if (p_create == NULL) return -1;
    VPSS_GRP_ATTR_S g;
    memset(&g, 0, sizeof(g));
    g.u32MaxW = 960;
    g.u32MaxH = 540;
    g.enPixelFormat = PIXEL_FORMAT_RGB_888;
    for (int grp = 1; grp <= 7; grp++) {
        CVI_S32 rc = p_create(grp, &g);
        printf("selftest: CreateGrp(%d) 960x540 RGB rc=0x%x\n", grp, (unsigned)rc);
        if (rc == CVI_SUCCESS && p_destroy_grp != NULL) p_destroy_grp(grp);
    }
    VPSS_GRP_ATTR_S g2;
    memset(&g2, 0, sizeof(g2));
    g2.u32MaxW = 100;
    g2.u32MaxH = 100;
    g2.enPixelFormat = PIXEL_FORMAT_NV21;
    {
        CVI_S32 rc = p_create(2, &g2);
        printf("selftest: CreateGrp(2) 100x100 NV21 rc=0x%x\n", (unsigned)rc);
        if (rc == CVI_SUCCESS && p_destroy_grp != NULL) p_destroy_grp(2);
    }
    fflush(stdout);
    return 0;
}

/* Остановить группу и её канал (нужно перед DestroyGrp, иначе он виснет). */
int vpss_face_stop_grp(int grp) {
    void *lib = vpss_lib;
    if (lib == NULL) lib = vpu_dlopen(RTLD_NOW | RTLD_LOCAL);
    if (lib == NULL) return -1;
    typedef CVI_S32 (*fn_stop_t)(VPSS_GRP);
    typedef CVI_S32 (*fn_dis_t)(VPSS_GRP, VPSS_CHN);
    fn_stop_t p_stop = (fn_stop_t)dlsym(lib, "CVI_VPSS_StopGrp");
    fn_dis_t p_dis = (fn_dis_t)dlsym(lib, "CVI_VPSS_DisableChn");
    int rc = 0;
    if (p_dis != NULL) rc |= p_dis((VPSS_GRP)grp, VPSS_CHN0);
    if (p_stop != NULL) rc |= p_stop((VPSS_GRP)grp);
    return rc;
}

/* Уничтожить VPSS-группу (TDL не делает этого при DestroyHandle — группа и её
   VB-пул остаются в драйвере, и следующий запуск упирается в нехватку памяти). */
int vpss_face_destroy_grp(int grp) {
    void *lib = vpss_lib;
    if (lib == NULL) lib = vpu_dlopen(RTLD_NOW | RTLD_LOCAL);
    if (lib == NULL) return -1;
    typedef CVI_S32 (*fn_destroy_t)(VPSS_GRP);
    fn_destroy_t p_destroy_grp = (fn_destroy_t)dlsym(lib, "CVI_VPSS_DestroyGrp");
    if (p_destroy_grp == NULL) return -1;
    return (int)p_destroy_grp((VPSS_GRP)grp);
}

/* Группа свободна, если её можно создать (у занятой CreateGrp даёт EXIST). */
int vpss_face_group_free(int grp) {
    void *lib = vpss_lib;
    if (lib == NULL) lib = vpu_dlopen(RTLD_NOW | RTLD_LOCAL);
    if (lib == NULL) return 0;
    typedef CVI_S32 (*fn_create_t)(VPSS_GRP, const VPSS_GRP_ATTR_S *);
    typedef CVI_S32 (*fn_destroy_t)(VPSS_GRP);
    fn_create_t p_create = (fn_create_t)dlsym(lib, "CVI_VPSS_CreateGrp");
    fn_destroy_t p_destroy_grp = (fn_destroy_t)dlsym(lib, "CVI_VPSS_DestroyGrp");
    if (p_create == NULL) return 0;
    VPSS_GRP_ATTR_S g;
    memset(&g, 0, sizeof(g));
    g.u32MaxW = 960;
    g.u32MaxH = 540;
    g.enPixelFormat = PIXEL_FORMAT_NV21;
    CVI_S32 rc = p_create((VPSS_GRP)grp, &g);
    if (rc != CVI_SUCCESS) return 0;
    if (p_destroy_grp != NULL) p_destroy_grp((VPSS_GRP)grp);
    return 1;
}

int vpss_face_available_grp(void) {
    void *lib = vpss_lib;
    if (lib == NULL) lib = vpu_dlopen(RTLD_NOW | RTLD_LOCAL);
    if (lib == NULL) return -1;
    typedef VPSS_GRP (*fn_avail_t)(void);
    fn_avail_t p_avail = (fn_avail_t)dlsym(lib, "CVI_VPSS_GetAvailableGrp");
    if (p_avail == NULL) return -1;
    return (int)p_avail();
}

void vpss_face_close(void) {
    if (!face_open_flag) return;
    drain_run = 0;
    pthread_cond_broadcast(&slot_cv);
    pthread_join(drain_thread, NULL);

    pthread_mutex_lock(&slot_mx);
    if (slot_have) {
        p_ReleaseChnFrame(FACE_GRP, FACE_CHN, &slot_fi);
        slot_have = 0;
    }
    pthread_mutex_unlock(&slot_mx);

    p_DetachVbPool(FACE_GRP, FACE_CHN);
    p_DisableChn(FACE_GRP, FACE_CHN);
    CVI_VB_DestroyPool(face_pool);
    face_pool = VB_INVALID_POOLID;
    if (rot_pool != VB_INVALID_POOLID) {
        CVI_VB_DestroyPool(rot_pool);
        rot_pool = VB_INVALID_POOLID;
    }
    g_rot_angle = 0;
    face_open_flag = 0;
}
