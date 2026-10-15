/* Hardware JPEG encoder (VENC, PT_JPEG payload) fed from a VPSS frame.

   Why VENC: cv::imencode on the single 1 GHz C906 takes 150-300 ms per
   640x640 frame and starved the TPU driver; VENC does the same work in
   hardware in a few ms.

   Why no staging buffer: the VPSS channel already emits NV21 in a VB pool
   with physical addresses, the NPU preprocess and the box drawing work on
   that same buffer, so the encoder can consume it directly. The only CPU
   obligation is flushing caches for the drawn region (done by the caller)
   before SendFrame. */

#include "venc_jpeg.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cvi_venc.h>

static int g_chn = 2; /* VENC channel; nobody else in this process uses VENC */
static int g_inited = 0;

int venc_jpeg_init(int width, int height, int quality) {
  if (g_inited) return 0;
  if (width <= 0 || height <= 0) return -1;
  if (quality < 1) quality = 75;
  if (quality > 99) quality = 99;

  VENC_CHN_ATTR_S attr;
  memset(&attr, 0, sizeof(attr));
  attr.stVencAttr.enType = PT_JPEG;
  attr.stVencAttr.u32MaxPicWidth = (CVI_U32)width;
  attr.stVencAttr.u32MaxPicHeight = (CVI_U32)height;
  attr.stVencAttr.u32PicWidth = (CVI_U32)width;
  attr.stVencAttr.u32PicHeight = (CVI_U32)height;
  attr.stVencAttr.u32BufSize = 1024 * 1024;
  attr.stVencAttr.bByFrame = CVI_TRUE;
  attr.stVencAttr.stAttrJpege.enReceiveMode = VENC_PIC_RECEIVE_MULTI;

  CVI_S32 ret = CVI_VENC_CreateChn(g_chn, &attr);
  if (ret != CVI_SUCCESS) {
    fprintf(stderr, "venc_jpeg: CreateChn failed %#x\n", ret);
    return -1;
  }

  VENC_JPEG_PARAM_S jp;
  memset(&jp, 0, sizeof(jp));
  jp.u32Qfactor = (CVI_U32)quality;
  ret = CVI_VENC_SetJpegParam(g_chn, &jp);
  if (ret != CVI_SUCCESS)
    fprintf(stderr, "venc_jpeg: SetJpegParam(%d) failed %#x\n", quality, ret);

  VENC_RECV_PIC_PARAM_S recv;
  memset(&recv, 0, sizeof(recv));
  recv.s32RecvPicNum = -1;
  ret = CVI_VENC_StartRecvFrame(g_chn, &recv);
  if (ret != CVI_SUCCESS) {
    fprintf(stderr, "venc_jpeg: StartRecvFrame failed %#x\n", ret);
    CVI_VENC_DestroyChn(g_chn);
    return -1;
  }

  g_inited = 1;
  printf("venc_jpeg: %dx%d q=%d chn=%d (VENC, frame-fed)\n", width, height, quality, g_chn);
  return 0;
}

int venc_jpeg_encode_frame(const VIDEO_FRAME_INFO_S *vf, uint8_t *out, uint32_t cap, uint32_t *len) {
  if (!g_inited || vf == NULL || out == NULL || len == NULL) return -1;
  *len = 0;

  CVI_S32 ret = CVI_VENC_SendFrame(g_chn, vf, 40);
  if (ret != CVI_SUCCESS) return -1; /* busy or rejected: skip this frame */

  VENC_CHN_STATUS_S status;
  memset(&status, 0, sizeof(status));
  ret = CVI_VENC_QueryStatus(g_chn, &status);
  if (ret != CVI_SUCCESS || status.u32CurPacks == 0) return -1;

  VENC_STREAM_S stream;
  memset(&stream, 0, sizeof(stream));
  stream.pstPack = (VENC_PACK_S *)calloc(status.u32CurPacks, sizeof(VENC_PACK_S));
  if (stream.pstPack == NULL) return -1;

  ret = CVI_VENC_GetStream(g_chn, &stream, 100);
  if (ret != CVI_SUCCESS) {
    free(stream.pstPack);
    return -1;
  }

  uint32_t total = 0;
  for (CVI_U32 i = 0; i < stream.u32PackCount; i++) {
    VENC_PACK_S *p = &stream.pstPack[i];
    uint32_t n = (p->u32Len > p->u32Offset) ? (p->u32Len - p->u32Offset) : 0;
    if (n == 0 || p->pu8Addr == NULL) continue;
    if (total + n > cap) {
      total = 0; /* does not fit: drop whole frame, keep stream in sync */
      break;
    }
    memcpy(out + total, p->pu8Addr + p->u32Offset, n);
    total += n;
  }

  CVI_VENC_ReleaseStream(g_chn, &stream);
  free(stream.pstPack);

  if (total == 0) return -1;
  *len = total;
  return 0;
}

void venc_jpeg_deinit(void) {
  if (!g_inited) return;
  CVI_VENC_StopRecvFrame(g_chn);
  CVI_VENC_DestroyChn(g_chn);
  g_inited = 0;
}
