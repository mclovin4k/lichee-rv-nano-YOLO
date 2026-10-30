# TPU hang fix: `soph_tpu.ko`

A patcher for the stock LicheeRV Nano kernel module (SG2002 / CV181x). It
removes the minute-long TPU stalls that keep the MJPEG stream running while
the detector boxes freeze.

The repository does **not** contain a ready-made `.ko`: the driver is
distributed under the GPL, and its sources are not available in the image.
The patcher only changes constants in the module already installed on your
board and does not require publishing a binary.

## Symptoms

- `cvi-tpu c100000.tpu: run dmabuf timeout` repeats in `dmesg`;
- the `stream_yolo` log shows `wait dmabuf failed[-110]` (`-ETIMEDOUT`) and
  `run cmdbuf failed:2`;
- inference takes seconds instead of 50–80 ms, and the detector returns stale
  boxes;
- the frame shows `no det Ns` even though the stream is running;
- `kill -9` does not kill a process stuck in the TPU ioctl until the driver
  timeout expires; sometimes only a reboot helps.

The problem is not in the model: it occurs with different YOLO models and with
other applications that use the TPU.

## Cause

In `soph_tpu.ko`, five command-completion waits
(`wait_for_completion_interruptible_timeout`) use a timeout of **15000 ticks**.
With `HZ=250` this is **60 seconds**. If the TPU does not send an interrupt
(a lost dmabuf fence, a scheduler failure), the calling thread blocks for a
minute. The command queue behind it stalls, and the application can neither
receive an error nor recreate the handle.

## What the patch does

The five constants are replaced with **250 ticks = 1 second**. The driver
logic does not change — only the maximum wait time is reduced. After a
failure, `stream_yolo` receives an error in about a second, unloads and
re-registers the model, and the stream keeps running.

| Site | Offset | Before | After |
|---|---|---|---|
| 1 | 7568 | `91 65 93 85 85 a9` | `81 45 93 85 a5 0f` |
| 2 | 7940 | `91 65 93 85 85 a9` | `81 45 93 85 a5 0f` |
| 3 | 8184 | `91 67 93 87 87 a9` | `81 47 93 87 a7 0f` |
| 4 | 8658 | `91 67 93 87 87 a9` | `81 47 93 87 a7 0f` |
| 5 | 9934 | `91 65 93 85 85 a9` | `81 45 93 85 a5 0f` |

Module checksums (size 70536 bytes):

| State | md5 |
|---|---|
| stock | `bc70f809190c0e41aeb2d3863da4581d` |
| patched | `3c0545f7a42b233987c9e52ebc56affd` |

The patcher first verifies all five sites and the md5. If the module has a
different size, a different version, or the bytes do not match, it refuses to
change anything.

## Installation

Copy the directory to the board and run the patcher (by default it uses
`/mnt/system/ko/soph_tpu.ko`):

```sh
scp -r tpu-fix root@<IP>:/root/
ssh root@<IP> 'sh /root/tpu-fix/patch_tpu.sh'
ssh root@<IP> reboot
```

If the module is located elsewhere:

```sh
ssh root@<IP> 'sh /root/tpu-fix/patch_tpu.sh /path/to/soph_tpu.ko'
```

The patcher creates a `soph_tpu.ko.orig` backup next to the module. Running it
again on an already patched file does nothing.

## Verification

```sh
ssh root@<IP> 'md5sum /mnt/system/ko/soph_tpu.ko'
# 3c0545f7a42b233987c9e52ebc56affd

ssh root@<IP> 'lsmod | grep soph_tpu'
ssh root@<IP> 'dmesg | grep -i "soph_tpu\|tpu"'
```

After starting the stream, leave it under load for a few minutes. When
`run dmabuf timeout` appears, detection should recover within 1–2 s instead of
freezing for tens of seconds.

## Rollback

```sh
ssh root@<IP> 'cp /mnt/system/ko/soph_tpu.ko.orig /mnt/system/ko/soph_tpu.ko'
ssh root@<IP> 'sync && reboot'
```

If there is no backup, take the original module from a fresh LicheeRV Nano
image.

## Compatibility and limitations

- Tested on the stock LicheeRV Nano image with kernel
  `5.10.260-tag-20260927-g3ac35f59d279` (`HZ=250`, module 70536 bytes).
- On a different kernel or driver version the patcher refuses to work: do not
  try to bypass the check.
- This is a **mitigation**, not a fix for the root cause in the
  hardware/driver. `run dmabuf timeout` messages may remain in `dmesg`, but
  the minute-long stalls are gone.
- The camera close hang (`vb_qbuf`, state `D`) is not fixed by the patch. If
  the process does not die, a `reboot` is required.
- The module is a GPL component from Cvitek/Sophgo; its `strings` contain
  `license=GPL` and the author `Wellken Chen`. The driver sources are not
  available in the image, so we distribute neither the original nor the
  patched `.ko`, only the patcher. If you publish a modified module, comply
  with the GPL and include the corresponding source code.
