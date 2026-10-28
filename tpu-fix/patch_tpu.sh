#!/bin/sh
# patch_tpu.sh -- reduce the command-completion timeout in soph_tpu.ko.
#
# The stock module waits 15000 ticks (60 s at HZ=250) in five places.  This
# script rewrites those five constants to 250 ticks (1 s).  It modifies the
# module that is already on your board; no GPL binary is redistributed.
#
#   sh patch_tpu.sh                    # /mnt/system/ko/soph_tpu.ko
#   sh patch_tpu.sh /path/to/soph_tpu.ko
#
# The original file is saved next to it as soph_tpu.ko.orig.  A reboot is
# required because the module is loaded at boot.
set -eu

KO=${1:-/mnt/system/ko/soph_tpu.ko}
ORIG_MD5=bc70f809190c0e41aeb2d3863da4581d
PATCHED_MD5=3c0545f7a42b233987c9e52ebc56affd
BACKUP="$KO.orig"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[ -f "$KO" ] || die "module not found: $KO"
command -v md5sum >/dev/null 2>&1 || die "md5sum not found"
command -v od >/dev/null 2>&1 || die "od not found"
command -v dd >/dev/null 2>&1 || die "dd not found"

md5_of() { md5sum "$1" | awk '{print $1}'; }
bytes_at() { od -An -tx1 -j "$2" -N 6 "$1" | tr -d ' \n'; }

check_bytes() { # check_bytes <offset> <expected hex> <label>
    got=$(bytes_at "$KO" "$1")
    if [ "$got" != "$2" ]; then
        printf 'ERROR: unexpected bytes at offset %s (%s):\n' "$1" "$3" >&2
        printf '  expected: %s\n  got:      %s\n' "$2" "$got" >&2
        exit 1
    fi
}

put6() { # put6 <offset> <printf-escaped 6 bytes>
    printf "$2" | dd of="$KO" bs=1 seek="$1" conv=notrunc 2>/dev/null || \
        die "failed to write at offset $1"
}

now=$(md5_of "$KO")
if [ "$now" = "$PATCHED_MD5" ]; then
    echo "already patched: $KO ($PATCHED_MD5)"
    exit 0
fi

if [ "$now" = "$ORIG_MD5" ]; then
    echo "original module detected: $KO ($ORIG_MD5)"
else
    echo "WARNING: md5 $now is neither the known original nor the known patched module."
    echo "         Checking the five known instruction sites before touching anything."
fi

# Five wait_for_completion_interruptible_timeout() call sites:
#   15000 ticks (60 s) -> 250 ticks (1 s)
#   lui/addi sequences: 91 65 93 85 85 a9 -> 81 45 93 85 a5 0f
#                       91 67 93 87 87 a9 -> 81 47 93 87 a7 0f
check_bytes 7568 9165938585a9 "site 1"
check_bytes 7940 9165938585a9 "site 2"
check_bytes 8184 9167938787a9 "site 3"
check_bytes 8658 9167938787a9 "site 4"
check_bytes 9934 9165938585a9 "site 5"

if [ ! -f "$BACKUP" ]; then
    cp -p "$KO" "$BACKUP" || die "cannot create backup: $BACKUP"
    echo "backup: $BACKUP"
else
    echo "backup exists, keeping it: $BACKUP"
fi

put6 7568 '\201\105\223\205\245\017'
put6 7940 '\201\105\223\205\245\017'
put6 8184 '\201\107\223\207\247\017'
put6 8658 '\201\107\223\207\247\017'
put6 9934 '\201\105\223\205\245\017'
sync

new=$(md5_of "$KO")
if [ "$new" != "$PATCHED_MD5" ]; then
    printf 'ERROR: patched md5 is %s, expected %s\n' "$new" "$PATCHED_MD5" >&2
    printf 'Restoring from %s\n' "$BACKUP" >&2
    cp -p "$BACKUP" "$KO"
    exit 1
fi

cat <<EOF

Patched: $KO
  before: $now
  after:  $new

The module is loaded at boot.  Reboot the board, then check:
  md5sum /mnt/system/ko/soph_tpu.ko    # $PATCHED_MD5
  lsmod | grep soph_tpu
  dmesg | grep -i soph_tpu

Rollback:
  cp $BACKUP $KO && sync && reboot
EOF
