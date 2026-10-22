#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -uo pipefail
output=$1
platform=$2
devices=$3
mkdir -p "$(dirname "$output")"
mkdir "$output" || exit 1
output=$(cd "$output" && pwd)
raw=$(mktemp -d "$output/../../so-cache-export-raw.XXXXXX") || exit 1
privilege=()
if [ "$(id -u)" -ne 0 ]; then privilege=(sudo -n); fi
status=0
python - "$platform" "$output" > "$output/provenance.log" 2>&1 <<'PYPROBE'
import hashlib
import os
import shutil
import sys
from pathlib import Path
import simpler
from simpler_setup.runtime_builder import RuntimeBuilder, parse_platform
print("python", sys.executable)
print("simpler", simpler.__file__)
print("cann", os.environ.get("ASCEND_HOME_PATH"))
arch, variant = parse_platform(sys.argv[1])
root = RuntimeBuilder._LIB_DIR
for directory in (root / arch / variant / "tensormap_and_ringbuffer", root / arch / "dispatcher"):
    for path in sorted(directory.iterdir()) if directory.is_dir() else ():
        if path.is_file():
            data = path.read_bytes()
            print("binary", path, hashlib.sha256(data).hexdigest(), "audit_marker", b"SO_CACHE_AUDIT_2651" in data)
            if path.suffix == ".so":
                destination = Path(sys.argv[2]) / "binaries"
                destination.mkdir(exist_ok=True)
                shutil.copyfile(path, destination / (directory.name + "-" + path.name))
for filename in ("version.cfg", "version.info"):
    path = Path(os.environ.get("ASCEND_HOME_PATH", "/nonexistent")) / filename
    if path.is_file():
        print(filename, path.read_text())
PYPROBE
[ "$?" = 0 ] || status=1
IFS=',' read -r -a device_ids <<< "$devices"
for device in "${device_ids[@]}"; do
  if [[ ! "$device" =~ ^[0-9]+$ ]]; then status=1; continue; fi
  (
    cd "$raw" || exit 1
    timeout 120 "${privilege[@]}" /usr/local/Ascend/driver/tools/msnpureport \
      report --type 1 --device "$device" --print 1
  ) > "$output/export-$device.log" 2>&1
  result=$?
  printf 'device=%s export_rc=%s\n' "$device" "$result" | tee -a "$output/status.log"
  [ "$result" = 0 ] || status=1
done
"${privilege[@]}" tar -czf - -C "$raw" . > "$output/device-system.tar.gz"
archive_result=$?
printf 'archive_rc=%s\n' "$archive_result" >> "$output/status.log"
[ "$archive_result" = 0 ] || status=1
if [ "$archive_result" = 0 ]; then
  "${privilege[@]}" rm -rf -- "$raw"
  cleanup_result=$?
  printf 'export_copy_cleanup_rc=%s\n' "$cleanup_result" >> "$output/status.log"
  [ "$cleanup_result" = 0 ] || status=1
fi
printf 'status=%s\n' "$status" >> "$output/status.log"
exit "$status"
