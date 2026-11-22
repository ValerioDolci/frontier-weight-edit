#!/bin/bash
# CPU: edited checkpoint (edit_checkpoint.py) -> capy-flash artifact with the official converter of the capy-flash
# release v0.16.0 (tag v0.16.0), components text,mtp like the published artifact.
#
# Usage: build_artifact.sh NAME EDITS.json OUT_DIR
#   -> OUT_DIR/NAME/{ckpt/, NAME.ninfer, edit.log, convert.log}
# Environment:
#   CAPY_FLASH_SRC  checkout of the capy-flash repository (needed to get `tools/` at tag v0.16.0)
#   PYTHON          python interpreter with the converter's dependencies (default: python3)
#   INKLING_CKPT    original checkpoint directory (default: models/inkling-2-mini)
#   TOOLS_DIR       where `tools/` is extracted (default: /tmp/inkedit-tools)
set -eu
NAME=$1; EDITS=$2; OUT=$3
: "${CAPY_FLASH_SRC:?set CAPY_FLASH_SRC to a capy-flash checkout}"
PY=${PYTHON:-python3}
CKPT=$(cd "${INKLING_CKPT:-models/inkling-2-mini}" && pwd)
SRC=${TOOLS_DIR:-/tmp/inkedit-tools}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/$NAME"
D=$(cd "$OUT/$NAME" && pwd)   # absolute: the converter runs from $SRC
[ -d "$SRC/tools" ] || { mkdir -p "$SRC"; git -C "$CAPY_FLASH_SRC" archive v0.16.0 tools | tar -x -C "$SRC"; }
"$PY" "$HERE/edit_checkpoint.py" --src "$CKPT" --dst "$D/ckpt" --edits "$EDITS" > "$D/edit.log" 2>&1
tail -2 "$D/edit.log"
cd "$SRC"
"$PY" -B -m tools.convert --model "$D/ckpt" --recipe inkling2_mini --components text,mtp \
  --device cpu --name "inkling-2-mini-$NAME" --out "$D/$NAME.ninfer" > "$D/convert.log" 2>&1
tail -2 "$D/convert.log"
ls -l "$D/$NAME.ninfer"
