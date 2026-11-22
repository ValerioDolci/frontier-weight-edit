#!/usr/bin/env python3
"""(CPU) Copy of the Inkling 2 Mini checkpoint with expert neurons scaled on the "up" row of w13; the capy-flash artifact is
then rebuilt with the official converter (`tools.convert --recipe inkling2_mini --components text,mtp`).

Neuron (layer L, expert e, neuron j) = row 2j+1 of `model.llm.layers.{L}.mlp.experts.w13_weight[e]` (row 2j = gate;
mapping verified bit-exactly in step 2, G4). Exact edits (no requantization):
  alpha = 0    NVFP4: the row's bytes set to 0 (E2M1 code 0 = +0) · MXFP8: the row's FP8 values set to 0
  alpha = 2^k  NVFP4: the row's E4M3 scales x 2^k (refused if a scale would exceed 448) · MXFP8: UE8M0 exponent +k
The copied checkpoint is identical to the original except for the touched bytes (checked at the end of the write).
mtp.safetensors and the other files are symlinks to the original.
Usage: edit_checkpoint.py [--src models/inkling-2-mini] --dst DIR --edits edits.json   (edits: [[L, e, j, alpha], ...])
  --src defaults to $INKLING_CKPT or models/inkling-2-mini.
"""
import argparse, json, math, os, struct, shutil

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=os.environ.get("INKLING_CKPT", "models/inkling-2-mini"))
ap.add_argument("--dst", required=True)
ap.add_argument("--edits", required=True)
A = ap.parse_args()
A.src = os.path.abspath(A.src)   # the symlinks to the original files must be absolute
edits = json.load(open(A.edits))
ST = "model-00001-of-00001.safetensors"
os.makedirs(A.dst, exist_ok=True)
for f in os.listdir(A.src):
    if f != ST and not os.path.exists(os.path.join(A.dst, f)):
        s = os.path.join(A.src, f)
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(A.dst, f))
        else:
            os.symlink(s, os.path.join(A.dst, f))

src, dst = os.path.join(A.src, ST), os.path.join(A.dst, ST)
with open(src, "rb") as f:
    hlen = struct.unpack("<Q", f.read(8))[0]
    H = json.loads(f.read(hlen))
base = 8 + hlen
if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
    shutil.copyfile(src, dst)
ITEM = {"U8": 1, "F8_E4M3": 1, "F32": 4, "BF16": 2}


def view(name):
    t = H[name]
    shape, (a, b) = t["shape"], t["data_offsets"]
    return t["dtype"], shape, base + a, b - a


changed = []
with open(dst, "r+b") as f:
    for L, e, j, alfa in edits:
        pre = f"model.llm.layers.{L}.mlp.experts."
        dt_w, shp_w, off_w, _ = view(pre + "w13_weight")
        dt_s, shp_s, off_s, _ = view(pre + "w13_weight_scale")
        r = 2 * j + 1
        assert shp_w[1] == 2304 and r < shp_w[1], (L, shp_w)
        row_w = shp_w[2] * ITEM[dt_w]
        row_s = shp_s[2] * ITEM[dt_s]
        pw = off_w + (e * shp_w[1] + r) * row_w
        ps = off_s + (e * shp_s[1] + r) * row_s
        nvfp4 = dt_w == "U8"
        if alfa == 0:
            f.seek(pw)
            f.write(b"\x00" * row_w)
            changed.append((pw, row_w))
            continue
        k = math.log2(alfa)
        assert k == int(k) and k > 0, f"alpha {alfa}: only 0 or powers of 2 > 1 (exact edits)"
        k = int(k)
        f.seek(ps)
        sc = bytearray(f.read(row_s))
        if nvfp4:   # E4M3: value = 2^(E-7)*(1+M/8); x2^k = E+k if E>0 and the result stays <= 448 (E <= 15, not NaN)
            for i, b in enumerate(sc):
                E, M = (b >> 3) & 0xF, b & 7
                if b & 0x80 or (E == 0 and M == 0):
                    continue
                assert E > 0, f"subnormal scale L{L} e{e} j{j}: x2^k is not exact"
                assert E + k <= 15 and not (E + k == 15 and M == 7), f"scale above 448 L{L} e{e} j{j}"
                sc[i] = (b & 0x87) | ((E + k) << 3)
        else:       # UE8M0: pure exponent
            for i, b in enumerate(sc):
                assert b + k < 255, f"exponent beyond the limit L{L} e{e} j{j}"
                sc[i] = b + k
        f.seek(ps)
        f.write(bytes(sc))
        changed.append((ps, row_s))

# check: outside the touched bytes the copy is identical to the original (compared in 64 MB blocks)
spans = sorted(changed)
diff_out = 0
with open(src, "rb") as a, open(dst, "rb") as b:
    pos, CH = 0, 64 << 20
    while True:
        x, y = a.read(CH), b.read(CH)
        if not x:
            break
        if x != y:
            ax, ay = np.frombuffer(x, np.uint8), np.frombuffer(y, np.uint8)
            for i in np.nonzero(ax != ay)[0]:
                p = pos + int(i)
                if not any(s <= p < s + n for s, n in spans):
                    diff_out += 1
        pos += len(x)
print(json.dumps({"edits": len(edits), "byte_spans": len(spans), "bytes_touched": sum(n for _, n in spans),
                  "diff_outside_edits": diff_out, "dst": dst}))
assert diff_out == 0
