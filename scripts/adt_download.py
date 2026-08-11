"""ADT (Aria Digital Twin) selective fetch: multi-person sequences, GT only.

Input: data/adt/ADT_download_urls.json -- the signed-URL file Meta gives you
after accepting the dataset license at projectaria.com/datasets/adt (see
docs/DATA_SETUP.md). We deliberately fetch ONLY `main_groundtruth` for the
`multiskeleton` (two-person) sequences: ~3.8GB instead of the full ~3.5TB.
VRS video, depth and segmentation are never downloaded -- our pipeline reads
object poses and 2D visibility from the GT CSVs, no pixels involved.
"""
import argparse
import json
import os
import time
import urllib.request
import zipfile

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "adt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", default=os.path.join(ROOT, "ADT_download_urls.json"))
    ap.add_argument("--match", default="multiskeleton",
                    help="sequence-name filter (default: the two-person sessions)")
    ap.add_argument("--part", default="main_groundtruth")
    ap.add_argument("--out", default=os.path.join(ROOT, "gt"))
    ap.add_argument("--keep-zips", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.urls):
        raise SystemExit(
            "%s 가 없습니다. projectaria.com/datasets/adt 에서 라이선스에 동의하고\n"
            "받은 링크 파일을 그 경로에 저장하세요 (docs/DATA_SETUP.md)." % args.urls)

    seqs = json.load(open(args.urls))["sequences"]
    picked = {n: v[args.part] for n, v in seqs.items()
              if args.match in n and args.part in v}
    total = sum(e["file_size_bytes"] for e in picked.values())
    print("%d sequences matched '%s'  (%.2f GB, %s only)"
          % (len(picked), args.match, total / 1e9, args.part), flush=True)
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    for i, (name, e) in enumerate(sorted(picked.items())):
        zpath = os.path.join(args.out, e["filename"])
        # sequence dir name: strip the ADT_/…_main_groundtruth.zip wrapper
        seq_dir = os.path.join(args.out, name)
        if os.path.isdir(seq_dir) and os.listdir(seq_dir):
            continue
        if not (os.path.exists(zpath) and os.path.getsize(zpath) == e["file_size_bytes"]):
            urllib.request.urlretrieve(e["download_url"], zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(seq_dir)
        if not args.keep_zips:
            os.remove(zpath)
        print("[%d/%d] %s  %.0f MB  (%.0fs)"
              % (i + 1, len(picked), name, e["file_size_bytes"] / 1e6, time.time() - t0),
              flush=True)
    print("ADT_GT_READY -> %s" % args.out)


if __name__ == "__main__":
    main()
