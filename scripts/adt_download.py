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
    ap.add_argument("--part", default="main_groundtruth",
                    help="comma-separated: e.g. main_groundtruth,video_main_rgb,segmentation")
    ap.add_argument("--out", default=os.path.join(ROOT, "gt"))
    ap.add_argument("--keep-zips", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.urls):
        raise SystemExit(
            "%s 가 없습니다. projectaria.com/datasets/adt 에서 라이선스에 동의하고\n"
            "받은 링크 파일을 그 경로에 저장하세요 (docs/DATA_SETUP.md)." % args.urls)

    seqs = json.load(open(args.urls))["sequences"]
    part_list = [p.strip() for p in args.part.split(",") if p.strip()]
    picked = []
    for n, v in sorted(seqs.items()):
        if args.match not in n:
            continue
        for part in part_list:
            if part in v:
                picked.append((n, part, v[part]))
    total = sum(e["file_size_bytes"] for _, _, e in picked)
    print("%d files matched '%s' x %s  (%.2f GB)"
          % (len(picked), args.match, part_list, total / 1e9), flush=True)
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    done = 0
    for name, part, e in picked:
        seq_dir = os.path.join(args.out, name)
        os.makedirs(seq_dir, exist_ok=True)
        fn = e["filename"]
        is_zip = fn.endswith(".zip")
        marker = os.path.join(seq_dir, ".done_" + part)
        if part == "main_groundtruth" and not os.path.exists(marker) \
                and os.path.exists(os.path.join(seq_dir, "2d_bounding_box.csv")):
            open(marker, "w").close()      # pre-marker downloads
        if os.path.exists(marker):
            done += 1
            continue
        dst = os.path.join(args.out if is_zip else seq_dir, fn)
        if not (os.path.exists(dst) and os.path.getsize(dst) == e["file_size_bytes"]):
            urllib.request.urlretrieve(e["download_url"], dst)
        if is_zip:
            sub = seq_dir if part == "main_groundtruth" else os.path.join(seq_dir, part)
            with zipfile.ZipFile(dst) as z:
                z.extractall(sub)
            if not args.keep_zips:
                os.remove(dst)
        open(marker, "w").close()
        done += 1
        print("[%d/%d] %s/%s  %.0f MB  (%.0fs)"
              % (done, len(picked), name, part, e["file_size_bytes"] / 1e6, time.time() - t0),
              flush=True)
    print("ADT_DOWNLOAD_READY -> %s" % args.out)


if __name__ == "__main__":
    main()
