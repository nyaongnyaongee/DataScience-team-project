"""item_meta.jsonl의 images URL에서 아이템당 대표 이미지 1장을 받는다.

data/images/{item_id}.jpg 로 저장하고, 이미 받은 파일은 건너뛴다(중단 후 재실행 가능).
실패 목록은 data/images/_failed.txt 에 남긴다.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def pick_url(images: list[dict]) -> str | None:
    ordered = sorted(images, key=lambda im: im.get("variant") != "MAIN")
    for im in ordered:
        for key in ("large", "hi_res", "thumb"):
            if im.get(key):
                return im[key]
    return None


def fetch(session: requests.Session, url: str, out: Path, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=20)
            if r.status_code == 200 and r.content:
                tmp = out.with_suffix(".part")
                tmp.write_bytes(r.content)
                tmp.rename(out)
                return True
            if r.status_code == 404:
                return False
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--meta", default=str(ROOT / "data/item_meta.jsonl"))
    p.add_argument("--out", default=str(ROOT / "data/images"))
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs, no_url = [], []
    with open(args.meta, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            item = d["item_id"]
            dest = out_dir / f"{item}.jpg"
            if dest.exists():
                continue
            url = pick_url(d.get("images") or [])
            (jobs.append((item, url, dest)) if url else no_url.append(item))

    print(f"to download: {len(jobs)}, no url: {len(no_url)}", flush=True)
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (course project image fetch)"
    failed = list(no_url)
    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch, session, url, dest): item for item, url, dest in jobs}
        for fut in as_completed(futs):
            done += 1
            if not fut.result():
                failed.append(futs[fut])
            if done % 1000 == 0:
                print(f"{done}/{len(jobs)} done, failed {len(failed)}", flush=True)
    (out_dir / "_failed.txt").write_text("\n".join(map(str, sorted(failed))) + "\n")
    print(f"finished. failed/missing: {len(failed)}", flush=True)


if __name__ == "__main__":
    main()
