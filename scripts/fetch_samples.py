"""Fetch small public sample data into data/samples (explicit, never run at app start).

    python scripts/fetch_samples.py levir rsicd dior sentinel
    python scripts/fetch_samples.py sentinel --bbox 91.62 26.13 91.78 26.22 --date 2024-01-01/2024-03-31

Sources and licenses are recorded in data/samples/SOURCES.md.
Only individual rows / files / raster windows are fetched — never whole archives.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SAMPLES = REPO / "data" / "samples"
DS_SERVER = "https://datasets-server.huggingface.co/rows"
BYTES = {"total": 0}


def _get(url: str, timeout: int = 120, retries: int = 5) -> bytes:
    import time
    import urllib.error

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = r.read()
            BYTES["total"] += len(data)
            return data
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == retries - 1:
                raise
            time.sleep(2 ** attempt * 3)  # polite backoff on rate limiting
        except urllib.error.URLError:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _rows(dataset: str, split: str, offset: int, length: int) -> list[dict]:
    q = urllib.parse.urlencode({"dataset": dataset, "config": "default", "split": split, "offset": offset, "length": length})
    payload = json.loads(_get(f"{DS_SERVER}?{q}"))
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return [r["row"] for r in payload["rows"]]


# --------------------------------------------------------------------------- LEVIR-CD+
def fetch_levir(n: int = 4) -> None:
    import cv2

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    out = SAMPLES / "change" / "levir_cdplus"
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("pair*"):
        stale.unlink()
    # The datasets-server re-encodes images as lossy JPEG, which corrupts binary masks.
    # Read original PNG bytes from the parquet instead. Row group 1 of the first test shard
    # holds only 16 rows (~60 MB); row group 0 holds 100 rows (~377 MB).
    fs = HfFileSystem()
    shard = "datasets/blanchon/LEVIR_CDPlus/data/test-00000-of-00003.parquet"
    with fs.open(shard, "rb", block_size=1 << 20) as f:
        pf = pq.ParquetFile(f)
        rg = pf.num_row_groups - 1
        BYTES["total"] += pf.metadata.row_group(rg).total_byte_size
        table = pf.read_row_group(rg, columns=["image1", "image2", "mask"]).to_pylist()

    # prefer pairs that actually contain change, so the demo is meaningful
    def frac(row):
        m = cv2.imdecode(np.frombuffer(row["mask"]["bytes"], np.uint8), cv2.IMREAD_GRAYSCALE)
        return float((m > 0).mean())

    ranked = sorted(table, key=frac, reverse=True)
    picks = ranked[: n - 1] + [min(table, key=frac)]  # plus one low-change pair as a control
    manifest = []
    for i, row in enumerate(picks):
        pair = {}
        for key, name in (("image1", "t1"), ("image2", "t2"), ("mask", "mask")):
            img = cv2.imdecode(np.frombuffer(row[key]["bytes"], np.uint8), cv2.IMREAD_UNCHANGED)
            if name == "mask":
                if img.ndim == 3:
                    img = img[..., 0]
                img = np.where(img > 0, 255, 0).astype(np.uint8)  # source masks may be 0/1 or 0/255
            path = out / f"pair{i:02d}_{name}.png"
            cv2.imwrite(str(path), img)
            pair[name] = path.name
        pair["source_path"] = row["image1"].get("path")
        pair["changed_fraction"] = frac(row)
        manifest.append(pair)
        print(f"  levir pair {i}: changed {pair['changed_fraction']:.3f}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


# --------------------------------------------------------------------------- RSICD
def fetch_rsicd(n: int = 20) -> None:
    out = SAMPLES / "captioning" / "rsicd"
    (out / "images").mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    records = []
    # The test split is a single ~55 MB row group; read it directly (datasets-server rate-limits).
    fs = HfFileSystem()
    with fs.open("datasets/arampacha/rsicd/data/test-00000-of-00001.parquet", "rb", block_size=1 << 20) as f:
        pf = pq.ParquetFile(f)
        BYTES["total"] += pf.metadata.row_group(0).total_byte_size
        rows = pf.read_row_group(0, columns=["filename", "captions", "image"]).to_pylist()
    seen_classes: set[str] = set()
    for row in rows:  # rows are sorted by class; keep one image per scene class
        cls = Path(row["filename"]).stem.rsplit("_", 1)[0]
        if cls in seen_classes:
            continue
        seen_classes.add(cls)
        name = Path(row["filename"]).name
        (out / "images" / name).write_bytes(row["image"]["bytes"])
        records.append({"image": f"images/{name}", "scene_class": cls, "captions": row["captions"]})
        if len(records) >= n:
            break
    with (out / "captions.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  rsicd: {len(records)} images, classes={sorted(seen_classes)[:10]}...")


# --------------------------------------------------------------------------- DIOR-RSVG
def fetch_dior(n: int = 20) -> None:
    import xml.etree.ElementTree as ET

    from huggingface_hub import HfFileSystem

    out = SAMPLES / "grounding" / "dior_rsvg"
    (out / "images").mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    base = "datasets/LittleCollections/DIOR-RSVG"

    ann_zip = zipfile.ZipFile(io.BytesIO(fs.cat_file(f"{base}/Annotations.zip")))
    BYTES["total"] += sum(z.compress_size for z in ann_zip.infolist())
    test_ids = fs.cat_file(f"{base}/test.txt").decode().split()

    # Parse annotations: one XML per image, possibly several <object> with <description>
    by_image: dict[str, list[dict]] = {}
    for info in ann_zip.infolist():
        if not info.filename.endswith(".xml"):
            continue
        root = ET.fromstring(ann_zip.read(info))
        fname = root.findtext("filename")
        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            by_image.setdefault(fname, []).append({
                "label": obj.findtext("name"),
                "expression": obj.findtext("description"),
                "bbox": [int(float(bb.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax")],
            })

    # test.txt lists sample indices; pick images with varied categories
    names = sorted(by_image)
    chosen, cats = [], set()
    for fname in names:
        labels = {o["label"] for o in by_image[fname]}
        if labels <= cats:
            continue
        cats |= labels
        chosen.append(fname)
        if len(chosen) >= n:
            break

    with fs.open(f"{base}/JPEGImages.zip", "rb", block_size=1 << 20) as remote:
        zimg = zipfile.ZipFile(remote)
        index = {Path(i.filename).name: i for i in zimg.infolist()}
        records = []
        for fname in chosen:
            info = index.get(fname)
            if info is None:
                continue
            data = zimg.read(info)
            BYTES["total"] += info.compress_size
            (out / "images" / fname).write_bytes(data)
            records.append({"image": f"images/{fname}", "objects": by_image[fname]})
    with (out / "annotations.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  dior-rsvg: {len(records)} images, categories={sorted(cats)}; test split size={len(test_ids)}")


# --------------------------------------------------------------------------- Sentinel-1/2
def fetch_sentinel(bbox: list[float], date: str, size: int = 1024) -> None:
    import planetary_computer
    import pystac_client
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import transform_bounds

    out = SAMPLES / "fusion" / "sentinel"
    out.mkdir(parents=True, exist_ok=True)
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace
    )

    s2_items = list(catalog.search(
        collections=["sentinel-2-l2a"], bbox=bbox, datetime=date, query={"eo:cloud_cover": {"lt": 5}},
        sortby=[{"field": "eo:cloud_cover", "direction": "asc"}], max_items=5,
    ).items())
    if not s2_items:
        raise RuntimeError("No low-cloud Sentinel-2 scene found for bbox/date")
    s2 = s2_items[0]
    epsg = s2.properties.get("proj:epsg") or int(str(s2.properties.get("proj:code", "EPSG:32646")).split(":")[1])
    dst_crs = f"EPSG:{epsg}"

    # 10 m grid centred on the bbox, `size` x `size` pixels
    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", dst_crs, *bbox)
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    half = size * 10 / 2
    transform = from_origin(cx - half, cy + half, 10, 10)
    grid = dict(crs=dst_crs, transform=transform, width=size, height=size)

    bands = {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir"}
    s2_stack = []
    for b in bands:
        with rasterio.open(s2.assets[b].href) as src, WarpedVRT(src, resampling=Resampling.bilinear, **grid) as vrt:
            s2_stack.append(vrt.read(1))
    s2_arr = np.stack(s2_stack).astype(np.uint16)
    s2_date = s2.datetime.isoformat()
    s2_path = out / "s2_optical_guwahati.tif"
    with rasterio.open(s2_path, "w", driver="GTiff", count=4, dtype="uint16", compress="deflate", nodata=0, **grid) as dst:
        dst.write(s2_arr)
        for i, name in enumerate(bands.values(), 1):
            dst.set_band_description(i, name)
        dst.update_tags(SENSOR="Sentinel-2 L2A", ACQUISITION_DATE=s2_date, SCENE_ID=s2.id, IMAGE_ROLE="optical",
                        SCALE="reflectance = DN / 10000 (offset handled by processing baseline)")
    print(f"  sentinel-2: {s2.id} {s2_date} cloud={s2.properties.get('eo:cloud_cover')}")

    # Sentinel-1 GRD closest in time to the S2 acquisition
    s1_items = list(catalog.search(collections=["sentinel-1-grd"], bbox=bbox, datetime=date, max_items=20).items())
    if not s1_items:
        raise RuntimeError("No Sentinel-1 GRD scene found for bbox/date")
    s1 = min(s1_items, key=lambda it: abs((it.datetime - s2.datetime).total_seconds()))
    sar = []
    for pol in ("vv", "vh"):
        if pol not in s1.assets:
            continue
        with rasterio.open(s1.assets[pol].href) as src:
            # GRD COGs are in radar geometry with GCPs; GDAL's warper uses the GCPs
            gcps, gcp_crs = src.gcps
            with WarpedVRT(src, src_crs=gcp_crs, resampling=Resampling.bilinear, **grid) as vrt:
                dn = vrt.read(1).astype(np.float32)
        cal = _s1_sigma0_constant(s1, pol)
        sigma0 = np.where(dn > 0, (dn ** 2) / (cal ** 2), 0).astype(np.float32)
        sar.append((pol, sigma0))
    s1_path = out / "s1_sar_guwahati.tif"
    with rasterio.open(s1_path, "w", driver="GTiff", count=len(sar), dtype="float32", compress="deflate", nodata=0, **grid) as dst:
        for i, (pol, arr) in enumerate(sar, 1):
            dst.write(arr, i)
            dst.set_band_description(i, pol.upper())
        dst.update_tags(SENSOR="Sentinel-1 GRD", ACQUISITION_DATE=s1.datetime.isoformat(), SCENE_ID=s1.id,
                        IMAGE_ROLE="sar", UNITS="sigma0 linear (approximate: scene-mean calibration constant)",
                        ORBIT=str(s1.properties.get("sat:orbit_state")))
    print(f"  sentinel-1: {s1.id} {s1.datetime.isoformat()} pols={[p for p, _ in sar]}")
    (out / "manifest.json").write_text(json.dumps({
        "aoi_bbox_wgs84": bbox, "grid_crs": dst_crs, "size_px": size, "gsd_m": 10,
        "optical": {"file": s2_path.name, "scene": s2.id, "datetime": s2_date},
        "sar": {"file": s1_path.name, "scene": s1.id, "datetime": s1.datetime.isoformat()},
        "time_gap_hours": round(abs((s1.datetime - s2.datetime).total_seconds()) / 3600, 1),
    }, indent=2))


def _s1_sigma0_constant(item, pol: str) -> float:
    """Mean sigmaNought calibration value from the GRD calibration annotation (approximation)."""
    import xml.etree.ElementTree as ET

    key = f"schema-calibration-{pol}"
    if key not in item.assets:
        raise RuntimeError(f"calibration asset {key} missing")
    root = ET.fromstring(_get(item.assets[key].href))
    vals = []
    for vec in root.iter("calibrationVector"):
        vals.extend(float(v) for v in vec.findtext("sigmaNought").split())
    return float(np.mean(vals))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", choices=["levir", "rsicd", "dior", "sentinel"])
    ap.add_argument("--bbox", nargs=4, type=float, default=[91.62, 26.13, 91.78, 26.22], help="lon/lat bbox (Guwahati)")
    ap.add_argument("--date", default="2024-01-01/2024-03-31")
    args = ap.parse_args()
    SAMPLES.mkdir(parents=True, exist_ok=True)
    for s in args.sources:
        print(f"==> {s}", flush=True)
        if s == "levir":
            fetch_levir()
        elif s == "rsicd":
            fetch_rsicd()
        elif s == "dior":
            fetch_dior()
        elif s == "sentinel":
            fetch_sentinel(args.bbox, args.date)
        print(f"    transferred so far: {BYTES['total'] / 1e6:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
