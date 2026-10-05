# Measured CPU Benchmarks

All numbers were measured on the dev machine: Intel i7-14650HX (8P+8E cores),
CPU-only PyTorch 2.14, float32, `torch.set_num_threads(8)`, transformers 5.17.
Single runs on a handful of samples. **These are smoke measurements, not benchmark results.**
Latency varied by up to 5x between runs (first-call warm-up and background CPU load), so
steady-state values are reported alongside the worst observed.

## 2026-09-17 — model feasibility on LEVIR-CD+ samples (0.5 m, 1024x1024 RGB)

### Florence-2-base
| Task | Setting | Latency | Observation |
|---|---|---|---|
| `<MORE_DETAILED_CAPTION>` | beams=3, 128 tokens | 20.0 s | Plausible generic caption ("aerial view of a residential area…") |
| `<CAPTION_TO_PHRASE_GROUNDING>` "buildings" | beams=3 | 8.7 s | **Returned whole-image boxes** — unusable for dense small objects |

Decision: greedy decoding for captions in the live path; Florence-2 grounding is a fine-tuning target only.

### Grounding DINO tiny (zero-shot)
| Input | Latency (steady / worst) | Result on a 512 px residential tile with ~60 houses |
|---|---|---|
| 512 px tile resized to 800 | 2.5 s / 15 s | 1–3 correct boxes plus neighbourhood-sized boxes |
| 512 px tile at 512 | 1.1 s / 7.6 s | 1–3 boxes |
| 256 px tiles zoomed to 800 (9 tiles) | 107 s | 3 boxes |

Decision: **not used for land-cover concepts** (buildings, roads, water, vegetation). Kept for discrete
objects (vehicles, ships, storage tanks, airplanes) where open vocabulary matters.

### SegFormer-B2 LoveDA
Metric: fraction of LEVIR ground-truth *new-building* pixels (T2) that the model labels `building`.
This is a partial recall check only — LEVIR masks mark changed buildings, not all buildings.

| Pair | Scene | Native 0.5 m, 512 tiles | Rescaled to ~0.3 m (x1.67), 512 tiles |
|---|---|---|---|
| pair00 | dense small houses | 0.444 (4.4–14.8 s) | **0.869** (9.2 s) |
| pair01 | large warehouses | 0.959 (4.4 s) | 0.934 (8.1 s) |
| pair02 | mixed residential | 0.291 (4.4 s) | **0.444** (8.0 s) |

Whole image resized to 512 (no tiling): 0.53–0.65 s but misses most buildings (recall 0.000 on pair00).

Decisions:
- GSD-aware rescale toward LoveDA's ~0.3 m before tiled inference (512 px tiles, 64 px overlap).
- Adjacent houses merge into connected blocks → responses say "building regions", never an exact house count.
- Imagery coarser than ~2 m GSD (e.g. Sentinel-2 10 m) is outside this model's domain → LOW CONFIDENCE warning.

### SlimSAM-50
Load 0.9 s. Box-prompted mask inference ran; timing to be measured inside the grounding specialist.

### Florence-2-base captioning, greedy decoding (num_beams=1)
| Image | Task | Latency | Mean token prob | Observed fabrication |
|---|---|---|---|---|
| LEVIR pair01 (warehouses) | `<DETAILED_CAPTION>` | 8.8 s | 0.68 | "surrounded by a large building" |
| LEVIR pair01 | `<MORE_DETAILED_CAPTION>` | 11.3 s | 0.86 | "multiple levels and multiple floors" (flat single-storey roofs) |
| RSICD farmland_366 | `<DETAILED_CAPTION>` | 9.8 s | 0.78 | "a watermark in the bottom right corner", "subtle reminder of the beauty" |
| RSICD pond_401 | `<DETAILED_CAPTION>` | 9.2 s | 0.79 | "filled with a variety of green algae" |

Decisions:
- Mean token probability does **not** track factual correctness (0.78 on a caption with a fabricated
  watermark) → caption confidence is reported as **unavailable**; the token probability is kept only as a
  diagnostic field.
- Captions are labelled as generic-model output and **cross-checked against land-cover segmentation**;
  mentions without supporting evidence are flagged in the response.
- `<DETAILED_CAPTION>` (shorter, fewer fabricated details) is the default.
