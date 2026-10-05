# Model Selection — GeoSense AI (CPU-first)

Status: selection made 2026-09-17. **Latency figures marked _TBM_ are "to be measured" on the
target CPU** and are filled in only after real runs (see `docs/BENCHMARKS.md`). Nothing below
is a claimed accuracy number.

## Environment facts that drove the selection

| Item | Value |
|---|---|
| Dev machine CPU | Intel i7-14650HX, 16C/24T |
| RAM | 15.7 GB total, ~5 GB typically free |
| GPU on dev machine | RTX 4060 Laptop 8 GB (present, **not assumed** — deployment target is CPU-only) |
| Python | 3.10 |
| Policy | `DEVICE=auto` → CUDA if available, else CPU. All specialists must work on CPU. |

Selection criteria, in priority order: CPU feasibility → RAM → license → native `transformers`
support (no fragile `trust_remote_code`) → remote-sensing suitability → fine-tunability →
provides a genuine score (for honest confidence).

## Per-task comparison

### Captioning (+ flagship fine-tuning target)

| Candidate | Params / weights | License | CPU | RS suitability | Scores? | Fine-tune | Verdict |
|---|---|---|---|---|---|---|---|
| **Florence-2-base** (`florence-community/Florence-2-base`) | 0.23B / 463 MB | MIT | good | generic, but strong dense/region captions; widely fine-tuned on RS data | token log-probs (uncalibrated) | easy, LoRA or full on 8 GB | **Selected** |
| SmolVLM-256M-Instruct | 0.26B / 513 MB | Apache-2.0 | good | generic chat VLM | token log-probs | moderate | alternate |
| Moondream2 | 1.9B / 3.9 GB | Apache-2.0 | slow on CPU | generic | no | hard | rejected (size) |
| GeoChat-7B | 7B | — | infeasible on CPU | RS-specific | no | GPU only | rejected (hardware) |
| Qwen2.5-VL-3B | 3B / 7.5 GB | Qwen license | very slow on CPU | generic, good | no | GPU | rejected (size) |

Florence-2 is now natively in `transformers` (`Florence2ForConditionalGeneration`,
`florence-community/*` checkpoints), which removes the `trust_remote_code` breakage the original
`microsoft/Florence-2-*` repos suffer from. One model covers caption, detailed caption, dense region
caption, and phrase grounding — which is exactly what makes it the right **fine-tuning target**
for the Captioning + Grounding flagship (DIOR-RSVG boxes + RSICD captions in one training format).

Honesty note: Florence-2 is not remote-sensing-trained. Pretrained captions will use
everyday vocabulary ("aerial view of a neighborhood"). RS vocabulary comes from (a) the RemoteCLIP
land-cover evidence below and (b) fine-tuning.

### Remote-sensing scene / land-cover evidence

| Candidate | Weights | License | Notes | Verdict |
|---|---|---|---|---|
| **RemoteCLIP ViT-B/32** (`chendelong/RemoteCLIP`, open_clip format) | 605 MB | Apache-2.0 | trained on RS image-text data; zero-shot scene/land-cover scores | **Selected** |
| clip-rsicd-v2 (flax-community) | 605 MB | Apache-2.0 | RSICD-only fine-tune of OpenAI CLIP | alternate |
| GeoRSCLIP ViT-B/32 | ~600 MB | check | larger RS pretraining corpus; less mature packaging | future option |

Its softmax scores are **relative to the label set we supply** — reported as
`model_score (relative to N labels)`, never as calibrated probability.

### Grounding (flagship)

| Candidate | Weights | License | Scores | Masks | Notes | Verdict |
|---|---|---|---|---|---|---|
| **Grounding DINO tiny** (`IDEA-Research/grounding-dino-tiny`) | 689 MB | Apache-2.0 | yes, per box | no | open-vocabulary, native transformers | **Selected (default backend)** |
| Florence-2-base phrase grounding | shared | MIT | no | no | same weights as captioning; fine-tuning target | **Selected (second backend)** |
| OWLv2 base | 1.2 GB | Apache-2.0 | yes | no | good on small objects, slower | alternate |
| **SlimSAM-50** (`nielsr/slimsam-50-uniform`) | 112 MB | Apache-2.0 | IoU pred | yes | box-prompted masks → polygons | **Selected (mask refiner)** |
| SAM ViT-B | 750 MB | Apache-2.0 | IoU pred | yes | ~6× larger than SlimSAM | alternate |

Grounding pipeline design (details in `docs/ARCHITECTURE.md`):
query parsing → label normalisation (RS synonyms) → **tiled** detection for large rasters
(small buildings vanish when a 1024 px image is resized to 800 px) → cross-tile NMS →
optional SlimSAM masks → polygons → pixel→map coordinate transform (GeoTIFF) → evidence.
For amorphous classes (vegetation, water) with NIR bands available, a **deterministic spectral
index** path (NDVI / NDWI) is used and labelled as such. RGB-only vegetation uses an ExG index,
labelled "RGB index (estimated)".

### Visual Question Answering

| Candidate | Weights | License | Verdict |
|---|---|---|---|
| **SmolVLM-256M-Instruct** | 513 MB | Apache-2.0 | **Selected** for open-ended questions |
| SmolVLM-500M-Instruct | 1.0 GB | Apache-2.0 | configurable upgrade |
| Moondream2 / Qwen2.5-VL | 3.9 / 7.5 GB | — | rejected on CPU |

Design: the VQA specialist first classifies the question type. Presence / count / location questions
("how many buildings", "is there water") are answered **from grounding evidence** (boxes/masks,
with scores). Only open-ended questions go to the generative VLM, and that answer is
marked `confidence: unavailable from model`. A generic 256M VLM on satellite imagery is weak;
we say so rather than hide it.

### Change analysis

| Candidate | License | Notes | Verdict |
|---|---|---|---|
| **Deterministic pipeline** (histogram matching, ECC/phase-correlation registration or CRS-based alignment, Lab-space change vector + gradient difference, Otsu, morphology, connected components) | own code | runs anywhere, fully explainable | **Selected (core)** |
| Object-level attribution via grounding on T1 and T2 | reuses selected models | labels a region "building appeared" **only** when detector evidence supports it | **Selected (semantic layer)** |
| HZDR-FWGEL UCD BIT / ChangeFormer (LEVIR-CD256) | no license, no model code on card | would be a learned change model | rejected for now (license + packaging); registry slot kept |

Change "confidence" is an **estimated** separability score (distance between changed / unchanged
magnitude distributions), labelled estimated.

### Optical–SAR fusion

| Candidate | Notes | Verdict |
|---|---|---|
| **Deterministic co-registration + rule-based fusion** | CRS/bounds/resolution compatibility check; reprojection to a common grid; SAR linear→dB + Lee speckle filter; optical NDVI/NDWI; per-pixel rules (water, built-up, vegetation, bare) with **agreement / disagreement** maps between modalities | **Selected** |
| Learned multimodal fusion (e.g. SEN12MS-trained segmentation) | would need GPU training + labelled data we don't have | future, registry slot kept |

Thresholds (e.g. VV < −18 dB → likely open water) are literature heuristics, documented in code, and
reported as such. The module **refuses** to fuse images whose footprints don't overlap or whose
shapes don't match without georeferencing.

### Task classifier

Rule/pattern scorer with input awareness (number of images, SAR/optical role tags), returning
per-task scores and a rationale. When the top-2 margin is small and inputs don't disambiguate,
the controller returns a clarification request. Pluggable: an embedding-based classifier can
replace it via the registry.

### Response generator

Template-based, evidence-driven (no LLM). Every sentence is produced from a specialist output field.
An optional LLM rewriter slot exists in the registry, disabled by default.

## Memory plan (CPU)

Weights on disk ≈ 2.4 GB total. Models are **lazy-loaded** and held in an LRU cache bounded by
`MAX_LOADED_MODELS` (default 2) so peak RAM stays within a ~5 GB free budget.

## Public sample data (not training data by default)

| Source | Use | License |
|---|---|---|
| LEVIR-CD+ (few test pairs + masks, 0.5 m) | change demo, grounding demo, change evaluation | research use (card says "unknown"; original LEVIR-CD: academic) |
| DIOR-RSVG (annotations + few images) | grounding sample/eval, training format | research use |
| RSICD (few image–caption rows) | captioning sample/eval | research use |
| Sentinel-2 L2A + Sentinel-1 GRD windows (Microsoft Planetary Computer) | optical–SAR fusion demo with real CRS/timestamps | Copernicus open data (S2: CC-BY-4.0 on PC; S1 GRD: Copernicus terms) |

Evaluation caveat: LEVIR-CD+ / DIOR-RSVG samples are standard benchmarks; any numbers we compute on
a handful of samples are smoke tests, **not** benchmark results.
