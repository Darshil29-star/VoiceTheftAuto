"""Remote-sensing concept vocabulary for grounding.

Each concept declares HOW it can be grounded:
  landcover — a land-cover segmentation class (area-like concepts: buildings, roads, water, forest…)
  composite — union of land-cover classes (e.g. urban area = building ∪ road)
  spectral  — deterministic index usable when the raster has the needed bands (NDWI, NDVI)
  object    — discrete countable objects for an open-vocabulary detector (vehicles, ships, tanks…)

Extending the flagship module = adding a concept here (and, after fine-tuning, a new backend).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Concept:
    name: str                                   # canonical label used in evidence
    synonyms: tuple[str, ...]
    kind: str                                   # landcover | composite | object
    landcover_classes: tuple[str, ...] = ()     # canonical land-cover names (see registry class_map)
    spectral: str | None = None                 # ndwi | ndvi
    detector_prompt: str | None = None          # text prompt for open-vocabulary detection
    small_object: bool = False                  # enables stricter "group box" filtering
    countable: bool = True                      # False for amorphous classes (report area, not count)
    notes: tuple[str, ...] = field(default_factory=tuple)


CONCEPTS: tuple[Concept, ...] = (
    # ---------------------------------------------------------------- land cover
    Concept("building", ("buildings", "building", "houses", "house", "homes", "rooftops", "rooftop", "roofs", "roof",
                         "structures", "warehouses", "warehouse", "residential buildings"),
            "landcover", ("building",), detector_prompt="building . house .",
            notes=("Adjacent buildings may merge into one region; counts are region counts, not house counts.",)),
    Concept("urban area", ("urban areas", "urban area", "built-up areas", "built-up area", "built up area", "settlements",
                           "settlement", "urban", "city", "town", "developed areas"),
            "composite", ("building", "road"), countable=False,
            notes=("Urban area = union of building and road pixels, closed morphologically into zones (rule-based).",)),
    Concept("road", ("roads", "road", "streets", "street", "highways", "highway", "motorway", "lanes", "road network"),
            "landcover", ("road",), countable=False),
    Concept("water", ("water bodies", "water body", "waterbodies", "water", "rivers", "river", "lakes", "lake", "ponds",
                      "pond", "reservoirs", "reservoir", "canals", "canal", "streams"),
            "landcover", ("water",), spectral="ndwi", countable=True),
    Concept("forest", ("dense vegetation", "forests", "forest", "trees", "tree cover", "woodland", "woods", "tree canopy",
                       "canopy", "jungle"),
            "landcover", ("forest",), spectral="ndvi", countable=False),
    Concept("vegetation", ("vegetation", "green cover", "greenery", "green areas", "plants"),
            "composite", ("forest", "agriculture"), spectral="ndvi", countable=False,
            notes=("Vegetation = forest ∪ agriculture land-cover classes (or NDVI when NIR is available).",)),
    Concept("agriculture", ("agricultural land", "agriculture", "farmland", "farms", "farm", "fields", "field", "crops",
                            "cropland", "crop fields", "paddy"),
            "landcover", ("agriculture",), countable=False),
    Concept("barren", ("barren land", "bare land", "bare soil", "barren", "open land", "empty land", "vacant land",
                       "bare ground", "construction sites", "construction site"),
            "landcover", ("barren",), countable=False),
    # ---------------------------------------------------------------- discrete objects (open vocabulary)
    Concept("vehicle", ("vehicles", "vehicle", "cars", "car", "trucks", "truck", "buses", "bus"), "object",
            detector_prompt="car . truck . vehicle .", small_object=True),
    Concept("ship", ("ships", "ship", "boats", "boat", "vessels", "vessel"), "object", detector_prompt="ship . boat .",
            small_object=True),
    Concept("airplane", ("airplanes", "airplane", "aircraft", "planes", "plane", "jets"), "object",
            detector_prompt="airplane .", small_object=True),
    Concept("storage tank", ("storage tanks", "storage tank", "tanks", "oil tanks", "silos"), "object",
            detector_prompt="storage tank . round tank .", small_object=True),
    Concept("swimming pool", ("swimming pools", "swimming pool", "pools", "pool"), "object",
            detector_prompt="swimming pool .", small_object=True),
    Concept("sports field", ("tennis courts", "tennis court", "basketball courts", "basketball court", "stadiums", "stadium",
                             "sports fields", "sports field", "playgrounds", "baseball field", "football field", "track"),
            "object", detector_prompt="tennis court . stadium . sports field ."),
    Concept("bridge", ("bridges", "bridge", "overpasses", "overpass"), "object", detector_prompt="bridge ."),
    Concept("windmill", ("windmills", "windmill", "wind turbines", "wind turbine"), "object",
            detector_prompt="wind turbine .", small_object=True),
    Concept("solar panel", ("solar panels", "solar panel", "solar farm", "solar farms"), "object",
            detector_prompt="solar panel ."),
)

_BY_SYNONYM: list[tuple[str, Concept]] = sorted(
    ((syn, c) for c in CONCEPTS for syn in c.synonyms), key=lambda t: len(t[0]), reverse=True
)


def by_name(name: str) -> Concept | None:
    return next((c for c in CONCEPTS if c.name == name), None)


def synonym_index() -> list[tuple[str, Concept]]:
    return _BY_SYNONYM
