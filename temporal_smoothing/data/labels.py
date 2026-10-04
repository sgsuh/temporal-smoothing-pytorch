"""CamVid label definitions, ported from the official implementation.

``camvid`` / ``camvid-11`` merge the 32 CamVid labels into 11 categories (Void is ignored),
``camvid-31`` keeps every non-void label. Class indices follow the official code so that
the memorized class weights apply unchanged.
"""

from collections import namedtuple
from typing import Dict, List, Tuple

import torch
from torch import Tensor

__all__ = ["Label", "CAMVID_LABELS", "color_map", "class_names", "num_classes", "memorized_class_weights"]

Label = namedtuple("Label", ["name", "category", "category_id", "ignore", "color"])

CAMVID_LABELS = [
    Label("Void", "Void", 0, True, (0, 0, 0)),
    Label("Sky", "Sky", 1, False, (128, 128, 128)),
    Label("Bridge", "Building", 2, False, (0, 128, 64)),
    Label("Building", "Building", 2, False, (128, 0, 0)),
    Label("Wall", "Building", 2, False, (64, 192, 0)),
    Label("Tunnel", "Building", 2, False, (64, 0, 64)),
    Label("Archway", "Building", 2, False, (192, 0, 128)),
    Label("Column_Pole", "Pole", 3, False, (192, 192, 128)),
    Label("TrafficCone", "Pole", 3, False, (0, 0, 64)),
    Label("Road", "Road", 4, False, (128, 64, 128)),
    Label("LaneMkgsDriv", "Road", 4, False, (128, 0, 192)),
    Label("LaneMkgsNonDriv", "Road", 4, False, (192, 0, 64)),
    Label("Sidewalk", "Pavement", 5, False, (0, 0, 192)),
    Label("ParkingBlock", "Pavement", 5, False, (64, 192, 128)),
    Label("RoadShoulder", "Pavement", 5, False, (128, 128, 192)),
    Label("Tree", "Tree", 6, False, (128, 128, 0)),
    Label("VegetationMisc", "Tree", 6, False, (192, 192, 0)),
    Label("SignSymbol", "SignSymbol", 7, False, (192, 128, 128)),
    Label("Misc_Text", "SignSymbol", 7, False, (128, 128, 64)),
    Label("TrafficLight", "SignSymbol", 7, False, (0, 64, 64)),
    Label("Fence", "Fence", 8, False, (64, 64, 128)),
    Label("Car", "Car", 9, False, (64, 0, 128)),
    Label("SUVPickupTruck", "Car", 9, False, (64, 128, 192)),
    Label("Train", "Car", 9, False, (192, 64, 128)),
    Label("Truck_Bus", "Car", 9, False, (192, 128, 192)),
    Label("OtherMoving", "Car", 9, False, (128, 64, 64)),
    Label("Pedestrian", "Pedestrian", 10, False, (64, 64, 0)),
    Label("Child", "Pedestrian", 10, False, (192, 128, 64)),
    Label("CartLuggagePram", "Pedestrian", 10, False, (64, 0, 192)),
    Label("Animal", "Pedestrian", 10, False, (64, 128, 64)),
    Label("Bicyclist", "Bicyclist", 11, False, (0, 128, 192)),
    Label("MotorcycleScooter", "Bicyclist", 11, False, (192, 0, 192)),
]

_ALIASES = {"camvid": "camvid-11", "camvid-11": "camvid-11", "camvid-31": "camvid-31"}

# Median frequency balancing weights memorized in the official code.
_MEMORIZED_WEIGHTS = {
    "camvid-11": [
        0.30734012, 0.19833793, 4.7175865, 0.16562003, 0.6806351, 0.42397258,
        4.2133756, 3.256359, 1.0, 6.7325764, 9.058633,
    ],
    "camvid-31": [
        0.027896924, 11.773657, 0.019115845, 0.3273610, 0.0000000, 10.607987,
        0.42964065, 128.60786, 0.016009688, 0.24783362, 44.399693, 0.06718957,
        1.2284949, 2.0446982, 0.041175604, 0.58860964, 3.7385745, 0.6753312,
        1.1540297, 0.29557616, 0.13028075, 0.7452813, 0.000000, 1.0,
        1.0005174, 0.66517824, 15.874812, 16.524097, 105.224625, 0.82981503,
        90.09858,
    ],
}


def _canonical(name: str) -> str:
    try:
        return _ALIASES[name.lower()]
    except KeyError:
        raise ValueError(f"unknown dataset: {name}; expected one of {sorted(_ALIASES)}") from None


def _labels() -> List[Label]:
    return [label for label in CAMVID_LABELS if not label.ignore]


def color_map(name: str = "camvid-11") -> Dict[Tuple[int, int, int], int]:
    """RGB color -> class index. Colors that are not listed (e.g. Void) map to void (-1)."""
    if _canonical(name) == "camvid-11":
        return {label.color: label.category_id - 1 for label in _labels()}
    return {label.color: i for i, label in enumerate(_labels())}


def class_names(name: str = "camvid-11") -> List[str]:
    if _canonical(name) == "camvid-11":
        names: Dict[int, str] = {}
        for label in _labels():
            names.setdefault(label.category_id - 1, label.category)
        return [names[i] for i in range(len(names))]
    return [label.name for label in _labels()]


def num_classes(name: str = "camvid-11") -> int:
    return len(class_names(name))


def memorized_class_weights(name: str = "camvid-11") -> Tensor:
    return torch.tensor(_MEMORIZED_WEIGHTS[_canonical(name)])
