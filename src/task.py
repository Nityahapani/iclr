"""
Synthetic compositional task for the Neural Archaeology 'killer experiment'.

Language: object + attribute -> label
We define a small vocabulary of "objects" (nonsense words like zor, plib, kade...)
and a fixed set of colors as labels. Most objects have a FIXED, unambiguous
mapping to a color for their entire training history (these are "filler" objects,
used so the model has to learn a real compositional task rather than memorize one
associations). One special object -- "zor" -- is the object whose meaning changes
across training phases (A: zor=red, B: zor=blue).

Input format (token ids): [OBJECT_TOKEN, QUERY_TOKEN] -> predict color label.
This is intentionally trivial (no real "attribute" needed) so that the *only*
thing happening in training is a change in the association for one object.
"""
import random
import numpy as np
import torch

COLORS = ["red", "blue", "green", "yellow", "purple", "orange"]
COLOR2ID = {c: i for i, c in enumerate(COLORS)}

# Filler objects: fixed mapping for the whole run (these keep the task "real")
FILLER_OBJECTS = ["plib", "kade", "worn", "fyx", "glim", "dron", "spek", "yult",
                   "brox", "twil", "quen", "harn", "ozzy", "vurn", "clef", "mibs"]

SPECIAL_OBJECT = "zor"  # the object under study

ALL_OBJECTS = FILLER_OBJECTS + [SPECIAL_OBJECT]
OBJ2ID = {o: i for i, o in enumerate(ALL_OBJECTS)}

VOCAB_SIZE = len(ALL_OBJECTS)
NUM_CLASSES = len(COLORS)


def make_filler_mapping(seed: int):
    """Fixed random object->color mapping for filler objects, held constant
    across an entire experiment (same seed) so filler task difficulty is
    identical across treatment/control populations."""
    rng = random.Random(seed)
    mapping = {}
    for o in FILLER_OBJECTS:
        mapping[o] = rng.choice(COLORS)
    return mapping


class PhaseDataset:
    """
    Generates (object_id, label_id) pairs for a given phase.

    phase='A': zor -> red, fillers -> fixed mapping
    phase='B': zor -> blue, fillers -> fixed mapping
    phase='B_only': same as B (used for the control population, semantics identical
                     to phase B, kept as separate name for clarity/logging)
    """
    def __init__(self, filler_mapping, phase: str, zor_frac: float = 1.0 / (len(FILLER_OBJECTS) + 1)):
        self.filler_mapping = filler_mapping
        self.phase = phase
        assert phase in ("A", "B", "B_only")
        self.zor_color = "red" if phase == "A" else "blue"
        self.zor_frac = zor_frac  # natural frequency of zor among all objects

    def sample_batch(self, batch_size: int, rng: np.random.RandomState):
        objs = []
        labels = []
        for _ in range(batch_size):
            if rng.rand() < self.zor_frac:
                o = SPECIAL_OBJECT
                c = self.zor_color
            else:
                o = rng.choice(FILLER_OBJECTS)
                c = self.filler_mapping[o]
            objs.append(OBJ2ID[o])
            labels.append(COLOR2ID[c])
        return (torch.tensor(objs, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long))

    def full_eval_set(self):
        """Deterministic eval set: every object once (for matching/KL comparisons)."""
        objs, labels = [], []
        for o in FILLER_OBJECTS:
            objs.append(OBJ2ID[o])
            labels.append(COLOR2ID[self.filler_mapping[o]])
        objs.append(OBJ2ID[SPECIAL_OBJECT])
        labels.append(COLOR2ID[self.zor_color])
        return (torch.tensor(objs, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long))
