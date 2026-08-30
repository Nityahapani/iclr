"""
Synthetic compositional task for the Neural Archaeology 'killer experiment'.

Language: object + attribute -> label
We define a small vocabulary of "objects" (nonsense words like zor, plib, kade...)
and a fixed set of colors as labels. Most objects have a FIXED, unambiguous
mapping to a color for their entire training history (these are "filler" objects,
used so the model has to learn a real compositional task rather than memorize one
associations). One special object -- "zor" -- is the object whose meaning changes
across training phases (A: zor=red, B: zor=blue).

REVISED INPUT FORMAT: [OBJECT_TOKEN, CONTEXT_TOKEN] -> predict color label.
The context token is a genuine second input dimension (not just a query
placeholder). Two context values exist: CTX_RED and CTX_BLUE, each of which
independently biases the color decision for a SUBSET of objects (a context
only matters for objects whose color is context-dependent; for context-
INDEPENDENT filler objects, the label is the same regardless of context, so
context is a real, separately-manipulable input variable and not
epiphenomenal). This lets us construct the required difference-in-differences
direction:
    v_A = [h(zor, CTX_RED) - h(zor, CTX_BLUE)]
        - [h(z_ctrl, CTX_RED) - h(z_ctrl, CTX_BLUE)]
where z_ctrl is a context-sensitive control object structurally matched to
zor (i.e. also has a context-dependent color mapping), so the subtraction
removes generic red/blue-context computation and isolates what's specific
to zor.
"""
import random
import numpy as np
import torch

COLORS = ["red", "blue", "green", "yellow", "purple", "orange"]
COLOR2ID = {c: i for i, c in enumerate(COLORS)}

CONTEXTS = ["CTX_RED", "CTX_BLUE"]
CTX2ID = {c: i for i, c in enumerate(CONTEXTS)}

# Filler objects: fixed, CONTEXT-INDEPENDENT mapping for the whole run.
FILLER_OBJECTS = ["plib", "kade", "worn", "fyx", "glim", "dron", "spek", "yult",
                   "brox", "twil", "quen", "harn", "ozzy", "vurn", "clef", "mibs"]

SPECIAL_OBJECT = "zor"          # the object under study (context-dependent, history changes)
CONTROL_OBJECT = "vex"          # structurally matched control: ALSO context-dependent,
                                 # but its context-dependence is stable across all of training
                                 # (never changes phase-to-phase). Used for the
                                 # difference-in-differences subtraction.

ALL_OBJECTS = FILLER_OBJECTS + [SPECIAL_OBJECT, CONTROL_OBJECT]
OBJ2ID = {o: i for i, o in enumerate(ALL_OBJECTS)}

VOCAB_SIZE = len(ALL_OBJECTS)
NUM_CLASSES = len(COLORS)
CONTEXT_VOCAB_SIZE = len(CONTEXTS)


def make_filler_mapping(seed: int):
    """Fixed random object->color mapping for filler objects (context-independent),
    held constant across an entire experiment (same seed)."""
    rng = random.Random(seed)
    mapping = {}
    for o in FILLER_OBJECTS:
        mapping[o] = rng.choice(COLORS)
    return mapping


# CONTROL_OBJECT's context-dependent mapping: fixed for ALL phases (A, B, B_only).
# This is what makes it a valid "generic context computation" reference: its
# red/blue-context sensitivity never changes, so subtracting it removes only
# the PART of the red/blue-context direction that has nothing to do with zor's
# specific history.
CONTROL_CTX_MAPPING = {"CTX_RED": "red", "CTX_BLUE": "blue"}


class PhaseDataset:
    """
    Generates (object_id, context_id, label_id) triples for a given phase.

    phase='A':      zor is context-dependent per CONTROL_CTX_MAPPING (CTX_RED->red, CTX_BLUE->blue)
                     [i.e. during phase A, zor behaves exactly like the control object]
    phase='B':       zor is CONTEXT-INDEPENDENT and always blue, regardless of context
                     [this is the "overwrite": zor's context-sensitivity is destroyed,
                      it now always says blue -- this IS the behavioral erasure of A]
    phase='B_only':  same as B (control population, zor never had context-dependence)

    Filler objects and the control object behave identically across all phases
    (their generation logic never changes), which is what makes cross-phase/
    cross-population comparison fair.
    """
    def __init__(self, filler_mapping, phase: str, zor_frac: float = 0.12, ctrl_frac: float = 0.12):
        self.filler_mapping = filler_mapping
        self.phase = phase
        assert phase in ("A", "B", "B_only")
        self.zor_frac = zor_frac
        self.ctrl_frac = ctrl_frac

    def _zor_label(self, ctx: str) -> str:
        if self.phase == "A":
            return CONTROL_CTX_MAPPING[ctx]  # context-dependent, same rule as control object
        else:  # B or B_only: context-independent, always blue
            return "blue"

    def sample_batch(self, batch_size: int, rng: np.random.RandomState):
        objs, ctxs, labels = [], [], []
        for _ in range(batch_size):
            ctx = rng.choice(CONTEXTS)
            r = rng.rand()
            if r < self.zor_frac:
                o = SPECIAL_OBJECT
                c = self._zor_label(ctx)
            elif r < self.zor_frac + self.ctrl_frac:
                o = CONTROL_OBJECT
                c = CONTROL_CTX_MAPPING[ctx]
            else:
                o = rng.choice(FILLER_OBJECTS)
                c = self.filler_mapping[o]  # context-independent
            objs.append(OBJ2ID[o])
            ctxs.append(CTX2ID[ctx])
            labels.append(COLOR2ID[c])
        return (torch.tensor(objs, dtype=torch.long),
                torch.tensor(ctxs, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long))

    def full_eval_set(self):
        """Deterministic eval set: every (object, context) combination once."""
        objs, ctxs, labels = [], [], []
        for ctx in CONTEXTS:
            for o in FILLER_OBJECTS:
                objs.append(OBJ2ID[o]); ctxs.append(CTX2ID[ctx]); labels.append(COLOR2ID[self.filler_mapping[o]])
            objs.append(OBJ2ID[CONTROL_OBJECT]); ctxs.append(CTX2ID[ctx]); labels.append(COLOR2ID[CONTROL_CTX_MAPPING[ctx]])
            objs.append(OBJ2ID[SPECIAL_OBJECT]); ctxs.append(CTX2ID[ctx]); labels.append(COLOR2ID[self._zor_label(ctx)])
        return (torch.tensor(objs, dtype=torch.long),
                torch.tensor(ctxs, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long))

    def zor_ctrl_probe_set(self):
        """The 4 canonical (object, context) pairs needed for the
        difference-in-differences v_A construction: zor & control object,
        each in CTX_RED and CTX_BLUE."""
        pairs = [
            (SPECIAL_OBJECT, "CTX_RED"), (SPECIAL_OBJECT, "CTX_BLUE"),
            (CONTROL_OBJECT, "CTX_RED"), (CONTROL_OBJECT, "CTX_BLUE"),
        ]
        objs = torch.tensor([OBJ2ID[o] for o, _ in pairs], dtype=torch.long)
        ctxs = torch.tensor([CTX2ID[c] for _, c in pairs], dtype=torch.long)
        return objs, ctxs

