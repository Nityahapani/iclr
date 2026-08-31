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
                                 # difference-in-differences subtraction (v1/v2, retained for reference).
SHAM_OBJECT = "fenn"            # sham-history object: undergoes an A-analog phase (phase "C")
                                 # matched to real phase A in every measurable training-dynamics
                                 # property (num examples, steps, loss trajectory, final accuracy),
                                 # but teaches a DIFFERENT binding (fenn=green, not fenn=red) before
                                 # being overwritten to fenn=blue in phase B, same as zor. Used to
                                 # test whether rho_A is specific to WHAT was learned (red) or merely
                                 # to the fact THAT this parameter region underwent an earlier phase.

ALL_OBJECTS = FILLER_OBJECTS + [SPECIAL_OBJECT, CONTROL_OBJECT, SHAM_OBJECT]
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
CONTROL_CTX_MAPPING = {"CTX_RED": "red", "CTX_BLUE": "blue"}

# SHAM_OBJECT's phase-C mapping: context-dependent like zor's phase-A mapping,
# but teaches a DIFFERENT color (green, not red) for CTX_RED. In phase C,
# CTX_BLUE still maps to blue (matching zor's phase-A rule structurally) so
# that only the "what does CTX_RED mean" binding differs between zor and fenn.
SHAM_CTX_MAPPING = {"CTX_RED": "green", "CTX_BLUE": "blue"}


class PhaseDataset:
    """
    Generates (object_id, context_id, label_id) triples for a given phase.

    phase='A':      zor is context-dependent per CONTROL_CTX_MAPPING (CTX_RED->red, CTX_BLUE->blue).
                     SHAM_OBJECT (fenn) is context-dependent per SHAM_CTX_MAPPING (CTX_RED->green,
                     CTX_BLUE->blue) -- this is phase "C": matched in structure/frequency/steps to
                     phase A, but teaches a different binding.
    phase='B':       BOTH zor and fenn become context-INDEPENDENT and always predict blue,
                     regardless of context (the overwrite/erasure phase, applied identically
                     to both the real-history and sham-history objects).
    phase='B_only':  same generation logic as B (from-scratch control population; zor and fenn
                     never had ANY earlier phase in this population).

    Filler objects and the control object (vex) behave identically across all
    phases (their generation logic never changes).
    """
    def __init__(self, filler_mapping, phase: str, zor_frac: float = 0.10,
                 ctrl_frac: float = 0.10, sham_frac: float = 0.10):
        self.filler_mapping = filler_mapping
        self.phase = phase
        assert phase in ("A", "B", "B_only")
        self.zor_frac = zor_frac
        self.ctrl_frac = ctrl_frac
        self.sham_frac = sham_frac

    def _zor_label(self, ctx: str) -> str:
        if self.phase == "A":
            return CONTROL_CTX_MAPPING[ctx]
        else:
            return "blue"

    def _fenn_label(self, ctx: str) -> str:
        if self.phase == "A":  # phase "C" for fenn happens concurrently with phase A for zor
            return SHAM_CTX_MAPPING[ctx]
        else:
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
            elif r < self.zor_frac + self.ctrl_frac + self.sham_frac:
                o = SHAM_OBJECT
                c = self._fenn_label(ctx)
            else:
                o = rng.choice(FILLER_OBJECTS)
                c = self.filler_mapping[o]
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
            objs.append(OBJ2ID[SHAM_OBJECT]); ctxs.append(CTX2ID[ctx]); labels.append(COLOR2ID[self._fenn_label(ctx)])
        return (torch.tensor(objs, dtype=torch.long),
                torch.tensor(ctxs, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long))

    def zor_ctrl_probe_set(self):
        """The 4 canonical (object, context) pairs needed for the
        difference-in-differences v_A construction (v1/v2, retained for reference)."""
        pairs = [
            (SPECIAL_OBJECT, "CTX_RED"), (SPECIAL_OBJECT, "CTX_BLUE"),
            (CONTROL_OBJECT, "CTX_RED"), (CONTROL_OBJECT, "CTX_BLUE"),
        ]
        objs = torch.tensor([OBJ2ID[o] for o, _ in pairs], dtype=torch.long)
        ctxs = torch.tensor([CTX2ID[c] for _, c in pairs], dtype=torch.long)
        return objs, ctxs

