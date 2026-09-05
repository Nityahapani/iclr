"""
Multi-instance extension of the color task, built specifically to test
whether J_A generalizes to HELD-OUT inputs, not just the canonical zor.

A POOL of N objects all follow the identical context-dependent rule during
phase A (CTX_RED->red, CTX_BLUE->blue) and are all overwritten to
context-independent blue in phase B. J_A is constructed from a TRAIN
subset; B_t, P_A(t), R_A(t) are measured on a disjoint HELD-OUT subset.
"""
import random
import numpy as np
import torch

COLORS = ["red", "blue", "green", "yellow", "purple", "orange"]
COLOR2ID = {c: i for i, c in enumerate(COLORS)}
CONTEXTS = ["CTX_RED", "CTX_BLUE"]
CTX2ID = {c: i for i, c in enumerate(CONTEXTS)}

N_POOL = 40
POOL_OBJECTS = [f"zorlike_{i}" for i in range(N_POOL)]
TRAIN_POOL = POOL_OBJECTS[:20]
HELDOUT_POOL = POOL_OBJECTS[20:]

N_FILLER = 16
FILLER_OBJECTS = [f"filler_{i}" for i in range(N_FILLER)]

ALL_OBJECTS = POOL_OBJECTS + FILLER_OBJECTS
OBJ2ID = {o: i for i, o in enumerate(ALL_OBJECTS)}
VOCAB_SIZE = len(ALL_OBJECTS)
NUM_CLASSES = len(COLORS)
CONTEXT_VOCAB_SIZE = len(CONTEXTS)

POOL_CTX_MAPPING = {"CTX_RED": "red", "CTX_BLUE": "blue"}


def make_filler_mapping(seed):
    rng = random.Random(seed)
    return {o: rng.choice(COLORS) for o in FILLER_OBJECTS}


class MultiInstanceDataset:
    def __init__(self, filler_mapping, phase, pool_frac=0.5):
        self.filler_mapping = filler_mapping
        self.phase = phase
        assert phase in ("A", "B", "B_only")
        self.pool_frac = pool_frac

    def _pool_label(self, ctx):
        return POOL_CTX_MAPPING[ctx] if self.phase == "A" else "blue"

    def sample_batch(self, batch_size, rng):
        objs, ctxs, labels = [], [], []
        for _ in range(batch_size):
            ctx = rng.choice(CONTEXTS)
            if rng.rand() < self.pool_frac:
                o = rng.choice(POOL_OBJECTS)
                c = self._pool_label(ctx)
            else:
                o = rng.choice(FILLER_OBJECTS)
                c = self.filler_mapping[o]
            objs.append(OBJ2ID[o]); ctxs.append(CTX2ID[ctx]); labels.append(COLOR2ID[c])
        return (torch.tensor(objs, dtype=torch.long), torch.tensor(ctxs, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long))

    def full_eval_set(self):
        objs, ctxs, labels = [], [], []
        for ctx in CONTEXTS:
            for o in FILLER_OBJECTS:
                objs.append(OBJ2ID[o]); ctxs.append(CTX2ID[ctx]); labels.append(COLOR2ID[self.filler_mapping[o]])
            for o in POOL_OBJECTS:
                objs.append(OBJ2ID[o]); ctxs.append(CTX2ID[ctx]); labels.append(COLOR2ID[self._pool_label(ctx)])
        return (torch.tensor(objs, dtype=torch.long), torch.tensor(ctxs, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long))
