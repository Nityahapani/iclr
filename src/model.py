"""
Tiny model for the archaeology experiment.

We deliberately use a small MLP-on-embedding classifier rather than a full
transformer: the "killer experiment" needs us to compute Hessian/Fisher
curvature many times (once per candidate direction, per model, per condition),
which is only tractable at small scale on CPU. The task itself (object -> color)
does not require attention or sequence modeling, so a small MLP is a faithful,
honest choice for this task, not a simplification that changes what's being
tested. All key I/O contracts (embedding lookup -> hidden repr -> logits)
generalize to larger architectures for follow-up work.
"""
import torch
import torch.nn as nn


class TinyClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def hidden(self, obj_ids):
        e = self.embed(obj_ids)
        h = self.act(self.fc1(e))
        return h

    def forward(self, obj_ids):
        h = self.hidden(obj_ids)
        return self.fc2(h)

    def get_flat_params(self):
        return torch.cat([p.reshape(-1) for p in self.parameters()])

    def set_flat_params(self, flat):
        idx = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(flat[idx:idx + n].reshape(p.shape))
            idx += n
