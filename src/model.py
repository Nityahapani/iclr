"""
Tiny model for the archaeology experiment.

Two-input model: object embedding + context embedding are concatenated,
then passed through an MLP. This gives us a genuine context dimension so we
can construct difference-in-differences directions (h(zor,red_ctx) -
h(zor,blue_ctx)) rather than only having a single object embedding to probe.
"""
import torch
import torch.nn as nn


class TinyClassifier(nn.Module):
    def __init__(self, vocab_size, context_vocab_size, num_classes,
                 embed_dim=16, ctx_embed_dim=8, hidden_dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.ctx_embed = nn.Embedding(context_vocab_size, ctx_embed_dim)
        self.fc1 = nn.Linear(embed_dim + ctx_embed_dim, hidden_dim)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def hidden(self, obj_ids, ctx_ids):
        e = self.embed(obj_ids)
        c = self.ctx_embed(ctx_ids)
        combined = torch.cat([e, c], dim=-1)
        h = self.act(self.fc1(combined))
        return h

    def forward(self, obj_ids, ctx_ids):
        h = self.hidden(obj_ids, ctx_ids)
        return self.fc2(h)

    def hidden_from_h(self, h):
        """Passthrough for interventions that directly modify h."""
        return h

    def get_flat_params(self):
        return torch.cat([p.reshape(-1) for p in self.parameters()])

    def set_flat_params(self, flat):
        idx = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(flat[idx:idx + n].reshape(p.shape))
            idx += n
