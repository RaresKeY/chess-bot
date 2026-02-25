from typing import Dict, List, Tuple

import torch
import torch.nn as nn

PAD = "<PAD>"
UNK = "<UNK>"


class NextMoveLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
        winner_embed_dim: int = 8,
        use_winner: bool = True,
    ) -> None:
        super().__init__()
        self.use_winner = use_winner
        dropout = float(max(0.0, dropout))
        lstm_dropout = dropout if int(num_layers) > 1 else 0.0
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=int(num_layers),
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.winner_embed = nn.Embedding(4, winner_embed_dim)
        classifier_in = hidden_dim + (winner_embed_dim if use_winner else 0)
        self.head_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(classifier_in, vocab_size)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor, winner_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embed_dropout(self.token_embed(tokens))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        last_hidden = h[-1]

        if self.use_winner:
            w_emb = self.winner_embed(winner_ids)
            x = torch.cat([last_hidden, w_emb], dim=-1)
        else:
            x = last_hidden
        x = self.head_dropout(x)
        return self.classifier(x)


def build_vocab(samples: List[Dict]) -> Dict[str, int]:
    vocab = {PAD: 0, UNK: 1}
    for sample in samples:
        for tok in sample["context"]:
            if tok not in vocab:
                vocab[tok] = len(vocab)
        for tok in sample["target"]:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def encode_tokens(tokens: List[str], vocab: Dict[str, int]) -> List[int]:
    unk = vocab[UNK]
    return [vocab.get(tok, unk) for tok in tokens]


def winner_to_id(side: str) -> int:
    mapping = {"W": 0, "B": 1, "D": 2, "?": 3}
    return mapping.get(side, 3)


def compute_topk(logits: torch.Tensor, labels: torch.Tensor, ks: Tuple[int, ...]) -> Dict[int, float]:
    out = {}
    for k in ks:
        topk = logits.topk(k, dim=1).indices
        correct = (topk == labels.unsqueeze(1)).any(dim=1).float().mean().item()
        out[k] = correct
    return out
