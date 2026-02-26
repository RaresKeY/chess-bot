from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

PAD = "<PAD>"
UNK = "<UNK>"

SIDE_TO_MOVE_WHITE = 0
SIDE_TO_MOVE_BLACK = 1


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
        phase_embed_dim: int = 8,
        use_phase: bool = False,
        side_to_move_embed_dim: int = 4,
        use_side_to_move: bool = False,
    ) -> None:
        super().__init__()
        self.use_winner = use_winner
        self.use_phase = bool(use_phase)
        self.use_side_to_move = bool(use_side_to_move)
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
        self.phase_embed = nn.Embedding(4, phase_embed_dim) if self.use_phase else None
        self.side_to_move_embed = nn.Embedding(2, side_to_move_embed_dim) if self.use_side_to_move else None
        classifier_in = hidden_dim
        if use_winner:
            classifier_in += winner_embed_dim
        if self.use_phase:
            classifier_in += phase_embed_dim
        if self.use_side_to_move:
            classifier_in += side_to_move_embed_dim
        self.head_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(classifier_in, vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        winner_ids: torch.Tensor,
        phase_ids: Optional[torch.Tensor] = None,
        side_to_move_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        emb = self.embed_dropout(self.token_embed(tokens))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        last_hidden = h[-1]

        features = [last_hidden]
        if self.use_winner:
            w_emb = self.winner_embed(winner_ids)
            features.append(w_emb)
        if self.use_phase:
            if phase_ids is None:
                phase_ids = torch.zeros_like(winner_ids)
            features.append(self.phase_embed(phase_ids.clamp(min=0, max=3)))
        if self.use_side_to_move:
            if side_to_move_ids is None:
                side_to_move_ids = (lengths.to(last_hidden.device) % 2).long()
            features.append(self.side_to_move_embed(side_to_move_ids.clamp(min=0, max=1)))
        x = torch.cat(features, dim=-1)
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


def side_to_move_id_from_context_len(context_len: int) -> int:
    try:
        value = int(context_len)
    except Exception:
        return SIDE_TO_MOVE_WHITE
    return SIDE_TO_MOVE_BLACK if (value % 2) else SIDE_TO_MOVE_WHITE


def compute_topk(logits: torch.Tensor, labels: torch.Tensor, ks: Tuple[int, ...]) -> Dict[int, float]:
    out = {}
    for k in ks:
        topk = logits.topk(k, dim=1).indices
        correct = (topk == labels.unsqueeze(1)).any(dim=1).float().mean().item()
        out[k] = correct
    return out
