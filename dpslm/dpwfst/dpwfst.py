from typing import Optional

import k2
import torch
import torch.nn as nn


class DPWFSTQuantizer(nn.Module):
    def __init__(self, codebook_size, codebook_dim):
        super().__init__()
        self.register_buffer(
            "codebook",
            torch.randn(codebook_size, codebook_dim),
        )

    @classmethod
    def from_codebook(cls, codebook: torch.Tensor):
        codebook_size, codebook_dim = codebook.shape
        model = cls(codebook_size, codebook_dim)
        model.codebook.data = codebook.clone()
        return model

    def forward(
        self,
        features: torch.Tensor,
        lmbda: float,
        num_neighbors: Optional[int] = None,
    ):
        return dpwfst(
            features=features,
            codebook=self.codebook,
            lmbda=lmbda,
            num_neighbors=num_neighbors,
        )


def dpwfst(
    features: torch.Tensor,
    codebook: torch.Tensor,
    lmbda: float,
    num_neighbors: Optional[int] = None,
):
    if features.dim() != 2:
        raise NotImplementedError("Only works for 2D input")

    assert features.device == codebook.device
    device = features.device

    if num_neighbors is None:
        num_neighbors = codebook.shape[0]

    distances = torch.cdist(features, codebook, p=2.0) ** 2
    top_k_distances, top_k_indices = torch.topk(
        distances, k=num_neighbors, dim=1, largest=False
    )

    arcs1, aux_labels1, scores1 = build_initial_transition_arcs(
        top_k_distances, top_k_indices
    )
    arcs2, aux_labels2, scores2 = build_intermediate_transition_arcs(
        top_k_distances, top_k_indices, lmbda
    )
    arcs3, aux_labels3, scores3 = build_final_transition_arcs(
        T=distances.shape[0], K=num_neighbors, device=device
    )

    arcs = torch.cat([arcs1, arcs2, arcs3], dim=0)
    aux_labels = torch.cat([aux_labels1, aux_labels2, aux_labels3], dim=0)
    scores = torch.cat([scores1, scores2, scores3], dim=0)

    # Create FSA
    fsa = k2.Fsa(arcs)

    # Now add the true float scores
    fsa.scores = scores

    # Define the auxilary labels. This can be anything. Simply using labels for now.
    fsa.aux_labels = aux_labels

    fsa_vec = k2.create_fsa_vec([fsa])

    # Shortest path
    best = k2.shortest_path(fsa_vec, use_double_scores=True)
    units = best.aux_labels[best.aux_labels != -1]

    return units


def build_initial_transition_arcs(top_k_distances, top_k_indices):
    T, K = top_k_distances.shape
    device = top_k_indices.device
    src = torch.zeros(K, dtype=torch.int32, device=device)
    dest = torch.arange(1, K + 1, dtype=torch.int32, device=device)
    labels = top_k_indices[0].to(torch.int32)
    aux_labels = top_k_indices[0].to(torch.long)
    scores = -top_k_distances[0]
    dummy_scores = torch.zeros_like(scores, dtype=torch.int32, device=device)
    arcs = torch.stack([src, dest, labels, dummy_scores], dim=1)
    return arcs, aux_labels, scores


def build_intermediate_transition_arcs(top_k_distances, top_k_indices, lmbda):
    T, K = top_k_indices.shape
    device = top_k_indices.device
    src = (
        torch.arange(1, (T - 1) * K + 1, dtype=torch.int32, device=device)
        .view(T - 1, K)
        .unsqueeze(-1)
        .expand(T - 1, K, K)
        .flatten()
    )
    dest = (
        torch.arange(K + 1, T * K + 1, dtype=torch.int32, device=device)
        .view(T - 1, K)
        .unsqueeze(1)
        .expand(T - 1, K, K)
        .flatten()
    )
    labels = (
        top_k_indices[1:].to(torch.int32).unsqueeze(1).expand(T - 1, K, K).flatten()
    )
    labels_from = (
        top_k_indices[:-1].to(torch.int32).unsqueeze(1).expand(T - 1, K, K).flatten()
    )
    aux_labels = torch.where(
        labels_from == labels, -1, labels
    )  # output no label (-1) when coming from the same label
    quant_scores = -top_k_distances[1:].unsqueeze(1).expand(T - 1, K, K).flatten()
    duration_scores = (
        top_k_indices[:-1].unsqueeze(-1).expand(T - 1, K, K).flatten()
        == top_k_indices[1:].unsqueeze(1).expand(T - 1, K, K).flatten()
    ).to(torch.float32) * lmbda
    scores = quant_scores + duration_scores
    dummy_scores = torch.zeros_like(scores, dtype=torch.int32, device=device)
    arcs = torch.stack([src, dest, labels, dummy_scores], dim=1)
    return arcs, aux_labels, scores


def build_final_transition_arcs(T, K, device):
    src = torch.arange((T - 1) * K + 1, T * K + 1, dtype=torch.int32, device=device)
    dest = torch.full_like(src, T * K + 1, dtype=torch.int32, device=device)
    labels = torch.full_like(src, -1, dtype=torch.int32, device=device)
    aux_labels = torch.full_like(src, -1, dtype=torch.long, device=device)
    scores = torch.zeros_like(src, dtype=torch.float32, device=device)
    dummy_scores = torch.zeros_like(scores, dtype=torch.int32, device=device)
    arcs = torch.stack([src, dest, labels, dummy_scores], dim=1)
    return arcs, aux_labels, scores
