"""
GISMo — Graph-based Ingredient Substitution Module
Implementation based on: "Learning to Substitute Ingredients in Recipes"
(Fatemi et al., 2023, arXiv:2302.07960)

Architecture:
  1. Ingredient Encoder (IE): GIN layers over FlavorGraph
  2. Context Encoder (CE): Average of recipe ingredient embeddings
  3. Ingredient Substitution Decoder (ISD): MLP scoring (source || candidate || context) → score

Training:
  Contrastive loss: maximize score for true substitution, minimize for negatives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.data import Data


class GINLayer(nn.Module):
    """Graph Isomorphism Network layer (Xu et al., 2019)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x, edge_index, edge_weight=None):
        # Message passing: aggregate neighbor features
        row, col = edge_index
        if edge_weight is not None:
            agg = torch.zeros_like(x)
            agg.index_add_(0, row, x[col] * edge_weight.unsqueeze(-1))
        else:
            agg = torch.zeros_like(x)
            agg.index_add_(0, row, x[col])

        out = (1 + self.eps) * x + agg
        return self.mlp(out)


class IngredientEncoder(nn.Module):
    """Ingredient Encoder: embedding layer + GIN layers over ingredient graph."""

    def __init__(self, num_ingredients, embed_dim=300, num_layers=2, dropout=0.25):
        super().__init__()
        self.embedding = nn.Embedding(num_ingredients, embed_dim)
        self.gin_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gin_layers.append(GINLayer(embed_dim, embed_dim))
        self.dropout = dropout

    def forward(self, x_ids, edge_index, edge_weight=None):
        """
        Args:
            x_ids: (num_nodes,) ingredient indices
            edge_index: (2, num_edges) graph connectivity
            edge_weight: (num_edges,) optional edge weights
        Returns:
            h: (num_nodes, embed_dim) ingredient embeddings
        """
        h = self.embedding(x_ids)
        for gin in self.gin_layers:
            h = gin(h, edge_index, edge_weight)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class ContextEncoder(nn.Module):
    """Context Encoder: average ingredient embeddings in a recipe."""

    def __init__(self, embed_dim=300):
        super().__init__()
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, ingredient_embeddings, mask=None):
        """
        Args:
            ingredient_embeddings: (batch, max_ingredients, embed_dim)
            mask: (batch, max_ingredients) boolean mask for valid ingredients
        Returns:
            context: (batch, embed_dim)
        """
        if mask is not None:
            masked = ingredient_embeddings * mask.unsqueeze(-1).float()
            counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
            context = masked.sum(dim=1) / counts
        else:
            context = ingredient_embeddings.mean(dim=1)
        return self.proj(context)


class SubstitutionDecoder(nn.Module):
    """Ingredient Substitution Decoder: MLP that scores (source, candidate, context) → score."""

    def __init__(self, embed_dim=300, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, source_emb, candidate_emb, context_emb):
        """
        Args:
            source_emb: (batch, embed_dim) source ingredient embedding
            candidate_emb: (batch, embed_dim) or (batch, num_candidates, embed_dim)
            context_emb: (batch, embed_dim) recipe context embedding
        Returns:
            scores: (batch,) or (batch, num_candidates)
        """
        if candidate_emb.dim() == 3:
            # Score multiple candidates at once
            batch, num_cands, dim = candidate_emb.shape
            source_exp = source_emb.unsqueeze(1).expand(-1, num_cands, -1)
            context_exp = context_emb.unsqueeze(1).expand(-1, num_cands, -1)
            concat = torch.cat([source_exp, candidate_emb, context_exp], dim=-1)
            return self.mlp(concat).squeeze(-1)  # (batch, num_candidates)
        else:
            concat = torch.cat([source_emb, candidate_emb, context_emb], dim=-1)
            return self.mlp(concat).squeeze(-1)  # (batch,)


class GISMo(nn.Module):
    """
    Full GISMo model: IE + CE + ISD.

    Given a FlavorGraph, a recipe context, and a source ingredient,
    scores all candidate ingredients as potential substitutions.
    """

    def __init__(self, num_ingredients, embed_dim=300, num_gin_layers=2, decoder_hidden=256, dropout=0.25):
        super().__init__()
        self.ingredient_encoder = IngredientEncoder(
            num_ingredients, embed_dim, num_gin_layers, dropout
        )
        self.context_encoder = ContextEncoder(embed_dim)
        self.decoder = SubstitutionDecoder(embed_dim, decoder_hidden)
        self.num_ingredients = num_ingredients
        self.embed_dim = embed_dim

    def encode_graph(self, node_ids, edge_index, edge_weight=None):
        """Run ingredient encoder over the full graph. Returns all node embeddings."""
        return self.ingredient_encoder(node_ids, edge_index, edge_weight)

    def get_context(self, all_embeddings, recipe_ingredient_ids):
        """
        Compute recipe context from ingredient IDs.
        Args:
            all_embeddings: (num_nodes, embed_dim) from encode_graph
            recipe_ingredient_ids: list of lists of ingredient indices
        Returns:
            context: (batch, embed_dim)
        """
        batch_size = len(recipe_ingredient_ids)
        max_len = max(len(ids) for ids in recipe_ingredient_ids)
        device = all_embeddings.device

        padded = torch.zeros(batch_size, max_len, self.embed_dim, device=device)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)

        for i, ids in enumerate(recipe_ingredient_ids):
            for j, idx in enumerate(ids):
                padded[i, j] = all_embeddings[idx]
                mask[i, j] = True

        return self.context_encoder(padded, mask)

    def score(self, all_embeddings, source_ids, candidate_ids, context):
        """
        Score substitution candidates.
        Args:
            all_embeddings: (num_nodes, embed_dim)
            source_ids: (batch,) source ingredient indices
            candidate_ids: (batch, num_candidates) candidate indices
            context: (batch, embed_dim) recipe context
        Returns:
            scores: (batch, num_candidates)
        """
        source_emb = all_embeddings[source_ids]  # (batch, embed_dim)
        cand_emb = all_embeddings[candidate_ids]  # (batch, num_candidates, embed_dim)
        return self.decoder(source_emb, cand_emb, context)

    def forward(self, node_ids, edge_index, edge_weight,
                source_ids, positive_ids, negative_ids, recipe_ingredient_ids):
        """
        Full forward pass for training with contrastive loss.
        Returns loss scalar.
        """
        all_emb = self.encode_graph(node_ids, edge_index, edge_weight)
        context = self.get_context(all_emb, recipe_ingredient_ids)

        pos_scores = self.decoder(
            all_emb[source_ids], all_emb[positive_ids], context
        )  # (batch,)

        neg_emb = all_emb[negative_ids]  # (batch, num_neg, embed_dim)
        neg_scores = self.decoder(
            all_emb[source_ids], neg_emb, context
        )  # (batch, num_neg)

        # Contrastive loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        pos_exp = torch.exp(pos_scores)  # (batch,)
        neg_exp = torch.exp(neg_scores).sum(dim=1)  # (batch,)
        loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-8)).mean()

        return loss
