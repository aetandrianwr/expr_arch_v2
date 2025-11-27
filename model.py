import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RecurrentTransformer(nn.Module):
    """
    Highly optimized for location prediction:
    - Strong user-location interaction
    - Last location emphasis  
    - Efficient attention-based sequence modeling
    """
    def __init__(self, 
                 num_locations=1200,
                 num_users=50,
                 num_weekdays=7,
                 num_start_min_bins=1440,
                 num_diff_bins=100,
                 embed_dim=64,
                 num_heads=4,
                 num_layers=2,
                 num_cycles=2,
                 num_refinements=8,
                 dropout=0.3):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_cycles = num_cycles
        self.num_refinements = num_refinements
        
        # Core embeddings
        self.loc_embed = nn.Embedding(num_locations, embed_dim, padding_idx=0)
        self.user_embed = nn.Embedding(num_users, embed_dim, padding_idx=0)
        
        # USER-LOCATION INTERACTION (critical for 50%!)
        self.user_loc_interaction = nn.Bilinear(embed_dim, embed_dim, embed_dim)
        
        # Context (lighter)
        self.weekday_embed = nn.Embedding(num_weekdays, 16, padding_idx=0)
        self.time_embed = nn.Embedding(48, 24, padding_idx=0)
        self.diff_embed = nn.Embedding(50, 16, padding_idx=0)
        
        # Position
        self.pos_embed = nn.Parameter(torch.randn(1, 100, embed_dim) * 0.02)
        
        # Recurrent state
        self.init_state = nn.Parameter(torch.zeros(1, embed_dim))
        
        # Lightweight transformer
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # State update
        self.state_update = nn.GRUCell(embed_dim, embed_dim)
        
        # LAST LOCATION emphasis
        self.last_loc_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
        # Prediction head - user-conditioned
        self.pred_query = nn.Linear(embed_dim, embed_dim)
        self.pred_key = nn.Linear(embed_dim, embed_dim)
        
        # Direct prediction
        self.direct_pred = nn.Linear(embed_dim, num_locations)
        
        # Ensemble weight
        self.ensemble_weight = nn.Linear(embed_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
    
    def forward(self, loc, user, weekday, start_min, duration, diff, mask):
        B, L = loc.shape
        
        # Core embeddings
        loc_emb = self.loc_embed(loc)  # [B, L, D]
        user_emb = self.user_embed(user[:, 0])  # [B, D] - same user for whole sequence
        
        # USER-LOCATION INTERACTION
        user_expanded = user_emb.unsqueeze(1).expand(-1, L, -1)  # [B, L, D]
        interaction = self.user_loc_interaction(user_expanded, loc_emb)  # [B, L, D]
        
        # Enhanced location representation
        x = loc_emb + interaction
        
        # Add position
        x = x + self.pos_embed[:, :L]
        
        # Attention mask
        pad_mask = ~mask
        
        # Lightweight transformer
        attn_out, _ = self.attn(x, x, x, key_padding_mask=pad_mask)
        x = self.norm1(x + attn_out)
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # Get last valid position
        lengths = mask.sum(1) - 1
        last_idx = lengths.clamp(min=0)
        
        # Extract last location representation
        last_repr = x[torch.arange(B), last_idx]  # [B, D]
        
        # Emphasize last location
        last_enhanced = self.last_loc_proj(last_repr)
        
        # Combine with user
        final_repr = last_enhanced + user_emb
        
        # PREDICTION via similarity to all location embeddings
        query = self.pred_query(final_repr)  # [B, D]
        
        # Get all location embeddings and interact with user
        all_loc_emb = self.loc_embed.weight  # [num_locs, D]
        user_for_all = user_emb.unsqueeze(1)  # [B, 1, D]
        all_loc_expanded = all_loc_emb.unsqueeze(0).expand(B, -1, -1)  # [B, num_locs, D]
        
        # User-location interaction for all locations
        all_interactions = self.user_loc_interaction(
            user_for_all.expand(-1, 1200, -1),
            all_loc_expanded
        )  # [B, num_locs, D]
        
        # Score via dot product
        keys = self.pred_key(all_interactions)  # [B, num_locs, D]
        scores = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1)  # [B, num_locs]
        
        # Direct prediction
        direct_logits = self.direct_pred(final_repr)  # [B, num_locs]
        
        # Ensemble
        weight = torch.sigmoid(self.ensemble_weight(final_repr))  # [B, 1]
        final_logits = weight * scores + (1 - weight) * direct_logits
        
        return final_logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
