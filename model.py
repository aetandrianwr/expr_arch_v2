import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RecurrentTransformer(nn.Module):
    """
    Pointer network + generation hybrid for location prediction.
    PRIMARY: Copy from input (69% targets in sequence!)
    SECONDARY: Generate new locations
    """
    def __init__(self, 
                 num_locations=1200,
                 num_users=50,
                 num_weekdays=7,
                 num_start_min_bins=1440,
                 num_diff_bins=100,
                 embed_dim=80,
                 num_heads=4,
                 num_layers=2,
                 num_cycles=2,
                 num_refinements=8,
                 dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_cycles = num_cycles
        self.num_refinements = num_refinements
        
        # Strong location embedding
        self.loc_embed = nn.Embedding(num_locations, embed_dim, padding_idx=0)
        
        # User-specific embedding (important!)
        self.user_embed = nn.Embedding(num_users, 32, padding_idx=0)
        self.weekday_embed = nn.Embedding(num_weekdays, 8, padding_idx=0)
        self.time_embed = nn.Embedding(48, 16, padding_idx=0)
        self.diff_embed = nn.Embedding(50, 8, padding_idx=0)
        
        # Context
        self.ctx_fc = nn.Linear(32 + 8 + 16 + 8, embed_dim)
        
        # Position
        self.pos_embed = nn.Parameter(torch.randn(1, 100, embed_dim) * 0.02)
        
        # Recurrent memory
        self.mem_init = nn.ParameterList([
            nn.Parameter(torch.zeros(1, embed_dim)) for _ in range(num_cycles)
        ])
        
        # Transformer
        self.transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ) for _ in range(num_layers)
        ])
        
        # Memory update
        self.mem_attn = nn.MultiheadAttention(embed_dim, 4, dropout=dropout, batch_first=True)
        
        # POINTER NETWORK: Strong copy mechanism
        self.pointer_query = nn.Linear(embed_dim, embed_dim)
        self.pointer_key = nn.Linear(embed_dim, embed_dim)
        
        # User preference modeling
        self.user_pref = nn.Linear(32, embed_dim)
        
        # Generation (for 31% not in sequence)
        self.gen_head = nn.Linear(embed_dim, num_locations)
        
        # Copy/generate balance (per-sample learned)
        self.copy_gen_balance = nn.Sequential(
            nn.Linear(embed_dim + 32, 1),
            nn.Sigmoid()
        )
        
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
        
        # Embeddings
        loc_emb = self.loc_embed(loc)
        user_emb = self.user_embed(user)
        
        ctx = torch.cat([
            user_emb,
            self.weekday_embed(weekday),
            self.time_embed(torch.div(start_min, 30, rounding_mode='floor').clamp(0, 47)),
            self.diff_embed(diff.clamp(0, 49))
        ], dim=-1)
        ctx_proj = self.ctx_fc(ctx)
        
        # Input
        x = loc_emb + ctx_proj + self.pos_embed[:, :L]
        
        # Mask
        pad_mask = ~mask
        
        # Recurrent cycles
        mem = self.mem_init[0].expand(B, -1)
        
        for cycle in range(self.num_cycles):
            # Add memory
            x_with_mem = x + mem.unsqueeze(1) * 0.3
            
            # Transformer
            h = x_with_mem
            for layer in self.transformer:
                h = layer(h, src_key_padding_mask=pad_mask)
            
            # Update memory
            mem_new, _ = self.mem_attn(
                mem.unsqueeze(1), h, h,
                key_padding_mask=pad_mask
            )
            mem = mem + mem_new.squeeze(1)
            
            if cycle + 1 < self.num_cycles:
                mem = mem + self.mem_init[cycle + 1].expand(B, -1)
        
        # Final representation
        lengths = mask.sum(1) - 1
        last_h = h[torch.arange(B), lengths.clamp(min=0)]
        final_repr = last_h + mem
        
        # User preference
        user_pref = self.user_pref(user_emb[:, 0, :])  # [B, D]
        final_repr = final_repr + user_pref
        
        # POINTER NETWORK: Copy mechanism
        query = self.pointer_query(final_repr).unsqueeze(1)  # [B, 1, D]
        keys = self.pointer_key(h)  # [B, L, D]
        
        # Pointer scores
        pointer_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1)  # [B, L]
        pointer_scores = pointer_scores / math.sqrt(self.embed_dim)
        pointer_scores = pointer_scores.masked_fill(~mask, -1e9)
        pointer_probs = F.softmax(pointer_scores, dim=-1)  # [B, L]
        
        # Map to vocabulary with STRONG aggregation
        copy_dist = torch.zeros(B, 1200, device=loc.device, dtype=pointer_probs.dtype)
        
        # Scatter-add: accumulate probabilities for repeated locations
        copy_dist.scatter_add_(1, loc, pointer_probs)
        
        # Add small boost for locations that appear multiple times (pattern reinforcement)
        for b in range(B):
            loc_counts = torch.bincount(loc[b, mask[b]], minlength=1200)
            boost = (loc_counts > 0).float() * 0.1
            copy_dist[b] = copy_dist[b] + boost[:1200]
        
        copy_logits = torch.log(copy_dist + 1e-10)
        
        # GENERATION: For new locations
        gen_logits = self.gen_head(final_repr)
        
        # Balance (learned per sample based on sequence and user)
        balance = self.copy_gen_balance(torch.cat([final_repr, user_emb[:, 0, :]], dim=-1))
        
        # Combined (bias towards copying since 69% targets are in sequence)
        copy_weight = 0.8 + balance * 0.2  # 0.8-1.0 range, biased to copy
        final_logits = copy_weight * copy_logits + (1 - copy_weight) * gen_logits
        
        return final_logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
