import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RecurrentTransformer(nn.Module):
    """
    Optimized for GeoLife: Copy mechanism + Transition modeling + Recurrent Transformer
    Key insight: 69% of targets are in the input sequence!
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
                 num_refinements=12,
                 dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_cycles = num_cycles
        self.num_refinements = num_refinements
        
        # Location embedding (most important)
        self.loc_embed = nn.Embedding(num_locations, embed_dim, padding_idx=0)
        
        # Context embeddings (smaller)
        self.user_embed = nn.Embedding(num_users, 24, padding_idx=0)
        self.weekday_embed = nn.Embedding(num_weekdays, 8, padding_idx=0)
        self.time_embed = nn.Embedding(48, 16, padding_idx=0)  # 30min buckets
        self.diff_embed = nn.Embedding(50, 8, padding_idx=0)
        
        # Context fusion
        ctx_dim = 24 + 8 + 16 + 8
        self.ctx_proj = nn.Linear(ctx_dim, embed_dim)
        
        # Positional encoding
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
        self.norm = nn.LayerNorm(embed_dim)
        
        # Memory attention
        self.mem_attn = nn.MultiheadAttention(embed_dim, 4, dropout=dropout, batch_first=True)
        
        # CRITICAL: Copy mechanism - predict which input location is the target
        self.copy_query = nn.Linear(embed_dim, embed_dim)
        self.copy_key = nn.Linear(embed_dim, embed_dim)
        
        # Transition modeling: last_loc -> target
        self.transition_embed = nn.Embedding(num_locations, embed_dim // 2, padding_idx=0)
        self.transition_fc = nn.Linear(embed_dim // 2, embed_dim)
        
        # Combine copy + generate
        self.combine_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, 1),
            nn.Sigmoid()
        )
        
        # Generation head
        self.generate_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_locations)
        )
        
        # Refinement
        self.refine_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
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
        loc_emb = self.loc_embed(loc)  # [B, L, D]
        
        # Context
        ctx = torch.cat([
            self.user_embed(user),
            self.weekday_embed(weekday),
            self.time_embed((start_min // 30).clamp(0, 47)),
            self.diff_embed(diff.clamp(0, 49))
        ], dim=-1)
        ctx = self.ctx_proj(ctx)
        
        # Combine
        x = loc_emb + ctx + self.pos_embed[:, :L]
        
        # Padding mask
        pad_mask = ~mask
        
        # Recurrent cycles
        mem = self.mem_init[0].expand(B, -1)
        
        for cycle in range(self.num_cycles):
            # Add memory to sequence
            x_with_mem = x + mem.unsqueeze(1) * 0.2
            
            # Transformer
            h = x_with_mem
            for layer in self.transformer:
                h = layer(h, src_key_padding_mask=pad_mask)
            h = self.norm(h)
            
            # Update memory
            mem_new, _ = self.mem_attn(
                mem.unsqueeze(1), h, h,
                key_padding_mask=pad_mask
            )
            mem = mem + mem_new.squeeze(1)
            
            if cycle + 1 < self.num_cycles:
                mem = mem + self.mem_init[cycle + 1].expand(B, -1)
        
        # Get representation
        # Use last valid position
        lengths = mask.sum(1) - 1
        last_h = h[torch.arange(B), lengths.clamp(min=0)]
        
        # Combine with memory
        combined = last_h + mem
        
        # Refinement loop
        for _ in range(self.num_refinements):
            refined = self.refine_mlp(combined.detach())
            combined = combined + 0.1 * refined
        
        # COPY MECHANISM: Calculate attention over input sequence
        query = self.copy_query(combined).unsqueeze(1)  # [B, 1, D]
        keys = self.copy_key(h)  # [B, L, D]
        
        # Copy scores over input locations
        copy_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / math.sqrt(self.embed_dim)  # [B, L]
        copy_scores = copy_scores.masked_fill(~mask, -1e9)
        
        # Convert to location-level scores
        copy_logits = torch.zeros(B, 1200, device=loc.device)
        for b in range(B):
            for i in range(L):
                if mask[b, i]:
                    loc_id = loc[b, i].item()
                    copy_logits[b, loc_id] = torch.max(
                        copy_logits[b, loc_id],
                        copy_scores[b, i]
                    )
        
        # TRANSITION: Last location -> target
        last_locs = loc[torch.arange(B), lengths.clamp(min=0)]
        trans_emb = self.transition_embed(last_locs)
        trans_feat = self.transition_fc(trans_emb)
        
        # GENERATION: Standard classification
        gen_logits = self.generate_head(combined + trans_feat)
        
        # COMBINE: Gate between copy and generate
        gate = self.combine_gate(torch.cat([combined, trans_feat], dim=-1))  # [B, 1]
        
        # Final logits
        final_logits = gate * copy_logits + (1 - gate) * gen_logits
        
        return final_logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
