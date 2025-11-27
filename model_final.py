import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RecurrentTransformer(nn.Module):
    """
    Optimized architecture for next-location prediction.
    Focus on what works: strong location embeddings, effective sequence modeling, and smart pooling.
    """
    def __init__(self, 
                 num_locations=1200,
                 num_users=50,
                 num_weekdays=7,
                 num_start_min_bins=1440,
                 num_diff_bins=100,
                 embed_dim=128,
                 num_heads=8,
                 num_layers=3,
                 num_cycles=2,
                 num_refinements=8,
                 dropout=0.2):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_cycles = num_cycles
        self.num_refinements = num_refinements
        
        # Strong location embedding
        self.loc_embed = nn.Embedding(num_locations, embed_dim, padding_idx=0)
        
        # Compact context features
        self.user_embed = nn.Embedding(num_users, 16, padding_idx=0)
        self.weekday_embed = nn.Embedding(num_weekdays, 8, padding_idx=0)
        self.time_embed = nn.Embedding(48, 16, padding_idx=0)  
        self.diff_embed = nn.Embedding(20, 8, padding_idx=0)
        
        # Simple context fusion
        self.context_fc = nn.Linear(16 + 8 + 16 + 8, embed_dim // 2)
        
        # Learned position
        self.pos_embed = nn.Parameter(torch.zeros(1, 100, embed_dim))
        
        # Recurrent memory states
        self.memory_init = nn.ParameterList([
            nn.Parameter(torch.zeros(1, embed_dim)) for _ in range(num_cycles)
        ])
        
        # Core Transformer
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 3,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Memory update via attention
        self.mem_attn = nn.MultiheadAttention(embed_dim, 4, dropout=dropout, batch_first=True)
        self.mem_norm = nn.LayerNorm(embed_dim)
        
        # Output
        self.output = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_locations)
        )
        
        self._init()
    
    def _init(self):
        nn.init.normal_(self.pos_embed, std=0.02)
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
        
        # Core: Location embeddings
        x = self.loc_embed(loc)
        
        # Context
        ctx = torch.cat([
            self.user_embed(user),
            self.weekday_embed(weekday),
            self.time_embed((start_min // 30).clamp(0, 47)),
            self.diff_embed(diff.clamp(0, 19))
        ], dim=-1)
        ctx = self.context_fc(ctx)
        
        # Fuse: location + context + position
        x = x + F.pad(ctx, (0, self.embed_dim - ctx.size(-1)))
        x = x + self.pos_embed[:, :L]
        
        # Mask
        pad_mask = ~mask
        
        # Recurrent processing
        mem = self.memory_init[0].expand(B, -1)
        
        for cycle in range(self.num_cycles):
            # Inject memory into sequence
            x_mem = x + mem.unsqueeze(1) * 0.3
            
            # Transform
            h = x_mem
            for layer in self.layers:
                h = layer(h, src_key_padding_mask=pad_mask)
            h = self.norm(h)
            
            # Update memory
            mem_upd, _ = self.mem_attn(
                mem.unsqueeze(1), h, h,
                key_padding_mask=pad_mask
            )
            mem = self.mem_norm(mem + mem_upd.squeeze(1))
            
            # Next cycle
            if cycle + 1 < self.num_cycles:
                mem = mem + self.memory_init[cycle + 1].expand(B, -1)
        
        # Pool: use last valid token + memory
        lengths = mask.sum(1) - 1
        last_hidden = h[torch.arange(B), lengths.clamp(min=0)]
        final = last_hidden + mem
        
        # Refinement
        for _ in range(self.num_refinements):
            final = final + 0.1 * self.output[:-1](final.detach())
        
        # Predict
        logits = self.output(final)
        
        return logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
