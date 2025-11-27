import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RecurrentTransformer(nn.Module):
    """
    Simplified effective model focusing on what matters for location prediction.
    """
    def __init__(self, 
                 num_locations=1200,
                 num_users=50,
                 num_weekdays=7,
                 num_start_min_bins=1440,
                 num_diff_bins=100,
                 embed_dim=160,
                 num_heads=8,
                 num_layers=4,
                 num_cycles=3,
                 num_refinements=12,
                 dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_cycles = num_cycles
        self.num_refinements = num_refinements
        
        # Main embeddings
        self.loc_embed = nn.Embedding(num_locations, embed_dim, padding_idx=0)
        
        # Auxiliary embeddings
        self.user_embed = nn.Embedding(num_users, 24, padding_idx=0)
        self.weekday_embed = nn.Embedding(num_weekdays, 8, padding_idx=0)
        self.time_embed = nn.Embedding(48, 24, padding_idx=0)  # 30-min buckets
        self.diff_embed = nn.Embedding(50, 8, padding_idx=0)
        self.duration_fc = nn.Linear(1, 24)
        
        # Context projection
        aux_dim = 24 + 8 + 24 + 8 + 24
        self.context_fc = nn.Linear(aux_dim, embed_dim)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, 100, embed_dim) * 0.02)
        
        # Recurrent state
        self.recurrent_states = nn.ParameterList([
            nn.Parameter(torch.zeros(1, embed_dim)) for _ in range(num_cycles)
        ])
        
        # Transformer
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=False
            ) for _ in range(num_layers)
        ])
        
        # Gating mechanism for recurrent integration
        self.recurrent_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Refinement
        self.refine_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
        # Classifier
        self.classifier = nn.Linear(embed_dim, num_locations)
        
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
        x = self.loc_embed(loc)
        
        user_emb = self.user_embed(user)
        weekday_emb = self.weekday_embed(weekday)
        time_emb = self.time_embed((start_min // 30).clamp(0, 47))
        diff_emb = self.diff_embed(diff.clamp(0, 49))
        dur_emb = self.duration_fc(duration.unsqueeze(-1))
        
        context = torch.cat([user_emb, weekday_emb, time_emb, diff_emb, dur_emb], -1)
        context = self.context_fc(context)
        
        # Fuse
        x = x + context
        x = x + self.pos_embed[:, :L, :]
        
        # Padding mask
        pad_mask = ~mask
        
        # Recurrent cycles
        state = self.recurrent_states[0].expand(B, -1)
        
        for cycle in range(self.num_cycles):
            # Broadcast state
            state_expanded = state.unsqueeze(1).expand(-1, L, -1)
            
            # Gate integration
            gate = self.recurrent_gate(torch.cat([x, state_expanded], -1))
            x_gated = x + gate * state_expanded
            
            # Transformer
            h = x_gated
            for layer in self.transformer_layers:
                h = layer(h, src_key_padding_mask=pad_mask)
            
            # Update state via attention
            attn_weights = self.attention_pool(h)
            attn_weights = attn_weights.masked_fill(~mask.unsqueeze(-1), 0)
            state = (h * attn_weights).sum(1)
            
            # Next cycle init
            if cycle + 1 < self.num_cycles:
                state = state + self.recurrent_states[cycle + 1].expand(B, -1)
            
            x = h
        
        # Final pooling
        attn_weights = self.attention_pool(x)
        attn_weights = attn_weights.masked_fill(~mask.unsqueeze(-1), 0)
        pooled = (x * attn_weights).sum(1)
        
        # Combine
        final = pooled + state
        
        # Refinement
        for _ in range(self.num_refinements):
            refined = self.refine_fc(final.detach())
            final = final + 0.1 * refined
        
        # Classify
        logits = self.classifier(final)
        
        return logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
