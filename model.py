import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RecurrentTransformer(nn.Module):
    """
    Recurrent Transformer: replaces RNN/LSTM with a Transformer that processes
    the full input sequence together with a hidden state across multiple cycles.
    """
    def __init__(self, 
                 num_locations=1200,
                 num_users=50,
                 num_weekdays=7,
                 num_start_min_bins=1440,
                 num_diff_bins=100,
                 embed_dim=128,
                 num_heads=4,
                 num_layers=2,
                 num_cycles=3,
                 num_refinements=16,
                 dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_cycles = num_cycles
        self.num_refinements = num_refinements
        
        # Embeddings
        self.loc_embed = nn.Embedding(num_locations, embed_dim, padding_idx=0)
        self.user_embed = nn.Embedding(num_users, embed_dim // 4, padding_idx=0)
        self.weekday_embed = nn.Embedding(num_weekdays, embed_dim // 8, padding_idx=0)
        self.start_min_embed = nn.Embedding(num_start_min_bins, embed_dim // 4, padding_idx=0)
        self.diff_embed = nn.Embedding(num_diff_bins, embed_dim // 8, padding_idx=0)
        
        # Duration processing
        self.duration_proj = nn.Linear(1, embed_dim // 4)
        
        # Project all features to embed_dim
        feature_dim = embed_dim + (embed_dim // 4) * 3 + (embed_dim // 8) * 2
        self.input_proj = nn.Linear(feature_dim, embed_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, dropout, max_len=100)
        
        # Hidden state initialization
        self.hidden_init = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Hidden state update
        self.hidden_proj = nn.Linear(embed_dim, embed_dim)
        
        # Refinement layers (outer loop)
        self.refinement_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(3)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, num_locations)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
    
    def forward(self, loc, user, weekday, start_min, duration, diff, mask):
        batch_size, seq_len = loc.shape
        
        # Embed all features
        loc_emb = self.loc_embed(loc)
        user_emb = self.user_embed(user)
        weekday_emb = self.weekday_embed(weekday)
        start_min_emb = self.start_min_embed(start_min)
        dur_emb = self.duration_proj(duration.unsqueeze(-1))
        diff_emb = self.diff_embed(diff)
        
        # Concatenate all embeddings
        x = torch.cat([loc_emb, user_emb, weekday_emb, start_min_emb, dur_emb, diff_emb], dim=-1)
        
        # Project to embed_dim
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Initialize hidden state
        hidden = self.hidden_init.expand(batch_size, 1, -1)
        
        # Recurrent Transformer cycles
        for cycle in range(self.num_cycles):
            # Concatenate input sequence with hidden state
            combined = torch.cat([hidden, x], dim=1)
            
            # Create extended mask
            hidden_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=mask.device)
            extended_mask = torch.cat([hidden_mask, mask], dim=1)
            
            # Create attention mask (True = not allowed to attend)
            attn_mask = ~extended_mask
            
            # Apply Transformer
            transformed = self.transformer(combined, src_key_padding_mask=attn_mask)
            
            # Extract updated hidden state
            hidden = transformed[:, 0:1, :]
            hidden = self.hidden_proj(hidden)
        
        # Extract final representation
        output = hidden.squeeze(1)
        
        # Outer refinement loop (detached manner)
        for refine_step in range(self.num_refinements):
            output_detached = output.detach()
            for layer in self.refinement_layers:
                output_detached = output_detached + layer(output_detached)
            output = output + 0.1 * (output_detached - output.detach())
        
        # Final prediction
        logits = self.output_proj(output)
        
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
