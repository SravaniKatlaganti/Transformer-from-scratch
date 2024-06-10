import torch.nn as nn
from .multi_head_attention import MultiHeadAttention
from .position_wise_ffn import PositionWiseFeedForward
from .positional_encoding import PositionalEncoding

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        """
        Initialize the TransformerBlock module.
        """
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionWiseFeedForward(d_model, dff)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass for the transformer block.
        """
        attn_output, _ = self.attention(x, x, x, mask)
        out1 = self.layernorm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + self.dropout2(ffn_output))
        return out2

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, max_positional_encoding, dropout=0.1):
        """
        Initialize the Transformer module.
        """
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(input_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_positional_encoding)
        self.encoder_layers = nn.ModuleList([TransformerBlock(d_model, num_heads, dff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.final_layer = nn.Linear(d_model, target_vocab_size)

    def forward(self, x, mask=None):
        """
        Forward pass for the transformer model.
        """
        seq_len = x.size(1)
        x = self.encoder_embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x, mask)

        logits = self.final_layer(x)
        return logits
