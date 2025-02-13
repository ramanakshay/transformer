from torch import nn
from model.transformer.utils import clones
from model.transformer.layers import SublayerConnection, LayerNorm, MultiHeadedAttention, PositionwiseFeedForward


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, h, d_model, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model)
        self.src_attn = MultiHeadedAttention(h, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # Follow Figure 1 (right) for connections.
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)



class Decoder(nn.Module):
    """
Generic N layer decoder with masking.
"""

    def __init__(self, N, h, d_model, d_ff, dropout):
        super(Decoder, self).__init__()
        self.layers = clones(DecoderLayer(h, d_model, d_ff, dropout), N)
        self.norm = LayerNorm(d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
