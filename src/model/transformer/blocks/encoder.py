from torch import nn
from model.transformer.utils import clones
from model.transformer.layers import SublayerConnection, LayerNorm, MultiHeadedAttention, PositionwiseFeedForward

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, h, d_model, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """
Core encoder is a stack of N layers
"""

    def __init__(self, N,  h, d_model, d_ff, dropout):
        super(Encoder, self).__init__()
        self.layers = clones(EncoderLayer(h, d_model, d_ff, dropout), N)
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask):
        # Pass the input (and mask) through each layer in turn.
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)







