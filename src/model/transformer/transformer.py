import torch
from torch import nn
import copy

from model.transformer.layers import *
from model.transformer.embeddings import *
from model.transformer.blocks import *


class EncoderDecoder(nn.Module):
    """
A standard Encoder-Decoder architecture. Base for this and many
other models.
"""

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Transformer(EncoderDecoder):
    def __init__(self, src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)

        EncoderDecoder.__init__(
            self,
            encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            src_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            generator = Generator(d_model, tgt_vocab)
            )