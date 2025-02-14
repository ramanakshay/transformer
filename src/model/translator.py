import torch

from model.transformer import Transformer

def subsequent_mask(size):
    # Mask out subsequent positions. Boolean mask
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

class Translator(object):
    def __init__(self, src_vocab, tgt_vocab, config):
        self.config = config.model
        self.transformer = Transformer(
            src_vocab = src_vocab,
            tgt_vocab = tgt_vocab,
            N = config.N,
            d_model = config.d_model,
            d_ff = config.d_ff,
            h = config.h,
            dropout = config.dropout)

    def train(self):
        self.transformer.train()

    def eval(self):
        self.transformer.eval()

    def predict(self, src, tgt, src_mask, tgt_mask):
        return self.transformer(src, tgt, src_mask, tgt_mask)

    def greedy_decode(self, src, src_mask, max_len, start_symbol):
        memory = self.transformer.encode(src, src_mask)
        ys = torch.zeros(1,1).fill_(start_symbol).type_as(src.data)
        for i in range(max_len - 1):
            out = self.transformer.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
            prob = self.transformer.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            ys = torch.cat([ys, torch.zeros(1,1).type_as(src.data).fill_(next_word)], dim=1)
        return ys

    def test(self):
        self.eval()
        src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        src_mask = torch.ones(1, 1, 10)

        memory = self.transformer.encode(src, src_mask)
        ys = torch.zeros(1, 1).type_as(src)

        for i in range(9):
            out = self.transformer.decode(
                memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
            )
            prob = self.transformer.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            ys = torch.cat(
                [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
            )

        print("Example Untrained Model Prediction:", ys)