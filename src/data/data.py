import torch
from torch.utils.data import DataLoader
from torchnlp.datasets import multi30k_dataset
from torchtext.data.functional import to_map_style_dataset
from data.preprocessing import load_tokenizers, load_vocab, tokenize, Collator


class TranslateData:
    def __init__(self, config):
        self.config = config.data

        spacy_de, spacy_en = load_tokenizers()
        self.tokenizer = {'de': spacy_de, 'en': spacy_en}

        vocab_src, vocab_tgt = load_vocab(self.tokenizer, config.vocab_path, config.dataset_path)
        self.vocab = {'de': vocab_src, 'en': vocab_tgt}

    def create_dataloader(self):
        collator = Collator(self.tokenizer, self.vocab, gpu, self.config.max_padding)

        train, val, test = multi30k_dataset(
            directory=self.config.dataset_path,
            train=True,
            dev=True,
            test=True,
            urls=[
                'https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz',
                'https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz',
                'https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz'
            ]
        )

        train_dataloader = DataLoader(
            train,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collator,
        )

        valid_dataloader = DataLoader(
            val,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collator,
        )

        return {'train': train_dataloader, 'valid': valid_dataloader}

