# Transformer: Attention Is All You Need

The Transformer is an encoder-decoder model architecture that uses the attention mechanism to process and learn from sequential data. This project implements the original transformer model introduced in the "Attention Is All You Need" paper for machine translation. 

<p align="center">
  <img width="80%" alt="Transformer Architecture" src="assets/transformer_architecture.svg" >
</p>


## Data

The transformer model is trained on the Multi30K German-English dataset. The `TranslateData` object automatically downloads the dataset and build vocabularies for source and target data using tokenizers from spaCy. You can set the dataset and vocabulary path in the config file. The `Collator` tokenises the input sentences and applies uniform padding across the batch. 

## Model

The Transformer consists of encoder, decoder, and generator block:

1. **Encoder**: The encoder takes in a sequence of source language data and produces a contextualized representation of the input. 

2. **Decoder**: The decoder takes the embeddings from the encoder and generates a target token representation one at a time.

3. **Generator**: Projects decoder output to vocabulary size and applies softmax to choose target token. 

The Translator model supports the following methods:

**`model.predict(src: Tensor, src_mask: Tensor, tgt: Tensor, tgt_mask: Tensor)`**

Given a batch of masked source and target language data, returns the output of the decoder for each target language input. 

**`model.generate(src: Tensor, src_mask: Tensor, max_len: Integer, start_symbol: Integer)`**

Given a masked source language data, performs greedy decoding to predict the target sequence. 

## Training

Update hyperparameters for training in `config.yaml` file. Label smoothing is implemented using KL-divergence loss, which uses a probability distribution instead of one-hot encoding. The trainer implements the same training regime mentioned in the original paper.

## Running Code

1. Install dependencies from requirements file. Make sure to create a virtual/conda environment before running this command.
```
# create new env transformer_env
conda create -n transformer_env python=3.11

# activate transformer_env
conda activate transformer_env

# install other dependencies
pip install -r requirements.txt
```

2. Run `main.py` which starts the training script.
```
# navigate to the src folder
cd src

# run the main file
python main.py
```

## TODOs

- [ ] implement distributed training
- [ ] support for loggers

## References

[1] [Transformer Paper](https://arxiv.org/abs/1706.03762): Attention Is All You Need

[2] [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/): Annotated version of the paper.

[3] [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/): Visual explanation of transformer model and attention mechanism.


