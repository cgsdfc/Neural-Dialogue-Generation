# Metrics
- Perplexity
- BLEU
- Speaker consistency measured by human judges

# Models
- Speaker Model
- Speaker Addressee Model

# Decoding
- beam_size: 200
- max_length: 20
- score function: log p(R|M, v) + λ log p(M |R) + γ|R|
- MMI reranking. plain backward model without speaker info.


## Params
- lambda
- gama

using MERT and optimizing BLEU.

# Datasets

## Twitter Persona Dataset
It is used to train the *Speaker Model*.

- Source: Twitter FireHose

## Twitter Sordoni Dataset
It is used by Sordoni et al., 2015.
Its purpose is to train the previous state-of-art LSTM-based model as a baseline.

## Television Series Transcripts
These are small datasets derived from two popular TV series:
- [Friends](https://en.wikipedia.org/wiki/Friends)
- [The Big Bang Theory](https://en.wikipedia.org/wiki/The_Big_Bang_Theory)

Source is *Internet Movie Script Database (IMSDb)* from http://www.imsdb.com.

NB: neither of these transcripts are found in imsdb

To find transcripts dataset is not hard. Try kaggle. They even have *Star Wars*.

### Statistics
- main characters: 13
- turns: 69,565
- develop and test set: 2000 turns, training set: 65565 turns.

### Training Procedures
Domain adaption strategy.
- pre-train a standard seq2seq model on OSDb using [standard protocols](#training-protocols)
- 10 iters for the above pretraining.
- 5 iters on the transcripts dataset to fine tuning.

# Training Protocols
following the approach of (Sutskever et al., 2014)

- 4 layer LSTM models with 1,000 hidden cells for each layer.
- Batch size is set to 128.
- Learning rate is set to 1.0.
- Parameters are initialized by sampling from the uniform distribution `[−0.1, 0.1]`.
- Gradients are clipped to avoid gradient explosion with a threshold of 5.
- Vocabulary size is limited to 50,000.
- Dropout rate is set to 0.2.
- Epochs: 14.
- Hardware: Tesla K40 GPU.
- Time: 1 month.
