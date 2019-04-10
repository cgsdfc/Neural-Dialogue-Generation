# Overview
The data directory is mostly for testing purpose. Most of the files found here are *tiny version* of a larger dataset with some exception, like the dictionary file also used in large dataset. For regression test in software development we can't afford to train the model in realistic data only to catch some bugs.

We will explain the usage of each file in the order of their filenames.


# Usages of files

    data/decoded_dev.txt
    data/decoded_test.txt
    data/decoded_train.txt

These files are used as the *machine-generated* examples in the [Adversarial Evaluation](../docs/Adversarial.md). They are also used as *negative examples* in `Adversarial/Discriminative/train_dis.lua`.

----

    data/movie_25000
    
This is the dictionary file for all datasets derived from *OpenSubtitles*.
It is a plain text file where each line is a distinct word in the vocabulary.
The lineno stands for the ID of that word.

----

    data/random_dev.txt
    data/random_test.txt
    data/random_train.txt
    
Unfortunately, their usage is currently unknown. Guess they are to do with [Adversarial Evaluation](../docs/Adversarial.md).

----

    data/s_given_t_dev.txt
    data/s_given_t_test.txt
    data/s_given_t_train.txt
    
These are the *source given target* dataset. They can be used to train a *backward* model -- a model that predicts *source* sentence (or context, or message) given the *target* sentence (or response, or hypothesis).
Backward models are used in these places:
* [decode with MMI reranking](docs/Decode.md).
* training the [future predictor that predicts backward probability](../docs/Future.md).

By the way, to train a backward model, the only thing you need to change from the setting for a forward model is to use the backward dataset.

----

    data/speaker_addressee_dev.txt
    data/speaker_addressee_test.txt
    data/speaker_addressee_train.txt
    
These are the *specific* dataset for training the [Persona_Addressee models](../docs/Persona.md). Both settings of Persona (speaker and speaker_addressee) use these dataset. Their format is slightly different from the `s_given_t_*.txt` in that, the first number in the utterance represents the ID of a speaker while the rest of them is for the utterance as before.

----

    data/t_given_s_dev.txt
    data/t_given_s_test.txt
    data/t_given_s_train.txt
    
These are the *target given source* dataset and can be used to train a forward model. A forward model is an ordinary Seq2Seq with attention and it is the most
commonplace model in the project. Nearly every module requires a pretrained forward model.

Modules directly using forward models are:
* Any form of decoding, including [Decode](../docs/Decode.md) and [Future Decode](../docs/Future.md).
* [Distill](../docs/Distill.md) will train a pool of forward models with the same params.

Modules based on forward models are (just to name a few):
* [Persona_Addressee models](../docs/Persona.md).
* The generative model (`GenModel`) of [Adversarial Evaluation](../docs/Adversarial.md).
* The `EncoderDistiller` from the [Distill module](../docs/Distill.md).

----

    data/wordmatrix_movie_25000
    
This is the pretrained GloVec word embeddings for the vocabulary of `movie_25000`, in the form of a 2D serialized `torch.Tensor`.
It is used in the `GloveDistiller` of the [Distill module](../docs/Distill.md).
