# Distill

This folder contains the code for the data distillation method described in [6].

To run the model:

    sh Distill/pipeline.sh

Steps for data distillation:

* First, decode a large input set (more than 1 million) using a pre-trained Seq2Seq model:


    th Decode/decode.lua \
        -params_file <hyperparameterFile_pretrained_seq2seq> \
        -model_file <modelFile_pretrained_seq2seq> \
        -batch_size 640 \
        -InputFile <yourTrainingData> \
        -OutputFile <yourDecodingOutputFile> \
        -batch_size 128 \
        -max_decoded_num 1000000

* Second, extract top frequent responses:


    sh Distill/extract_top/select_top_decoded.sh <yourDecodingOutputFile> <yourFileToStoreTopResponses>

* Third, compute relevance scores for the entire training set and then distill the training set.
The code provides two different ways to compute the scores: using a pre-trained Seq2Seq model or averaging Glove embeddings.

## Glove

Options include:

    -TrainingData       (path for your training data to distill)
    -TopResponseFile    (path for your extracted top frequent responses)
    -batch_size         (default 1280, batch size)
    -distill_rate       (default 0.08, the proportion of training data to distill in this round)
    -distill_four_gram  (default true, whether to remove all training instances that share four-grams
                         with any one of the top frequent responses)
    -load_score          (default false, whether to load already-computed relevance scores)
    -save_score         (default false, whether to save relevance scores)
    -save_score_file    (default "relevance_score", path for saving relevance_score for each instance in the training set)

Compute relevance scores:

    th Distill/Glove/distill.lua \
        -TopResponseFile <yourFileToStoreTopResponses> \
        -TrainingData <yourTrainingData> \
        -OutputFile <FileForRemainingData> \
        -save_score \
        -save_score_file <relevance_score>

Distill the Data:

    th Distill/Glove/distill.lua \
        -TopResponseFile <yourFileToStoreTopResponses> \
        -TrainingData <yourTrainingData> \
        -OutputFile <FileForRemainingData> \
        -total_lines <number of lines in your TrainingData> \
        -load_score \
        -save_score_file <relevance_score>

The remaining data after this round of data distillation will be stored in ``FileForRemainingData``,
on which a new Seq2Seq model will be trained.

## Encoder

Use a pre-trained Seq2Seq model for data distillation.
Other than input parameters in Glove, the path for a pre-trained Seq2Seq model needs to be pre-specified:

    -params_file    (default Atten/save_t_given_s/params, hyperparameters for the pre-trained generative model)
    -model_file     (default Atten/save_t_given_s/model1, path for loading a pre-trained generative model)

To run the model:

    th Distill/Encoder/distill.lua \
        -TopResponseFile <yourFileToStoreTopResponses> \
        -TrainingData <yourTrainingData> \
        -OutputFile <FileForRemainingData> \
        -params_file <Seq2SeqParamsFile> \
        -model_file <Seq2SeqModelFile> \
        -batch_size 6400
