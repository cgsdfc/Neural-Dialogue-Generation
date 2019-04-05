# Future_Prediction

the future prediction (Soothsayer) models described in [4]

## train_length

to train the Soothsayer Model for Length Prediction

Available options include:

    -dimension          (default 512, vector dimensionality. The value should be the same as that of the pretrained Seq2Seq model.
                        Otherwise, an error will be reported)
    -params_file        (default "../../Atten/save_t_given_s/params", load hyperparameters for a pre-trained generative model)
    -generate_model     (default ../../Atten/save_t_given_s/model1, path for loading the pre-trained generative model)
    -save_model_path    (default "save", path for saving the model)
    -train_file         (default "../../data/t_given_s_train.txt", path for the training set)
    -dev_file           (default "../../data/t_given_s_dev.txt", path for the training set)
    -test_file          (default "../../data/t_given_s_test.txt", path for the training set)
    -alpha              (default 0.0001, learning rate)
    -readSequenceModel  (default true, whether to read a pretrained seq2seq model. this variable has to be set to true when training the model)
    -readFutureModel    (default false, whether to load a pretrained Soothsayer Model. this variable has to be set to false when training the model)
    -FuturePredictorModelFile   (path for load a pretrained Soothsayer Model. does not need it at model training time)

to train the model (a pretrained Seq2Seq model is required)

    th train_length.lua [params]

## train_backward

train the Soothsayer Model to predict the backward probability p(s|t) of the mutual information model

Available options include:

    -dimension              (default 512, vector dimensionality. This value should be the same as that of the pretrained Seq2Seq model.
                            Otherwise, an error will be reported)
    -batch_size             (default 128, batch_size)
    -save_model_path        (default "save")
    -train_file             (default "../../data/t_given_s_train.txt", path for the training set)
    -dev_file               (default "../../data/t_given_s_dev.txt", path for the training set)
    -test_file              (default "../../data/t_given_s_test.txt", path for the training set)
    -alpha                  (default 0.01, learning rate)
    -forward_params_file(default "../../Atten/save_t_given_s/params",input parameter files for a pre-trained Seq2Seqmodel p(t|s))
    -forward_model_file     (default "../../Atten/save_s_given_t/model1", path for loading the pre-trained Seq2Seq model p(t|s))
    -backward_params_file   (default "../../Atten/save_s_given_t/params",input parameter files for a pre-trained backward Seq2Seq model p(s|t))
    -backward_model_file    (default "../../Atten/save_s_given_t/model1" path for loading the pre-trained backward Seq2Seq model p(s|t))
    -readSequenceModel      (default true, whether to read a pretrained seq2seq model. this variable has to be set to true when during the model training period)
    -readFutureModel        (default false, whether to load a pretrained Soothsayer Model. this variable has to be set to false during the model training period)
    -PredictorFile          (path for load a pretrained Soothsayer Model. does not need it at model training time)


to train the model (a pretrained forward Seq2Seq model p(t|s) and a backward model p(s|t) are both required)

    th train.lua [params]


## decode

decoding by combining a pre-trained Seq2Seq model and a Soothsayer future prediction model

Other than the input parameters of the standard decoding model in the folder ``Decode/``, additional options include:

    -Task                   (the future prediction task, taking values of "length" or "backward")
    -target_length          (default 0, forcing the model to generate sequences of a pre-specific length.
            0 if there is no such a constraint. If your task is "length", a value for -target_length is required)
    -FuturePredictorModelFile   (path for loading a pre-trained Soothsayer future prediction model.
            If "Task" takes a value of "length", the value of FuturePredictorModelFile should be a model saved from
            training length prediction model in folder train_length. If "Task" takes a value of "backward",
            the model is a model saved from training the backward probability model in the folder train_backward)
    -PredictorWeight        (default 0, the weight for the Soothsayer model)

To run the decoder with a pre-trained Soothsayer model of length:

    th decode.lua \
        -params_file hyperparameterFile_pretrained_seq2seq \
        -model_file modelFile_pretrained_seq2seq \
        -InputFile yourInputFileToDecode \
        -OutputFile yourOutputFile \
        -FuturePredictorModelFile modelFile_Soothsayer_length \
        -PredictorWeight 1 \
        -Task length \
        -target_length 15

To run the decoder with a pre-trained Soothsayer model of backward probability:

    th decode.lua \
        -params_file hyperparameterFile_pretrained_seq2seq \
        -model_file modelFile_pretrained_seq2seq \
        -InputFile yourInputFileToDecode \
        -OutputFile yourOutputFile \
        -FuturePredictorModelFile modelFile_Soothsayer_backward \
        -PredictorWeight 1 \
        -Task backward

If you want to perform MMI reranking at the end,  ``-MMI_params_file`` and ``-MMI_model_file`` have to be pre-specified.
