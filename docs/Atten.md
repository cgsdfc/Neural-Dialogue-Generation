# Atten
Training a vanilla attention encoder-decoder model.

## Available Options

    -batch_size     (default 128,batch size)
    -dimension      (default 512, vector dimensionality)
    -dropout        (default 0.2, dropout rate)
    -train_file     (default ./data/t_given_s_train.txt)
    -dev_file       (default ./data/t_given_s_dev.txt)
    -test_file      (default ./data/t_given_s_test.txt)
    -init_weight    (default 0.1, random uniform initialized from [-iw, iw])
    -alpha          (default 1, initial learning rate)
    -start_halve    (default 6, when the model starts halving its learning rate)
    -max_length     (default 100, sentences longer than that will be dropped?)
    -vocab_source   (default 25010, source vocabulary size)
    -vocab_target   (default 25010, target vocabulary size)
    -thres          (default 5, threshold for gradient clipping)
    -max_iter       (default 8, max number of iteration)
    -source_max_length  (default 50, max length of source sentences)
    -target_max_length  (default 50, max length of target sentences)
    -layers         (default 2, number of lstm layers)
    -saveFolder     (default "save", the folder to save models and parameters)
    -reverse        (default false, whether to reverse the sources)
    -gpu_index      (default 1, the index of the GPU you want to train your model on)
    -saveModel      (default true, whether to save the trained model)
    -dictPath       (default ./data/movie_25000, dictionary file)

## Dataset Format
training/dev/testing data: each line corresponds a source-target pair (in ``t_given_s*.txt``) or target-source pair (in ``s_given_t*.txt``) separated by ``"|"``.

## Commands
To train the *forward* `p(t|s)` model, run

    th Atten/train_atten.lua \
        -train_file ./data/t_given_s_train.txt \
        -dev_file ./data/t_given_s_dev.txt \
        -test_file ./data/t_given_s_test.txt \
        -saveFolder save_t_given_s

After training, the trained models will be saved in ``save_t_given_s/model*``. input parameters will be stored in ``save_t_given_s/params``

To train the *backward* model `p(s|t)`, run

    th Atten/train_atten.lua \
        -train_file ./data/s_given_t_train.txt \
        -dev_file ./data/s_given_t_dev.txt \
        -test_file ./data/s_given_t_test.txt \
        -saveFolder save_s_given_t

The trained models will be stored in ``save_s_given_t/model*``. input parameters will be stored in ``save_s_given_t/params``


## Tests
    Atten/test_atten.sh
    Atten/test_atten_backward.sh
    Atten/test_atten_forward.sh