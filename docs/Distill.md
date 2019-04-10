# Distill

This folder contains the code for the data distillation method described in [6].

*New Feature*: The `Pool` API lets you train a pool of Seq2Seq models on the gradually distilled data with one single command. 
It is implemented in pure Lua so *no subprocess* is involved and it is much faster.
It automatically runs the distillation procedures step by step and round after round, managing feeding input
to each component and saving their output in an organized directory structure. 
 
## Command Reference
To train the model pool:

    th Distill/Pool/train.lua \
        -atten_params $ATTEN_PARAMS \
        -decoder_params $DECODER_PARAMS \
        -batch_size $BATCH_SIZE \
        -saveFolder $SAVE_FOLDER \
        -distiller $DISTILLER \
        -freq_threshold $FREQ_THRES \
        -rounds $ROUNDS \
        -save_summary
        
Options include:
    
      -rounds            number of distillation rounds to run [8]
      -train_file        initial train data [data/t_given_s_train.txt]
      -dev_file          initial develop data [data/t_given_s_dev.txt]
      -test_file         initial test data [data/t_given_s_test.txt]
      -dictPath          dictionary file [data/movie_25000]
      -atten_params      params file for training the attention model []
      -decoder_params    params file for the decoder []
      -max_decoded_num   the maximum number of instances to decode [1000000]
      -distiller         distillation method to use. Choose from {Encoder,Glove} [Encoder]
      -save_summary      whether to write a summary file per round [false]
      -batch_size        number of examples to be processed at one operation.Note: 
                            it should *not* be larger than the total number of all examples [64]
      -distill_rate       the proportion of training data to distill in a round [0.08]
      -gpu_index         GPU to allocate memory on [2]
      -saveFolder        directory for saving output data []
      -freq_threshold    frequency threshold controlling what responses are considered generic [100]
      -WordMatrix        [Glove] pretrained Glove embedding file [data/wordmatrix_movie_25000]
      -distill_four_gram [Glove] whether to consider four-gram cooccurrence in Glove distillation [false]
      -encoder_params    [Encoder] params file for the pre-trained generative model []
      -encoder_model     [Encoder] path for loading a pre-trained generative model []

The `atten_params` and similar options take a binary dump of the params of the corresponding model.
To obtain one, you can invoke the training script of the model and just save its params.

For the internal of the distillation algorithm, there is one bash script for each step under the `Distill/test/`.
If you need to debug or inspect the operation of each step, you can use these *step-by-step* scripts as a starting point.
The in-code document of `Distill/Pool/Model.lua` also explains how things work in richer details.

For each round, the *conceptual layout* of the directories is:

     * round_dir
          * data_dir
              * train_file
              * dev_file
              * test_file
          * model_dir
              * params_file
              * model_file
          * tmp_dir
              * decoder_output
              * top_response
          * distill_dir
              * train_file
              * dev_file
              * test_file

Note some dirs might not have correspondence to physical directories.
An actual directory might look like this:
    
    test-pool/1/
    ├── distill
    │   ├── t_given_s_dev.txt
    │   ├── t_given_s_dev.txt_summary.csv
    │   ├── t_given_s_test.txt
    │   ├── t_given_s_test.txt_summary.csv
    │   ├── t_given_s_train.txt
    │   └── t_given_s_train.txt_summary.csv
    ├── model
    │   ├── model1
    │   ├── model2
    │   ├── model3
    │   ├── model4
    │   ├── model5
    │   ├── model6
    │   ├── model7
    │   ├── model8
    │   └── params
    └── tmp
        ├── decode.txt
        └── top_response.txt

The `data_dir` holds input files to train the model. In practice, it is mapped to either the dir holding initial data files
or the `distill_dir` of the previous round.
The `model_dir` holds the params and weights of the trained model, which is also called the `saveFolder` of the model.
The `tmp_dir` holds intermediate results of the algorithm, such the decoded responses and the extracted top responses.
The `distill_dir` holds the data files *after* distillation and optionally an associated summary file.

The summary file is `csv` file with these fields: `Score,Distilled,Example`:
- Score: the similarity score of the response of this example.
- Distilled: true if this example was removed.
- Example: the context and response of this example, separated by a bar. 

          
          
## Steps for data distillation

1. train a Seq2Seq model with attention on the dataset to be distilled:


    th Atten/train.lua \
        -train_file ${TRAIN_FILE} \
        -dev_file ${DEV_FILE} \
        -test_file ${TEST_FILE} \
        -saveFolder ${SAVE_FOLDER} \

For details about the `Atten` API, please refer to [document of Atten](docs/Atten.md).

2. decode a large input set (more than 1 million) using the pre-trained Seq2Seq model:

    
    th Decode/decode.lua \
            -params_file $PARAMS_FILE \
            -model_file $MODEL_FILE \
            -InputFile $INPUT_FILE \
            -OutputFile $OUTPUT_FILE \
            -batch_size ${BATCH_SIZE} \
            -max_decoded_num ${MAX_DECODED_NUM} \
        
For details about the `Decoder` API, please refer to [document of Decode](docs/Decode.md).

3. extract top frequent responses:


    python Distill/Extract/extract_top.py $INPUT_FILE  $OUTPUT_FILE -dictPath $DICT_PATH

`$INPUT_FILE` should be the `OutputFile` of the *decode* step. `$OUTPUT_FILE` is conventionally named after `top_response.txt`.
The format of the output is a line-oriented text file, where each line is the response transferred to ids and
the its frequency, separated by a bar character `|`. For example:

    11 11 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7|124
    
4. compute relevance scores for the entire training set and then distill the training set.
The code provides two different ways to compute the scores: using a pre-trained Seq2Seq model or averaging Glove embeddings.


    th Distill/Encoder/distill.lua \
        -TopResponseFile $TOP_RES_FILE \
        -TrainingData $TRAIN_DATA \
        -saveFolder $SAVE_FOLDER \
        -params_file $PARAMS_FILE \
        -model_file $MODEL_FILE \
        -batch_size $BATCH_SIZE \
        -save_summary

Or
  
    th Distill/Glove/distill.lua \
        -TopResponseFile $TOP_RES_FILE \
        -TrainingData $TRAIN_DATA \
        -saveFolder $SAVE_FOLDER \
        -batch_size $BATCH_SIZE \
        -save_summary \
        -distill_four_gram

`$TOP_RES_FILE` should be the output of the `extract_top` step and `$TRAIN_DATA` should be one of the train, dev and test files.
If `save_summary` is on, a decent summary about this distillation will be saved.
The output of the distillation step will be used as the input to the Seq2Seq model in the next round.

## Distillers

### Glove
Use the pretrained Global Vector (Pennington et al. 2018) embeddings to compute similarity score.

Options include:

    -TrainingData      path for your training data to distill []
    -TopResponseFile   path for your extracted top frequent responses []
    -saveFolder        directory for saving output data []
    -batch_size         [1280]
    -gpu_index          [2]
    -WordMatrix        pretrained Glove embedding file [data/wordmatrix_movie_25000]
    -distill_rate      the proportion of training data to distill in this round [0.08]
    -distill_four_gram whether to remove all training instances that share four-grams with any one of the top frequent responses [false]
    -save_summary      whether to write a summary file [false]

Distill the data:

    th Distill/Glove/distill.lua \
        -TopResponseFile $TOP_RES_FILE \
        -TrainingData $TRAIN_DATA \
        -saveFolder $SAVE_FOLDER \
        -batch_size $BATCH_SIZE \
        -save_summary \
        -distill_four_gram

### Encoder

Use a pretrained Seq2Seq model for data distillation.
Other than input parameters in Glove, the path for a pretrained Seq2Seq model needs to be pre-specified.

Option includes:
    
    -TrainingData    path for your training data to distill []
    -TopResponseFile path for your extracted top frequent responses []
    -saveFolder      directory for saving output data []
    -batch_size       [1280]
    -gpu_index        [1]
    -distill_rate    the proportion of training data to distill in this round [0.08]
    -dictPath        dictionary file [data/movie_25000]
    -params_file     hyperparameters for the pre-trained generative model []
    -model_file       path for loading a pre-trained generative model []
    -save_summary    whether to write a summary file [false]

Distill the data:

    th Distill/Encoder/distill.lua \
        -TopResponseFile $TOP_RES_FILE \
        -TrainingData $TRAIN_DATA \
        -saveFolder $SAVE_FOLDER \
        -params_file $PARAMS_FILE \
        -model_file $MODEL_FILE \
        -batch_size $BATCH_SIZE \
        -save_summary
    