# Decode

Decode given a pre-trained generative model. The pre-trained model doesn't have to be a vanilla Seq2Seq model (for example, it can be a trained model from adversarial learning).

## Available Options

    -beam_size      (default 7, beam size)
    -batch_size     (default 128, decoding batch size)
    -params_file    (default "../Atten/save_t_given_s/params", input parameter files for a pre-trained Seq2Seq model.)
    -model_file     (default "../Atten/save_s_given_t/model1", path for loading a pre-trained Seq2Seq model)
    -setting        (default "BS", the setting for decoding, taking values of "sampling": sampling tokens from the distribution; "BS" standard beam search; "DiverseBS", the diverse beam search described in [5]; "StochasticGreedy", the StochasticGreedy model described in [6])
    -DiverseRate    (default 0. The diverse-decoding rate for penalizing intra-sibling hypotheses in the diverse decoding model described in [5])
    -InputFile      (default "../data/t_given_s_test.txt", the input file for decoding)
    -OutputFile     (default "output.txt", the output file to store the generated responses)
    -max_length     (default 20, the maximum length of a decoded response)
    -min_length     (default 1, the minimum length of a decoded response)
    -NBest          (default false, whether to output a decoded N-best list (if true) or only  output the candidate with the greatest score(if false))
    -gpu_index      (default 1, the index of GPU to use for decoding)
    -allowUNK       (default false, whether to allow to generate UNK tokens)
    -MMI            (default false, whether to perform the mutual information reranking after decoding as in [1])
    -MMI_params_file    (default "../Atten/save_s_given_t/params", the input parameter file for training the backward model p(s|t))
    -MMI_model_file     (default "../Atten/save_s_given_t/model1", path for loading the backward model p(s|t))
    -max_decoded_num    (default 0. the maximum number of instances to decode. decode the entire input set if the value is set to 0.)
    -output_source_target_side_by_side  (default true, output input sources and decoded targets side by side)
    -dictPath       (default ../data/movie_25000, dictionary file)

## Commands
To run the model, run

    th decode.lua [params]

To run the mutual information reranking model in [1],  ```-MMI_params_file``` and ``-MMI_model_file`` need to be pre-specified.
