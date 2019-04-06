# Adversarial

The adversarial-reinforcement learning model and the adversarial-evaluation model described in [3].

## Discriminative

Adversarial-evaluation: Train a binary evaluator (a hierarchical neural network) to label dialogues as 
machine-generated (negative) or human-generated (positive).

Available options are:

    -batch_size     (default 128, batch size)
    -dimension      (default 512, vector dimensionality)
    -dropout        (default 0.2, dropout rate)
    -pos_train_file     (default "data/t_given_s_train.txt", human generated training examples)
    -neg_train_file     (default "data/decoded_train.txt", machine generated training examples)
    -pos_dev_file       (default "data/t_given_s_dev.txt", human generated dev examples)
    -neg_dev_file       (default "data/decoded_dev.txt", machine generated dev examples)
    -pos_test_file      (default "data/t_given_s_test.txt", human generated test examples)
    -neg_test_file      (default "data/decoded_test.txt", machine generated test examples)
    -source_max_length      (default 50, maximum sequence length)
    -dialogue_length        (default 2, the number of turns for a dialogue. the model supports multi-turn dialgoue classification)
    -saveFolder             (default save/, the folder to save models and parameters)
    -saveModel              (default true, whether to save the model)

To the train the discriminator, run

    th Adversarial/Discriminative/train_dis.lua [params]
    
## Reinforce

Train the adversarial-reinforcement learning model in [3].

Available options include:

    -disc_params        (default "Discriminative/save/params", hyperparameters for the pre-trained Discriminative model)
    -disc_model         (default "Discriminative/save/iter1", path for loading a pre-trained Discriminative model)
    -generate_params    (default "Atten/save_t_given_s/params", hyperparameters for the pre-trained generative model)
    -generate_model     (default Atten/save_t_given_s/model1, path for loading a pre-trained generative model)
    -trainData      (default "data/t_given_s_train.txt", path for the training set)
    -devData        (default "data/t_given_s_train.txt", path for the dev set)
    -testData       (default "data/t_given_s_train.txt", path for the test set)
    -saveFolder     (default "save", path for data saving)
    -vanillaReinforce       (default false, whether to use vanilla Reinforce or Monte Carlo)
    -MonteCarloExample_N    (default 5, number of tries for Monte Carlo search to approximnate the expectation)
    -baseline       (default true, whether to use baseline or not)
    -baselineType   (default "critic", taking value of either "aver" or "critic". If set to "critic", another neural model is
                    trained to estimate the reward, the role of which is similar to the critic in the actor-critic RL model;
                    If set to "aver", just use the average reward for earlier examples as a baseline")
    -baseline_lr    (default 0.0005, learning rate for updating the critic)
    -logFreq        (default 2000, how often to print the log and save the model)
    -Timeslr        (default 0.5, increasing the learning rate)
    -gSteps         (default 1, how often to update the generative model)
    -dSteps         (default 5, how often to update the discriminative model)
    -TeacherForce   (default true, whether to run the teacher forcing model)

To run the adversarial-reinforcement learning model, a pretrained generative model and a pretrained discriminative model are needed.
Trained models will be saved and can be later re-loaded for decoding using different decoding strategies in the folder ``Decode/``.

To train the model, run

    th Adversarial/Reinforce/train.lua [params]

## Note 

if you encounter the following error ``"bad argument # 2 to '?' (out of range) in function model_backward"`` after training the model
for tens of hours, this means the model has exploded (see the teacher forcing part in Section 3.2 of the paper).
The reason why the error appears as ``"bad argument #2 to '?'"`` is because of the sampling algorithm in Torch.
If you encounter this issue, shrink the value of the variable ``-Timeslr``.
