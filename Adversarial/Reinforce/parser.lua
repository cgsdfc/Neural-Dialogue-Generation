require 'logroll'

local logger = logroll.print_logger()

local function parse_args()
    local cmd = torch.CmdLine()

    -- Paths for the weights and params of the GenModel and DisModel.
    cmd:option("-disc_params", "Adversarial/discriminative/save/params",
        "hyperparameters for the pre-trained discriminative model")

    cmd:option("-disc_model", "Adversarial/discriminative/save/iter1",
        "path for loading a pre-trained discriminative model")

    cmd:option("-generate_params", "Atten/save_t_given_s/params",
        "hyperparameters for the pre-trained generative model")

    cmd:option("-generate_model", "Atten/save_t_given_s/model1",
        "path for loading a pre-trained generative model")

    -- Paths for train, dev and test files.
    cmd:option("-trainData", "data/t_given_s_train.txt", "path for the training set")
    cmd:option("-devData", "data/t_given_s_dev.txt", "path for the dev set")
    cmd:option("-testData", "data/t_given_s_test.txt", "path for the test set")

    cmd:option("-saveFolder", "save/", "path for data saving")

    cmd:option("-gpu_index", 2, "")
    cmd:option("-lr", 1, "")
    cmd:option("-dimension", 512, "")
    cmd:option("-batch_size", 64, "")
    cmd:option("-sample", true, "")
    cmd:option("-max_iter", 8, "max number of iteration")
    cmd:option("-saveModel", true, "whether to save the trained model")

    cmd:option("-vanillaReinforce", false,
        "true is using vanilla reinforce model, false doing intermediate-step monte-carlo for reward estimation")

    cmd:option("-MonteCarloExample_N", 5, "number of instances for Monte carlo search")

    cmd:option("-baseline", true, "whether to use baseline or not")
    cmd:option("-baselineType", "critic",
        [[
        how to compute the baseline, taking values of critic or aver. 
        for critic, training another neural model to estimate the reward,
        the role of which is similar to the critic in the actor-critic RL model.
        for aver, just use the average reward for earlier examples as a baseline.
        ]])

    cmd:option("-baseline_lr", 0.0005, "learning rate for updating the critic")
    cmd:option("-logFreq", 2000, "how often to print the log and save the model")
    cmd:option("-Timeslr", 0.5, "increasing the learning rate")
    cmd:option("-gSteps", 1, "how often to update the generative model")
    cmd:option("-dSteps", 5, "how often to update the discriminative model")
    cmd:option("-TeacherForce", true, "whether to run the teacher forcing model")

    local params = cmd:parse(arg)
    print(params)
    return params
end

return parse_args
