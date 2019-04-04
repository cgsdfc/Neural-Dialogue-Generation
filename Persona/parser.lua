require 'logroll'

logger = logroll.print_logger()

local function parse_args()
    local cmd = torch.CmdLine()
    cmd:option("-batch_size", 64, "batch size")
    cmd:option("-dimension", 1024, "vector dimensionality")
    cmd:option("-dropout", 0.2, "dropout rate")

    cmd:option("-dictPath", "./data/movie_25000", "dictionary file")
    -- note the default files are different from Atten.
    cmd:option("-train_file", "./data/speaker_addressee_train.txt", "training file")
    cmd:option("-dev_file", "./data/speaker_addressee_dev.txt", "develop file")
    cmd:option("-test_file", "./data/speaker_addressee_test.txt", "test file")

    cmd:option("-init_weight", 0.1, "")
    cmd:option("-alpha", 1, "")
    cmd:option("-start_halve", 6, "")
    cmd:option("-max_length", 100, "")
    cmd:option("-vocab_source", 25010, "")
    cmd:option("-vocab_target", 25010, "")
    cmd:option("-thres", 5, "gradient clipping thres")
    cmd:option("-max_iter", 8, "max number of iteration")
    cmd:option("-source_max_length", 50, "")
    cmd:option("-target_max_length", 50, "")
    cmd:option("-layers", 2, "")

    cmd:option("-saveModel", true, "")
    cmd:option("-saveFolder", "save", "")
    cmd:option("-output_file", "", "")

    cmd:option("-reverse", false, "")
    cmd:option("-reverse_target", false, "")
    cmd:option("-gpu_index", 2, "the index of GPU to use")

    cmd:option("-SpeakerNum", 10000, "number of distinct speakers")
    cmd:option("-AddresseeNum", 10000, "number of distinct addressees")
    cmd:option("-speakerSetting", "speaker",
        [[
        taking values of speaker or speaker_addressee;
        speaker: only model the user who speaks;
        speaker_addressee: modeling both the speaker and the addressee
        ]])

    local params = cmd:parse(arg)
    params.save_prefix = params.saveFolder .. "/model"
    params.save_params_file = params.saveFolder .. "/params"
    params.output_file = params.saveFolder .. "/log"

    if not path.isdir(params.saveFolder) then
        paths.mkdir(params.saveFolder)
    end

    local files_to_check = {
        'train_file',
        'dev_file',
        'test_file',
        'dictPath',
    }

    for _, file in ipairs(files_to_check) do
        if not path.isfile(params[file]) then
            error(string.format('%s %s does not exist', file, params[file]))
        end
    end
    logger.info('all required files exist.')

    print(params)
    return params
end

return parse_args
