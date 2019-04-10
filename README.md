# Neural Dialogue Generation

This project contains the code or part of the code for the dialogue generation part in the following papers:
* [1] J.Li, M.Galley, C.Brockett, J.Gao and B.Dolan. "[A Diversity-Promoting Objective Function for Neural Conversation Models](https://arxiv.org/pdf/1510.03055.pdf)". NAACL2016.
* [2] J.Li, M.Galley, C.Brockett, J.Gao and B.Dolan. "[A persona-based neural conversation model](https://arxiv.org/pdf/1603.06155.pdf)". ACL2016.
* [3] J.Li, W.Monroe, T.Shi, A.Ritter, D.Jurafsky. "[Adversarial Learning for Neural Dialogue Generation](https://arxiv.org/pdf/1701.06547.pdf)" arxiv
* [4] J.Li, W.Monroe, D.Jurafsky. "[Learning to Decode for Future Success](https://arxiv.org/pdf/1701.06549.pdf)" arxiv
* [5] J.Li, W.Monroe, D.Jurafsky. "[A Simple, Fast Diverse Decoding Algorithm for Neural Generation](https://arxiv.org/pdf/1611.08562.pdf)" arxiv
* [6] J.Li, W.Monroe, D.Jurafsky. "[Data Distillation for Controlling Specificity in Dialogue Generation](https://arxiv.org/abs/1702.06703)"

This project is maintained by [Jiwei Li](http://www.stanford.edu/~jiweil/). Feel free to contact jiweil@stanford.edu for any relevant issue. This repo will continue to be updated. Thanks to all the collaborators: [Will Monroe](http://stanford.edu/~wmonroe4/), [Michel Galley](https://www.microsoft.com/en-us/research/people/mgalley/), [Alan Ritter](http://aritter.github.io/), [TianLin Shi](http://www.timshi.xyz/home/index.html), [Jianfeng Gao](http://research.microsoft.com/en-us/um/people/jfgao/), [Chris Brockett](https://www.microsoft.com/en-us/research/people/chrisbkt/), [Bill Dolan](https://www.microsoft.com/en-us/research/people/billdol/) and [Dan Jurafsky](https://web.stanford.edu/~jurafsky/).

# Setup

This code requires Torch7 and the following luarocks packages
* [torch-trepl](https://github.com/torch/trepl.git)
* [cutorch](https://github.com/torch/cutorch)
* [cunn](https://github.com/torch/cunn)
* [nngraph](https://github.com/torch/nngraph)
* [torchx](https://github.com/nicholas-leonard/torchx)
* [tds](https://github.com/torch/tds)
* [logroll](https://github.com/rosejn/logroll.git)

You are recommended to use `docker` to set up your environment. In particular, the code is tested against the [cuda-torch-mega](https://github.com/Kaixhin/docker-torch-mega.git) image. Follow these instructions to get started as soon as possible.

First, ensure the necessary building blocks `nvidia-docker2` is installed and working. Then pull the docker image.

We provide some handy scripts in `scripts/` to start the *torch interactive shell* or `th` and bash. All you need to do is
to adjust the hard-coded path in the `script/start-bash.sh` and `script/start-th.sh`. Now test your environment with

    scripts/start-th.sh
    
If everything works, you shall see the greeting from `th`.
To run the regression tests, fire the script:

    script/test-all.sh
    
*Important:* All the bash scripts and all the hard-coded path assume the current working directory is the project root, namely
`Neural-Dialogue-Generation/`. To run a script, you should sit on `Neural-Dialogue-Generation` and write the relative path to
the script. You should *never* `cd` into the directory holding the script and run it from there!
In short:

    # Correct.
    cd Neural-Dialogue-Generation/
    th Atten/train.lua
    
    # Wrong!
    cd Neural-Dialogue-Generation/Atten/
    th train.lua

For the most of time, you won't notice the problem of paths.

# Download Data
Processed training datasets can be downloaded at [link](http://nlp.stanford.edu/data/OpenSubData.tar) (unpacks to 8.9GB). 
All tokens have been transformed to indexes (dictionary file found at ``data/movie_2500``)

    t_given_s_dialogue_length2_3.txt: dialogue length 2, minimum utterance length 3, sources and targets separated by "|"
    s_given_t_dialogue_length2_3.txt: dialogue length 2, minimum utterance length 3, targets and sources separated by "|"
    t_given_s_dialogue_length2_6.txt: dialogue length 2, minimum utterance length 6, sources and targets separated by "|"
    s_given_t_dialogue_length2_6.txt: dialogue length 2, minimum utterance length 6, targets and sources separated by "|"
    t_given_s_dialogue_length3_6.txt: dialogue length 3, minimum utterance length 6, contexts (consisting of 2 utterances) and targets separated by "|"


# Document
- [Atten](docs/Atten.md)
- [Decode](docs/Decode.md)
- [Persona](docs/Persona.md)
- [Adversarial](docs/Adversarial.md)
- [Future_Prediction](docs/Future.md)
- [Distill](docs/Distill.md)

# Modification

## Enhancement
- Dependency on `fbtorch` is dropped.
- More informative logging.
- Use canonical `torch.class()` rather than alcoholic tables in OOP.
- Regression tests for every module.
- Sensible help strings for every option (well, almost).
- Cleaner source codebase.
    * No overly long line.
    * No semicolons.
    * Proper line breaking.
    * No more *suspicious creation of globals*.
    * Self-explanatory namings. 
    
## Downsides
- Possibly incompatible with the original code. Like slight changes in model saving filenames.
- Source files are renamed quite a lot.

# Acknowledgments
- [Jiwei Li's Original Repo](https://github.com/jiweil/Neural-Dialogue-Generation.git)
- [Yoon Kim](http://people.fas.harvard.edu/~yoonkim)'s [MT repo](https://github.com/harvardnlp/seq2seq-attn)
- LantaoYu's [SeqGAN Repo](https://github.com/LantaoYu/SeqGAN)

# Licence
MIT Licence
