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
- [Future_Prediction](docs/Future_Prediction.md)
- [Distill](docs/Distill.md)

# Acknowledgments
[Yoon Kim](http://people.fas.harvard.edu/~yoonkim)'s [MT repo](https://github.com/harvardnlp/seq2seq-attn)

LantaoYu's [SeqGAN Repo](https://github.com/LantaoYu/SeqGAN)

# Licence
MIT Licence
