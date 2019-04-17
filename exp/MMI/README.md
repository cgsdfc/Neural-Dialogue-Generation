# Title
A Diversity-Promoting Objective Function for Neural Conversation Models

# Metrics
- BLEU
- Human evaluation
- distinct-1,-2

# Models
## MMI-antiLM

    log p(T|S) - λ log p(T)
    
Modification:

Replace language model term `p(T)` with `U(T)`, adapted with a normalizer factor:


    U(T) = p(T) * g(k)
    
    g(k) = 1 if k <= γ
         = 0 if k > γ
            
`g(k)` decreases as `k` increases.

Final form is:

    log p(T|S) − λ log U(T)
    
Decode form is:

    Score(T) = p(T|S) - λ U(T) + γ Nt     


## MMI-bidi

    (1 - λ) log p(T|S) + λ log p(S|T)

Modification:

First generate a N-best list using forward probability and then rerank the list using backward
probability.

Drawbacks:
1. global suboptimal. Emphasize the forward probability too much.
2. require a rather long N-best list in order to make it diverse enough.

Final reranking score:

    Score(T) = p(T|S) + λ p(S|T) + γ Nt
    
    
Note: the final forms of the two variants both use the length of the target.

# Datasets
## Twitter Conversation Triple Dataset

## OpenSubtitles dataset

Limitation: no specification of which character speaks each subtitle line.
preventing us from inferring speaker turns.

Follow: Vinyals et al. (2015)
 
Adaptation:
1. Each line of subtitle constitutes a full speaker turn.
2. Predict the current turn given the preceding ones. 

## IMSDB
- Purposes: evaluation
- Randomly selected 2 subsets as dev and test set, each has 2k pairs.
- source and target length restricted to the range `[6, 18]`.
- About: IMSDB (http://www.imsdb.com/) is a relatively
small database of around 0.4 million sentences and thus not
suitable for open domain dialogue training.


# Training

> broadly aligned with Sutskever et al. (2014).

- layers: 4
- hidden_units: 1000
- dimension: 1000
- init_weight: 0.08
- fixed_alpha: 0.1 (never halving)
- batch_size: 256
- gradient_clipping: threshold=1
- hardware: Tesla K40.
- backward model is *the same* as the forward model
with *S* and *T* exchanged.

# Decoding

- beam_search: beam_size=200
- max_length: 20

# Logging
- MMI_forward: 

    
    /var/lib/docker/containers/46c2fd721f3b5352be58b531468a6d599019037f40566a272335308a9f6a3869/46c2fd721f3b5352be58b531468a6d599019037f40566a272335308a9f6a3869-json.log

- MMI_backward: 


    /var/lib/docker/containers/9063a25070ba48702df19b7121088b2738bdd2a8cf37fb53351d41912989b97e/9063a25070ba48702df19b7121088b2738bdd2a8cf37fb53351d41912989b97e-json.log
