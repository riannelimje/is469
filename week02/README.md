# Week 2: Word2Vec and Word Embeddings

## Overview
- **Topic:** Word2Vec and Word Embeddings
- **Purpose:** Train a classifier to compute Word2Vec embedding vectors - done with a shallow (2-layer) neural network


## Word2Vec
- Trains classifier to predict likelihood of a word with respect to its context
- Embeddings captures relations between words - words with similar context placed closer in vector space
- Embeddings are dense, not sparse - low dimensional dense vectors (more memory efficient)
- Dimension of embeddings can be much smaller than vocab size
- note: Word2Vec is a static embedding model 
    - assigns a single, fixed vector representation to each word in the vocab 
    - this vector **does not** change depending on the context
    - eg. "bank" will have same embedding as "river bank" and "bank account"
    - captures semantic relationship but not how word meaning varies with different contexts

There are two main architectures:
- **CBOW (Continuous Bag of Words):** Predicts a target word from its surrounding context words.
    - Best on common words
- **Continuous Skip-gram:** Predicts surrounding context words given a target word.
    - Best on rare words

## Comparison Table

| Feature             | Bag of Words (BoW)      | TF-IDF                | Word2Vec             |
|---------------------|-------------------------|-----------------------|----------------------|
| Representation      | Sparse, counts          | Sparse, weighted counts| Dense, learnt  representations|
| Captures Meaning    | No                      | No                    | Yes                  |
| Memory Usage        | High                    | High                  | Low                  |
| Handles Synonyms    | No                      | No                    | Yes (to some extent) |

## References
- [Gensim Documentation](https://radimrehurek.com/gensim/)