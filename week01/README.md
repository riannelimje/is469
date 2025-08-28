# Week 1

## Key Terms

**Corpus**: A collection of text documents used for analysis or modeling. In NLP, a corpus is the dataset of texts that you want to process or analyze.

**Vocab (Vocabulary)**: The set of unique words (tokens) found in the corpus. The vocabulary is used to build features such as in the Bag of Words model, where each unique word becomes a feature.


## Bag of Words (BoW) vs TF-IDF

**Bag of Words (BoW):**
- Represents text as a vector of word counts (frequency of each word in the document).
- Ignores grammar and word order, focusing only on the presence and frequency of words.
- Simple and effective for many tasks, but treats all words as equally important.
- BOW length == vocab length

**TF-IDF (Term Frequency-Inverse Document Frequency):**
- Represents text by weighting word counts by how unique or important a word is across all documents.
- Words that appear frequently in a document but rarely across the corpus get higher scores.
- Helps reduce the impact of common words (like "the", "and") and highlight more meaningful terms.
- Often leads to better performance in information retrieval and text classification tasks.

## Notes on Cosine Similarity

**Cosine similarity** is commonly used to measure the similarity between two text vectors (e.g., BoW or TF-IDF vectors). Here are some important points to take note of:

- **Ignores Magnitude:** Cosine similarity measures the angle between vectors, not their length. It focuses on the direction (pattern of word usage), not the absolute word counts.
- **Normalization:** Input vectors should be non-zero. Zero vectors (e.g., documents with no shared vocabulary) will result in undefined similarity.
- **Sensitive to Sparse Data:** In high-dimensional, sparse text data, many vectors may be nearly orthogonal (low similarity), even for related documents.
- **Does Not Capture Semantics:** Cosine similarity only considers the presence and frequency of words, not their meaning or context.
- **Preprocessing Matters:** Results can be affected by tokenization, stopword removal, and other preprocessing steps.

Use cosine similarity for quick, effective comparison of text vectors, but be aware of its limitations in capturing deeper semantic relationships.

## Scikit-learn 
Does not account for single letter words like "i" in the counts - explains the discrepancy in length

## References
- [Scikit-learn: Bag of Words](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
