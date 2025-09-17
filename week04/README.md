# Week 4: Fine-Tuning Transformers for Text Classification

## Overview
- **Topic:** Fine-tuning BERT and transformer models for sequence classification
- **Purpose:** Learn how to adapt pre-trained transformer models to specific text classification tasks using the Hugging Face ecosystem

## Fine-Tuning with Yelp Review Dataset
- This lab demonstrates how to fine-tune a BERT-based model for sentiment classification using the Yelp Review Full dataset
- Covers data loading, tokenization, model setup, training, and evaluation
- Explores hyperparameter tuning (learning rate, batch size, weight decay, etc.)

## Key Steps
1. Install and import required libraries (`torch`, `transformers`, `datasets`, `accelerate`, `evaluate`)
2. Load and explore the Yelp Review Full dataset
3. Tokenize and preprocess text data for BERT
4. Create small training and evaluation subsets for faster experimentation
5. Set up a BERT model for sequence classification
6. Define training arguments and metrics (accuracy)
7. Train and evaluate the model using Hugging Face’s `Trainer` API
8. Experiment with different hyperparameters


## Hyperparameters 
Fine-tuning transformer models involves adjusting several key hyperparameters that affect training performance and final accuracy:

- **Learning Rate (`learning_rate`)**: Controls how much the model weights are updated during training. Lower values (e.g., 2e-5) are common for transformers to avoid overshooting minima. Higher values can speed up training but risk instability.
- **Batch Size (`per_device_train_batch_size`, `per_device_eval_batch_size`)**: Number of samples processed before updating model weights. Larger batch sizes can improve training stability but require more memory. Smaller batch sizes may generalize better but can be noisier.
- **Number of Epochs (`num_train_epochs`)**: How many times the model sees the entire training dataset. More epochs can improve accuracy but risk overfitting.
- **Weight Decay (`weight_decay`)**: Regularization technique to prevent overfitting by penalizing large weights. Typical values are 0.01 or 0.1.
- **Warmup Ratio (`warmup_ratio`)**: Fraction of training steps used to gradually increase the learning rate from zero to its initial value. Helps stabilize early training.
- **Gradient Accumulation Steps (`gradient_accumulation_steps`)**: Number of steps to accumulate gradients before updating weights. Useful for simulating larger batch sizes on limited hardware.
- **Adam Optimizer Parameters (`adam_beta1`, `adam_beta2`, `adam_epsilon`)**: Control the behavior of the Adam optimizer, which is commonly used for training transformers.
- **Logging and Evaluation (`logging_steps`, `eval_strategy`)**: Determine how often to log metrics and evaluate the model during training.

Tuning these hyperparameters can significantly impact model performance, training speed, and generalization. Try different values to find the best configuration for your dataset and task.

