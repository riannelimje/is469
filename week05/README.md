# Week 5: Fine-Tuning GPT-OSS 20B Model with Unsloth

## Overview
- **Topic:** Fine-tuning large language models (GPT-OSS 20B) using Unsloth for efficient training
- **Purpose:** Learn how to fine-tune OpenAI's GPT-OSS model with reasoning capabilities using parameter-efficient techniques like LoRA

## GPT-OSS Model and Reasoning Effort
- GPT-OSS is OpenAI's open-source model that supports adjustable reasoning effort levels
- Three reasoning levels available: **Low**, **Medium**, and **High**
- Higher reasoning effort provides better performance but increases latency and token usage
- Uses OpenAI's Harmony format for conversation structures, reasoning output, and tool calling

## Key Concepts

### Unsloth Framework
- Unsloth provides memory-efficient training with up to 30% less VRAM usage
- Supports 4-bit quantization for faster downloading and reduced memory usage
- Enables LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning

### LoRA (Low-Rank Adaptation)
- Allows training only 1% of model parameters while maintaining performance
- Freeze base weights and only train adapters (A,B)
- Key parameters:
  - `r`: Rank of adaptation (8, 16, 32, 64, 128)
  - `lora_alpha`: Scaling parameter (typically 16)
  - `target_modules`: Which model layers to adapt

### Training on Completions Only
- Uses `train_on_responses_only` method to focus training on assistant outputs
- Masks user inputs during training to improve accuracy and reduce loss
- Helps the model learn to generate better responses

## Key Steps
1. **Installation**: Install Unsloth, PyTorch, Transformers, and dependencies
2. **Model Loading**: Load GPT-OSS 20B with 4-bit quantization for memory efficiency
3. **LoRA Setup**: Add LoRA adapters for parameter-efficient fine-tuning
4. **Reasoning Effort Testing**: Experiment with low, medium, and high reasoning levels
5. **Data Preparation**: Use HuggingFaceH4/Multilingual-Thinking dataset for multilingual reasoning
6. **Training Configuration**: Set up SFTTrainer with optimized hyperparameters
7. **Training**: Fine-tune the model with masked completion training
8. **Inference**: Test the fine-tuned model with different reasoning efforts
9. **Saving**: Export model as LoRA adapters or merged weights

## Dataset: Multilingual-Thinking
- Contains reasoning chain-of-thought examples in multiple languages
- Derived from user questions translated from English to four other languages
- Same dataset used in OpenAI's official fine-tuning cookbook
- Helps model learn reasoning capabilities across different languages

## Training Configuration
Key hyperparameters used in the notebook:
- **Batch Size**: 1 (with gradient accumulation of 4)
- **Learning Rate**: 2e-4 (higher than typical BERT fine-tuning)
- **Optimizer**: adamw_8bit for memory efficiency
- **Max Steps**: 5 (for quick testing) or full epochs for complete training
- **Weight Decay**: 0.01 for regularization
- **Warmup Steps**: 5 for learning rate scheduling

## Reasoning Effort Levels
- **Low**: Fast responses, minimal reasoning steps
- **Medium**: Balanced performance and speed
- **High**: Maximum reasoning capability, higher latency

## Usage Notes
For quick testing with the notebook:
1. Change the model (quantised version): `model_name = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"`
2. Increase token generation: `max_new_tokens = 256`
3. Reduce training steps for testing: `max_steps = 5`

## References
- [Original Colab Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-Fine-tuning.ipynb)
- [Unsloth Documentation](https://docs.unsloth.ai/)
- [OpenAI GPT-OSS Fine-tuning Cookbook](https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers)
- [HuggingFaceH4/Multilingual-Thinking Dataset](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking)