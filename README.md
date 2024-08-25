# Fine-Tuning PaliGemma for Visual Question Answering (VQA)

This notebook demonstrates how to fine-tune the PaliGemma model for Visual Question Answering (VQA) using the VQAv2 dataset. PaliGemma is a state-of-the-art multimodal model developed by Google that combines vision and language processing to handle complex tasks involving both text and images.

## Background

### PaliGemma

**PaliGemma** is a large-scale multimodal model developed by Google Research. It integrates vision and language processing capabilities, allowing it to handle tasks that require understanding both text and images. The model leverages advanced techniques to effectively represent and process visual and textual information, making it well-suited for tasks like Visual Question Answering (VQA), where the goal is to answer questions based on the content of an image.

### Fine-Tuning

Fine-tuning is a process in transfer learning where a pre-trained model is adapted to a specific task by training it on a smaller, task-specific dataset. This approach is used to leverage the knowledge the model has gained from large-scale pre-training and adapt it to new, often smaller datasets.

In this notebook, we use several advanced techniques to fine-tune PaliGemma:

1. **Low-Rank Adaptation (LoRA)**: LoRA is a technique used to efficiently adapt large pre-trained models to new tasks. It involves adding low-rank matrices to the model's layers, which allows for parameter-efficient fine-tuning. This means fewer parameters are updated during training, reducing computational requirements and memory usage.

2. **Quantization**: We use quantization to convert model weights to lower precision (4-bit in this case) to speed up training and reduce memory consumption. Quantized models can perform inference more efficiently while maintaining a high level of accuracy.

## Requirements

To run this notebook, ensure you have the following:

- A Google Colab environment with GPU support
- Necessary Python packages: `torch`, `transformers`, `datasets`, `peft`

You can install the required packages using the following commands:

```python
!pip install torch transformers datasets peft
```
## Dataset

We use the **VQAv2** dataset, which is a large-scale dataset designed for Visual Question Answering (VQA) tasks. The dataset includes images, questions, and multiple-choice answers.

- **Training Split:** 10% of the VQAv2 training data
- **Columns Used:** `question` and `multiple_choice_answer`

## Model Overview

The model used for fine-tuning is `google/paligemma-3b-pt-224`, a variant of the PaliGemma model.

### Key Components

- **`PaliGemmaProcessor`**: Handles preprocessing of both text and image inputs.
- **`PaliGemmaForConditionalGeneration`**: The main multimodal model for generating answers based on the input.
- **`LoRA`**: Used for parameter-efficient fine-tuning by adding low-rank adapters.
- **`BitsAndBytesConfig`**: Configures the model for 4-bit quantization to optimize memory and performance.

## Training Configuration

The training process is configured with the following hyperparameters:

- **Number of Epochs:** 2
- **Batch Size:** 2 (per device)
- **Gradient Accumulation Steps:** 8
- **Learning Rate:** 2e-5
- **Weight Decay:** 1e-6
- **Warmup Steps:** 2
- **Optimizer:** AdamW with β2 = 0.999
- **Mixed Precision Training:** BF16 (bfloat16)
- **Checkpointing:** Every 1000 steps, with a maximum of one checkpoint saved

## Usage

Run the cells in this notebook to perform the following tasks:

1. Load and preprocess the VQAv2 dataset.
2. Initialize and configure the PaliGemma model with LoRA and quantization.
3. Define a data collator function for batching the inputs.
4. Set up training arguments and initialize the Trainer.
5. Train the model and save the fine-tuned version.

## Results

The fine-tuned model is pushed to the Hugging Face Model Hub:

- [srimoyee12/new_paligemma_vqa_2](https://huggingface.co/srimoyee12/new_paligemma_vqa_2)

You can use this model for Visual Question Answering tasks directly.
