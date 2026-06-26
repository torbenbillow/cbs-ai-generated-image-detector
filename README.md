# AI-Generated Image Detector

This repository contains a machine learning project on detecting AI-generated images. The project compares multiple model families for binary image classification, where the goal is to classify an image as either real or AI-generated.

The work was completed as part of the **Artificial Intelligence and Machine Learning** course at Copenhagen Business School.

## Project Overview

Generative AI systems such as Stable Diffusion, DALL-E, and MidJourney can now create highly realistic images. This creates practical challenges for digital platforms, media organizations, businesses, and users who need to evaluate whether visual content is authentic or synthetic.

This project investigates the following research question:

> How effectively can machine learning models distinguish between real and AI-generated images, and how do different modeling approaches influence detection performance?

The project should be viewed as a research and proof-of-concept comparison of AI-generated image detection methods, not as a production-ready authenticity verification tool.

## Repository Contents

```text
ai-generated-image-detector/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_environment_and_data_check.ipynb
    └── 02_logistic_regression_baseline_model_STANDARDIZED.ipynb
    └── 03_cnn_resnet18_model_finetuned_STANDARDIZED.ipynb
    └── 04_vit_base_patch16_224_model_finetuned_STANDARDIZED.ipynb
    └── 05_vit_all_layer_mold_model_STANDARDIZED.ipynb
│   └── 06_clip_mold_model_frozen_backbone_STANDARDIZED.ipynb
│
├── report/
│   └── AI_Generated_Image_Detector_Report.pdf
│
├── data/
│   └── README.md
│
└── models/
    └── README.md
```

### Included Notebooks

This repository includes two representative notebooks:

- `01_environment_and_data_check.ipynb`  
  Sets up the environment, loads the Hugging Face dataset, inspects labels and class balance, checks image properties, compares preprocessing options, and checks for duplicate leakage across splits.

- `06_clip_mold_model_frozen_backbone_STANDARDIZED.ipynb`  
  Implements the final CLIP-MoLD model using a frozen CLIP vision backbone and all-layer MoLD aggregation for AI-generated image detection.

The full written report documents the broader model comparison across all model pipelines.

## Dataset

The project uses the **MS COCOAI / Defactify Image Dataset**, a benchmark dataset designed for AI-generated image detection.

Each sample includes:

- `id` — unique image identifier
- `image` — image object
- `caption` — original MS COCO caption
- `label_1` — binary label  
  - `0` = real image  
  - `1` = AI-generated image
- `label_2` — source generator label for AI-generated images

The dataset includes real MS COCO images and AI-generated images created from the same captions using multiple image generation systems, including:

- Stable Diffusion 2.1
- Stable Diffusion 3
- SDXL
- DALL-E 3
- MidJourney v6

The project also incorporates an additional real-image dataset into the training split to help mitigate class imbalance.

The dataset is not included directly in this repository. The notebooks load the data from Hugging Face.

## Modeling Approach

The full project compared models of increasing complexity:

| Model | Description |
|---|---|
| Logistic Regression | Traditional baseline using flattened pixel vectors |
| Custom CNN | Convolutional neural network trained from scratch |
| ResNet18 | Fine-tuned pretrained convolutional neural network |
| Vision Transformer | Fine-tuned ViT model |
| All-Layer ViT-MoLD | Vision Transformer with multi-layer aggregation |
| CLIP-MoLD | Frozen CLIP vision backbone with all-layer MoLD aggregation |

The included CLIP-MoLD notebook represents the strongest model pipeline from the project.

## Preprocessing

For neural models, image preprocessing included:

- Converting images to RGB
- Resizing images
- Center-cropping to `224 × 224`
- Applying image augmentation during training
- Converting images to PyTorch tensors
- Normalizing images using model-specific statistics

The CLIP-MoLD pipeline uses CLIP-specific preprocessing because the CLIP vision backbone expects inputs in the format used during pretraining.

## CLIP-MoLD Model

The CLIP-MoLD model uses a frozen pretrained CLIP vision encoder as a feature extractor. Instead of using only the final CLIP layer, the model uses representations from all 24 CLIP vision transformer layers.

Each layer output is projected into a shared representation space. A learned softmax gate then assigns weights across layers, allowing the model to learn which CLIP layers are most useful for detecting AI-generated images.

Only the task-specific projection layers, layer-gating mechanism, and classifier head are trained. The CLIP backbone remains frozen.

## Results Summary

The project found a clear performance progression from simpler baselines to pretrained transformer-based models. CLIP-MoLD achieved the strongest overall results.

| Model | Macro Precision | Macro Recall | Macro F1 | Accuracy |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.530 | 0.521 | 0.521 | 0.772 |
| Custom CNN | 0.761 | 0.805 | 0.779 | 0.867 |
| ResNet18 | 0.867 | 0.944 | 0.899 | 0.938 |
| Vision Transformer | 0.919 | 0.971 | 0.942 | 0.966 |
| All-Layer ViT-MoLD | 0.927 | 0.975 | 0.949 | 0.970 |
| CLIP-MoLD | 0.953 | 0.989 | 0.970 | 0.983 |

The strongest performance came from the transformer-based models, especially CLIP-MoLD. However, the results should be interpreted as a comparison of full pipelines rather than a perfectly controlled architecture experiment, since the models differ in architecture, pretraining, backbone size, preprocessing, and training setup.

## Key Findings

- Traditional logistic regression performed weakest, suggesting that flattened pixel vectors are insufficient for this visual classification problem.
- The custom CNN improved performance by preserving spatial image structure.
- Pretrained CNN and transformer models substantially outperformed models trained from scratch.
- Transformer-based models achieved the strongest results overall.
- CLIP-MoLD delivered the best overall performance, but its performance should be interpreted as the result of the full pipeline, not as proof that MoLD alone explains the improvement.
- Strong benchmark performance does not guarantee robustness to unseen generators or future image-generation models.

## Limitations

Several limitations should be considered:

- The dataset is imbalanced toward AI-generated images.
- The models may learn generator-specific artifacts rather than general indicators of AI-generated imagery.
- The evaluation is based on known generators in the benchmark dataset.
- The model has not been tested against unseen future generators.
- Preprocessing improves consistency but may remove useful image information.
- The model uses fixed binary classification thresholds and does not include an abstention mechanism for uncertain cases.
- The model should not be used as definitive proof that an image is real or AI-generated.

## Ethical Considerations

AI-generated image detection can be useful for research, content analysis, and media literacy, but it also creates risks if used as an automated enforcement tool.

A detector may produce false positives or false negatives. Incorrectly labeling a real image as AI-generated can harm trust and credibility, while failing to detect synthetic content may contribute to misinformation. For that reason, this project should be treated as a decision-support tool, not as an authoritative judge of image authenticity.

## Technologies Used

- Python
- PyTorch
- PyTorch Lightning
- Torchvision
- Hugging Face Datasets
- Hugging Face Transformers
- CLIP vision model
- scikit-learn
- pandas
- NumPy
- Matplotlib
- Google Colab

## How to Run

The notebooks were originally developed in **Google Colab**.

### Option 1: Run in Google Colab

1. Open one of the notebooks in Google Colab.
2. Install the required packages using the setup cells.
3. Authenticate with Hugging Face if required.
4. Run the notebook cells in order.

### Option 2: Run Locally

1. Clone this repository:

```bash
git clone https://github.com/YOUR-USERNAME/ai-generated-image-detector.git
cd ai-generated-image-detector
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Launch Jupyter:

```bash
jupyter notebook
```

5. Open the notebook from the `notebooks/` folder.

## Notes on Reproducibility

The notebooks rely on externally hosted Hugging Face datasets and pretrained models. Exact reproducibility may depend on:

- Hugging Face dataset availability
- package versions
- GPU availability
- local or Colab runtime configuration
- authentication access
- random seed behavior across hardware environments

Large model checkpoints and raw datasets are intentionally excluded from this repository.

## Academic Context

This project was completed as part of the Artificial Intelligence and Machine Learning course at Copenhagen Business School.

The repository is intended to demonstrate applied machine learning workflow, model comparison, data preprocessing, evaluation, and practical reflection on AI-generated image detection.
