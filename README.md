# FLYP Unofficial Implementation

This repository provides an unofficial implementation of the method described in the paper **"Finetune like you pretrain: Improved finetuning of zero-shot vision models"**. The code is designed for fine-tuning vision models on ImageNet-format classification datasets.

---

## Repository Structure

```
.
├── datacreation/                # Scripts for dataset preparation
│   ├── mean_std.py              # Computes mean and variance of the dataset for regularization
│   └── trans_standordcars.py    # Converts StanfordCars dataset to ImageNet format
├── src/                         # Main source code
│   ├── main_flyp_linearprob.py  # Main script for FLYP implementation
│   └── template/                    # Templates for CLIP text encoder prompts
└── README.md                    # This file
```

---

## Dataset Preparation

### 1. Compute Dataset Mean and Variance
The `mean_std.py` script calculates the mean and variance of your dataset. These values are used for regularization during fine-tuning.

**Usage:**
```bash
python datacreation/mean_std.py --data_path /path/to/dataset
```

### 2. Convert StanfordCars Dataset to ImageNet Format
The `trans_standordcars.py` script converts the StanfordCars dataset into the ImageNet format. If you need to convert other datasets, you can modify this script accordingly.

**Usage:**
```bash
python datacreation/trans_standordcars.py --source_path /path/to/stanfordcars --target_path /path/to/output
```

---

## Fine-Tuning with FLYP

The `main_flyp_linearprob.py` script is the main entry point for implementing the FLYP method.

**Usage:**
```bash
python src/main_flyp_linearprob.py \
    --data_path /path/to/imagenet_format_dataset \
    --mean /path/to/mean_values \
    --std /path/to/variance_values \
    --template_dir /path/to/template_folder
```

**Arguments:**
- `--data_path`: Path to the dataset in ImageNet format.
- `--mean`: Path to the precomputed mean values.
- `--std`: Path to the precomputed variance values.
- `--template_dir`: Path to the folder containing CLIP text encoder prompts.

---

## Templates for CLIP Text Encoder

The `template/` folder contains predefined prompts for the CLIP text encoder. These prompts are used during classification tasks.

---

## Customization

If you want to adapt this code for other datasets:
1. Modify `trans_standordcars.py` to convert your dataset into ImageNet format.
2. Update the prompts in the `template/` folder if necessary.

---

## Requirements

- Python 3.x
- PyTorch
- torchvision
- CLIP (OpenAI)

Install dependencies using:
```bash
pip install torch torchvision clip
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Paper: [Finetune like you pretrain: Improved finetuning of zero-shot vision models](https://arxiv.org/abs/your-paper-link)
- CLIP: [OpenAI CLIP](https://github.com/openai/CLIP)

---

For questions or issues, please open an issue in this repository. Contributions are welcome!
