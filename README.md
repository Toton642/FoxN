## 1. Header

# Neural FOXP2: Language Specific Neuron Steering for Targeted Language Improvement in LLMs

Official implementation of **Neural FOXP2**, to appear at **ACL 2026**.

> Anusa Saha, Tanmay Joshi, Vinija Jain, Aman Chadha, Amitava Das  
> Paper: [arXiv:2602.00945](https://arxiv.org/abs/2602.00945)

Neural FOXP2 identifies a sparse set of language selective features and steers them so that a chosen target language becomes the default output of a multilingual LLM, without finetuning the backbone.

## 2. Abstract

Despite multilingual pretraining, large language models often default to English under weak prompts, while other languages remain harder to elicit. Neural FOXP2 treats this as a mechanistic control problem: it isolates a sparse set of language neurons in a sparse autoencoder basis and discovers low rank directions that govern language choice. By steering only these directions at inference time, Neural FOXP2 makes Hindi or Spanish the default output language of LLaMA 3 8B across machine translation, question answering, natural language inference, and summarisation tasks, while preserving task performance.

## 3. Method Overview

We model language defaultness as the early step probability mass that a multilingual LLM assigns to a target language versus English under neutral prompts. Neural FOXP2 implements a three stage pipeline to localise and steer the underlying control mechanism:

1. **Stage I: Localise language neurons**  
   Train sparse autoencoders on residual stream activations, compute language selectivity and causal lift scores for English versus a target language, and identify a compact set of language specific features.

2. **Stage II: Low rank steering directions**  
   Using matched meaning prompts in English and the target language, build activation difference matrices in the language neuron coordinates, perform SVD to obtain a low dimensional steering subspace, and select an intervention window of contiguous layers.

3. **Stage III: Signed sparse intervention**  
   Within the selected layers, construct prototype directions for the target language and English, tune intervention strengths on development data, and apply signed sparse activation edits during generation to shift defaultness toward the target language while keeping task metrics stable.


## 4. Installation

We recommend a fresh virtual environment with the versions below.

- Python 3.8 or later
- PyTorch 2.0 or later
- NVIDIA GPU with 24 GB or more of VRAM (for Llama-3.1-8B)
- Hugging Face account with access to Llama weights

```bash
# Clone the repository
git clone https://github.com/PhotonTJ/NeuralFoxP2.git
cd NeuralFoxP2

# (Optional) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```


## 5. Usage and CLI options

### 5.1. Single Layer experiment

```bash
python main.py --hf-token YOUR_TOKEN --layers 18 --epochs 50
```
### 5.2. Full Pipeline 

```bash
python main.py --hf-token YOUR_TOKEN --layers 8-23 --n-prompts 2500
```

### 5.3. Specific Stage experiment

```bash
# Run only Stage I
python main.py --stage 1 --hf-token YOUR_TOKEN

# Resume from Stage I checkpoint
python main.py --stage 2 --resume-from ./outputs/stage1_checkpoint.pkl
```

### 5.4. Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--hf-token` | "" | HuggingFace API token |
| `--layers` | "8-23" | Layer range (e.g., "8-23" or "18") |
| `--n-prompts` | 2500 | Number of prompts used to collect activations |
| `--epochs` | 150 | SAE training epochs |
| `--output-dir` | "./outputs" | Results directory |
| `--resume-from` | None | Checkpoint path to resume from |
| `--stage` | "all" | Stage to run: "all" runs Stage I, II, III; "1", "2", "3" run a single stage |

### 5.5.Example Commands


#### Hindi steering on all tasks (LLaMA 3 8B)

```bash
python main.py \
  --hf-token YOUR_TOKEN \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --target-lang hi \
  --layers 8-23 \
  --n-prompts 2500 \
  --stage all \
  --output-dir ./outputs/llama3_8b_hi_all
```
#### Spanish steering on all tasks (LLaMA 3 8B)
```bash
python main.py \
  --hf-token YOUR_TOKEN \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --target-lang es \
  --layers 8-23 \
  --n-prompts 2500 \
  --stage all \
  --output-dir ./outputs/llama3_8b_es_all
```

### 5.6. Language defaultness on neutral prompts
For measuring early step defaultness on a neutral prompt set without running all downstream tasks (Example Target Language as Hindi)

```bash
python main.py \
  --hf-token YOUR_TOKEN \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --target-lang hi \
  --layers 8-23 \
  --n-prompts 2500 \
  --stage 3 \
  --eval-neutral-only \
  --output-dir ./outputs/llama3_8b_hi_neutral
```
## 6. Additional Sections
### Data and tasks

Neural FOXP2 is evaluated on LLaMA 3 8B in Hindi and Spanish across four task families: machine translation, question answering, natural language inference, and summarisation. In each case we pair a target language dataset with an English reference so that we can check both task performance and whether the model now starts in the target language under weak prompts.

All datasets are loaded through the Hugging Face `datasets` library or standard public releases. The code expects them to be downloadable at runtime and caches processed versions under a configurable data root (by default `./data`). If any task requires a manual download or license acceptance, we note that in the corresponding configuration file under `config.py` and in the script level help.

We also build a neutral prompt set for each language and a collection of matched meaning pairs in English and the target language. These sets are used to estimate activation differences for steering and to evaluate language defaultness without mixing in task loss.

### Measuring language defaultness

A central object in the paper is the model’s default language under weak or neutral prompts. In the code we operationalise this by looking at the first few decoding steps and measuring how much next token probability mass goes to target language tokens compared to English tokens.

Concretely, for each decoding step we sum the probabilities over a target vocabulary subset and over an English subset, then take the difference as a defaultness score for that step. We average this score across a small time horizon to get a single defaultness value per prompt, and later aggregate over datasets and languages. This metric is implemented and evaluated in `stage3.py`. When you run the full pipeline with `--stage all`, the defaultness metrics are computed alongside task accuracy and written to the outputs directory.

In addition to this mechanistic proxy, the paper also calibrates defaultness against human judgments of whether the model “naturally” starts in Hindi or Spanish. The README does not try to replicate the full human study, but the logs produced by the code contain the quantities that are used for that analysis.

### Reproducing results from the paper

The core scripts in this repository are enough to rerun the main experiments from the paper: Hindi and Spanish steering on LLaMA 3 8B across machine translation, question answering, natural language inference, and summarisation. The commands in the “Usage” section show how to run the full pipeline and write metrics and checkpoints to `./outputs`.


## 7. Project Structure

```
NeuralFoxP2/
├── config.py       # Configuration and hyperparameters
├── models.py       # SAE class and model loading
├── data.py         # Data loading and preprocessing
├── utils.py        # Helper functions
├── stage1.py       # Stage I: sparse autoencoder training and language neuron scoring
├── stage2.py       # Stage II: construction of language shift matrices and SVD directions
├── stage3.py       # Stage III: steering rule and evaluation hooks
├── main.py         # Main pipeline runner
├── README.md       # This file
└── requirements.txt
```

## 8. Extending Neural FOXP2

To apply Neural FOXP2 to a new model, update `config.py` with the new model name, layer range, and tokeniser, then rerun Stage I to Stage III. To add a new target language, define token subsets for the language and English, build matched meaning prompts, and follow the same pipeline. We provide example configs for Hindi and Spanish in `configs/` as starting points.

### 8.1 Configuration

Edit `config.py` to modify hyperparameters:

```python
@dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    layer_range: List[int] = field(default_factory=lambda: list(range(8, 24)))
    n_features: int = 512        # SAE feature dimensions
    epochs: int = 150            # SAE training epochs
    lr: float = 5e-4             # Learning rate
    lambda_sparse: float = 5e-3  # Sparsity penalty
    # ... more options
```


## 9. Output

The pipeline saves checkpoints after each stage:
- `outputs/stage1_checkpoint.pkl` - SAEs and language neurons
- `outputs/stage2_checkpoint.pkl` - Steering directions
- `outputs/final_checkpoint.pkl` - Complete results

## 10. Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{saha2026neuralfoxp2,
  title     = {Neural FOXP2: Language Specific Neuron Steering for Targeted Language Improvement in LLMs},
  author    = {Saha, Anusa and Joshi, Tanmay and Jain, Vinija and Chadha, Aman and Das, Amitava},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics},
  year      = {2026}
}
```

## License

MIT License
