# NER_deliverable2

Named Entity Recognition on the GMB corpus — four models benchmarked: CRF, Structured Perceptron, BiLSTM-CRF, and fine-tuned BERT (`bert-base-cased`).

## Structure
```
main.pdf                  - full report
train_models.ipynb        - trains all 4 models, saves to fitted_models/
reproduce_results.ipynb   - loads fitted models, evaluates, runs TINY TEST
utils/utils.py            - shared feature engineering & eval helpers
data/                     - train/test/tiny_test CSVs
fitted_models/            - saved model checkpoints
environment.yml           - conda environment
```

## Run
```bash
conda env create -f environment.yml
conda activate <env_name>
jupyter notebook train_models.ipynb       # trains & saves models
jupyter notebook reproduce_results.ipynb  # loads & evaluates
```

## Results (test set, non-O tokens)

| Model | Accuracy | F1 Macro |
|---|---|---|
| CRF | 0.415 | 0.318 |
| Structured Perceptron | 0.634 | 0.418 |
| BiLSTM-CRF | 0.488 | 0.370 |
| **BERT-base-cased** | **0.668** | **0.528** |

See `main.pdf` for full analysis.

## Author
Núria Torquet Luna