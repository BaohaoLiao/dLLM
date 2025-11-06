# dLLM


## Install
```bash
pip install lmdeploy==0.10.2
```

## Evaluation
1. Download eval set
    ```bash
    python data/download_data.py --dataset AIME2024
    ```
2. Eval
    ```bash
    # Edit this file
    bash scripts/eval_model.sh
    ```