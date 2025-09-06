# FYP-FinAgent
Final Year Project

## This branch is specially for training models on the Google Colab A100 GPU using JAX to optimise the environment and training code.

# Google Colab Set-up
## Open terminal and clone repo
```bash
git clone -b gpu-training-scripts https://<username>:<access_token>@github.com/KrishSaraf/FYP-FinAgent.git
cd FYP-FinAgent
```

## Install dependencies and prepare GPU
```bash
pip install -r requirements.txt

nvidia-smi -pm 1
nvidia-smi -ac 1215,1410
mkdir -p /content/jax_cache

python colab_setup.py
```

## Start training
```bash
python train_ppo_lstm.py
```

## After training ensure to push all changes to this branch
```bash
git add .
git commit -m "Trained PPO LSTM model"
git push origin gpu-training-scripts
```

---
---
---
# FinAgent

# Installation
```
conda create -n finagent python=3.10
conda activate finagent

#for linux
apt-get update && apt-get install -y libmagic-dev
# for mac
pip install python-magic-bin==0.4.14

conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl
pip install -r requirements.txt
playwright install
# for linux
playwright install-deps
```

# Using GPU acceleration for training
```bash

conda create -n rapids-24.06 -c rapidsai -c conda-forge -c nvidia cudf=24.06 python=3.10 cudatoolkit=11.8

conda activate rapids-24.06
```

# Prepare the environment variables
The suggested way to do it is to create a .env file in the root of the repository (never push this file to GitHub) where variables can be defined.
Please check the examples below.
Sample .env file containing private info that should never be on git/GitHub:
```
OA_OPENAI_KEY = "abc123abc123abc123abc123abc123" # https://platform.openai.com/docs/overview
OA_FMP_KEY = "abc123abc123abc123abc123abc123" # https://site.financialmodelingprep.com/developer/docs
OA_POLYGON_KEY = "abc123abc123abc123abc123abc123" # https://polygon.io/
OA_YAHOOFINANCE_KEY = "abc123abc123abc123abc123abc123" # https://finnhub.io/
HUGGINEFACE_KEY = "abc123abc123abc123abc123abc123" # https://huggingface.co/
RAPIDAPI_KEY = "abc123abc123abc123abc123abc123" # https://rapidapi.com/
ALPHA_VANTAGE_KEY = "abc123abc123abc123abc123abc123" # https://www.alphavantage.co/

INDIAN_API_KEY = "" # https://indianapi.in/indian-stock-market
REDDIT_CLIENT_ID = "" # https://www.reddit.com/prefs/apps create an app here
REDDIT_CLIENT_SECRET = ""
REDDIT_USER_AGENT = ""
TWITTER_API_KEY = "" # https://twitterapi.io/
```

# Prepare the data
```

1. Download the data

python download_data.py
python download_reddit_data.py
python download_twitter_data.py
...

2. Process the data
python clean_data.py
python process_data.py
```

# Run - not done yet
```
python train_ppo.py
...
```

# While the training is running, in a different terminal window navigate to project directory and:
```bash
tensorboard --logdir logs/
```
Open the link generated in your browser to see the model performance.
