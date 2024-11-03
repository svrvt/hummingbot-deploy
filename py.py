#!/bin/env python3
# import os
# import math
# import guidance
# from guidance import models, gen, select
from huggingface_hub import hf_hub_download

# current_dir = os.path.dirname(__file__)
# tokenizer_path = os.path.join(current_dir, "tokenizer")

model = None


def load_model():
    global model
    if model is None:
        # ???"EmeraldFundLLama.Q4_K_M.gguf",
        repo_id = "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF"
        # filename = "llama-3.2-1b-instruct-q4_k_m.gguf"
        filename = "README.md"
        model_kwargs = {
            "n_ctx": 8192,
            "n_gpu_layers": -1,
            "echo": False,
            # "verbose": True
        }
        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            # local_dir="/home/ru/.cache/lm-studio/models",
            local_dir = f"/home/ru/.cache/lm-studio/models/{repo_id}",
        )
        model = models.LlamaCpp(downloaded_file, **model_kwargs)

load_model()
