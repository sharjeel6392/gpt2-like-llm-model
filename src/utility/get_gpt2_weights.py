import os
import requests
import tensorflow as tf
import json
import numpy as np

# Available GPT-2 model sizes
MODEL_SIZES = ["124M", "355M", "774M", "1558M"]

# Base URL for GPT-2 weights
BASE_URL = "https://openaipublic.blob.core.windows.net/gpt-2/models"

def download_file(url, output_path):
    """Download a file from a URL with streaming."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {output_path}")

def download_gpt2(model_size="124M", out_dir="models"):
    """Download GPT-2 weights and extract them."""
    if model_size not in MODEL_SIZES:
        raise ValueError(f"Invalid model size. Choose from {MODEL_SIZES}")

    os.makedirs(out_dir, exist_ok=True)

    model_dir = os.path.join(out_dir, model_size)
    os.makedirs(model_dir, exist_ok=True)

    # URL of the model files (checkpoint, encoder.json, hparams.json, vocab.bpe)
    files = ["checkpoint", "encoder.json", "hparams.json", "model.ckpt.data-00000-of-00001",
             "model.ckpt.index", "model.ckpt.meta", "vocab.bpe"]

    for filename in files:
        url = f"{BASE_URL}/{model_size}/{filename}"
        out_path = os.path.join(model_dir, filename)
        if not os.path.exists(out_path):
            print(f"Downloading {filename}...")
            download_file(url, out_path)
        else:
            print(f"Already exists: {filename}")

    print(f"✅ GPT-2 {model_size} downloaded to {model_dir}")
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, 'hparams.json'), 'r', encoding='utf-8'))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
    return settings, params

def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params

if __name__ == "__main__":
    # Change "124M" to "355M", "774M", or "1558M" if needed
    settings, params = download_gpt2("124M")
    print(f'Settings: \n{settings}, \nParams: \n{params}')