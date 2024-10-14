import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
import yaml
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import trange

from adam import AdamW
from model_def import Classifier


def load_config():
    parser = argparse.ArgumentParser(
        prog="AG News",
        description="AG News Headline Classifier",
    )
    parser.add_argument("-c", "--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()
    return yaml.safe_load(args.config.read_text())


if __name__ == "__main__":

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

    config = load_config()
    n_iters = config["training"]["n_iters"]
    batch_size = config["training"]["batch_size"]

    train_val_data = load_dataset("fancyzhx/ag_news", split="train")

    training_size = tf.cast(0.8 * tf.size(train_val_data["text"]), dtype=tf.int32)
    print(training_size)
    exit()
    shuffled_indices = tf.random.shuffle(tf.range(tf.size(train_val_data["labels"])))
    train_idxs = shuffled_indices[:training_size]
    val_idxs = shuffled_indices[(training_size):]

    sentence_transformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    classifier = Classifier(
        num_inputs=config["architecture"]["embedding_dim"],
        num_outputs=config["architecture"]["n_classes"],
        hidden_layers=config["architecture"]["hidden_layers"],
    )
    optimizer = AdamW(
        alpha=config["optimizer"]["alpha"],
        lam=config["optimzier"]["lambda"],
    )

    train_embeddings = tf.cast(
        sentence_transformer.encode(train_val_data[:1024]["text"]).T, dtype=tf.float32
    )
