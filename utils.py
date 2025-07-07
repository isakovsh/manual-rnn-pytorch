import os
import torch
import requests
import matplotlib.pyplot as plt


def load_data():
    filename = "tiny_shakespeare.txt"

    if os.path.exists(filename):
        print("File already exists. Loading from disk...")
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    
    print("Downloading tiny_shakespeare.txt...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(response.text)

    return response.text


def save_training_results(losses, smooth_losses, val_losses, output_dir="training_results"):
    
    os.makedirs(output_dir, exist_ok=True)

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Train Loss')
    plt.plot(smooth_losses, label='Smoothed Loss')
    if val_losses:
        val_x = [i for i in range(0, len(losses), 100)][:len(val_losses)]
        plt.plot(val_x, val_losses, label='Val Loss', linestyle='--')

    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(output_dir, "loss_plot.png")
    plt.savefig(save_path)
    print(f"📈 Loss plot saved to: {save_path}")
    plt.close()


def clip_gradients(grads, max_norm):
    total_norm = torch.sqrt(sum((g**2).sum() for g in grads))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for g in grads:
            g.mul_(scale)  # in-place scale

    return total_norm


def normalize_gradients(grads):
    for g in grads:
        g /= (g.norm() + 1e-6)

    return grads