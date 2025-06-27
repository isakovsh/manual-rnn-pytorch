# 🧠 Manual-RNN-Pytorch

A character-level RNN trained **from scratch** in PyTorch — with **manual forward pass**, **backpropagation through time (BPTT)**, **gradient clipping**, and **custom training loop**. No autograd. No fancy APIs. Just raw neurons and tensors.

## ✨ Highlights

- ✅ Manual forward pass
- ✅ Hand-coded backward pass (BPTT)
- ✅ Gradient clipping for stability
- ✅ Character-level text generation
- ✅ Training/validation loss tracking
- ✅ Sampling with temperature control
- ✅ 100% `torch.tensor` logic (no `nn.Module`!)

## 📊 Loss Curve

![Training Loss](loss_curve.png)

## 🧪 Training Example

```bash
python train.py
