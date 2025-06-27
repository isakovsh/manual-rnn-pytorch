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

## 📉 Loss Curve

![Training Loss](https://github.com/isakovsh/manual-rnn-pytorch/raw/master/loss.png) 

## 🧠 Example Training Output ( Actually, my RNN isn’t learning anything—he's too busy scrolling Instagram 😂😂 )
Epoch   0 | Loss: 4.4631 | Val Loss: 4.4279
Ggp?tLMAdhlK cQvzcFxp,wx'foqs; alrjGmL3
uIXIHCpp&Lbk
...

Epoch 100 | Loss: 3.9322 | Val Loss: 3.9296
dADwn Ahw;kNghUzpjVAHUKL3eLcNlxnRwkHYeQdt&yoNl  qPeQety,...
...

Epoch 200 | Loss: 3.4338 | Val Loss: 3.4369
YlegAnhdn t:i 
oXi dX ARIfV.We
hT
uW;ns&dcqneeKo ,ctEnhe,...
...

Epoch 300 | Loss: 3.3396 | Val Loss: 3.3493
OQui.sie ! vlww e bagnoh nrfrelc is.Eet;t Ai- eye...
...

Epoch 900 | Loss: 3.3524 | Val Loss: 3.2838
BI JmW,s teWnu
eom
ntoymsd e
ape tawse foahsgarpeltK...
...

