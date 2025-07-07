import torch
import torch.nn.functional as F
from utils import load_data, save_training_results, clip_gradients, normalize_gradients
import torch.optim as optim

data = load_data()

chars = sorted(list(set(data)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Convert the entire text into a list of indices
data = torch.tensor([char_to_idx[c] for c in data], dtype=torch.long) 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Split data
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# 2. Manual batch generator
def get_batch(split,batch_size=32, block_size=32):
    data_src = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_src) - block_size, (batch_size,))

    x = torch.stack([data_src[i:i+block_size] for i in ix])
    y = torch.stack([data_src[i+1:i+1+block_size] for i in ix])

    return x.to(device), y.to(device)

vocab_size = len(chars)
embedding_size = 32
hidden_size = 128


# Weight matrices
W_emb = torch.empty((vocab_size, embedding_size), device=device, requires_grad=True)
torch.nn.init.xavier_uniform_(W_emb)

W_xh = torch.empty((embedding_size, hidden_size), device=device, requires_grad=True)
torch.nn.init.xavier_uniform_(W_xh)

W_hh = torch.empty((hidden_size, hidden_size), device=device, requires_grad=True)
torch.nn.init.xavier_uniform_(W_hh)

W_hy = torch.empty((hidden_size, vocab_size), device=device, requires_grad=True)
torch.nn.init.xavier_uniform_(W_hy)

# Biases
b_h = torch.zeros((hidden_size,), device=device, requires_grad=True)
b_y = torch.zeros((vocab_size,), device=device, requires_grad=True)


def rnn_forward(xb):

  cache = {
    "x_t": [],     # all x_t
    "h_t": [],     # all h_t
    "h_prev": [],  # all h_{t-1}
  }

  B, T = xb.shape
  h = torch.zeros(B, hidden_size, device=device)
  logits = []

  for t in range(T):
        x_t = W_emb[xb[:, t]]                        # shape: [B, E]
        h_pre = x_t @ W_xh + h @ W_hh + b_h          # [B, H]
        h = torch.tanh(h_pre)                        # [B, H]
        y_t = h @ W_hy + b_y                         # [B, vocab_size]
        logits.append(y_t)

        cache["x_t"].append(x_t)
        cache["h_t"].append(h)
        cache["h_prev"].append(h_pre)

  return torch.stack(logits, dim=1) , cache


def calculate_loss_backward(logits,y_true,cache,xb):
  """This function calculates loss and backward"""

  # ----------------------------------------------------------------
  # loss calculation
  B, T, V = logits.shape

  logits = logits.view(B * T, V)
  y_true = y_true.view(B * T)

  logits = logits - logits.max(dim=1, keepdim=True).values
  probs = torch.exp(logits)
  probs = probs / probs.sum(dim=1,keepdim=True)
  loss = -torch.log(probs[range(B * T), y_true]).mean()
  # -----------------------------------------------------------------

  # -----------------------------------------------------------------

  dWxh = torch.zeros_like(W_xh)
  dWhh = torch.zeros_like(W_hh)
  dWhy = torch.zeros_like(W_hy)
  db_h  = torch.zeros_like(b_h)
  db_y  = torch.zeros_like(b_y)
  dW_emb = torch.zeros_like(W_emb)

  # backward
  dlogits = probs.clone()
  dlogits[range(B * T), y_true] -= 1
  dlogits /= B * T
  dlogits = dlogits.view(B, T, V)
  dh_next = torch.zeros_like(cache['h_t'][0])

  x_t_list = cache['x_t']
  h_t_list = cache['h_t']
  h_prev_list = cache['h_prev']

  for t in reversed(range(T)):
      x_t = x_t_list[t]                    # [B, E]
      h_t = h_t_list[t]                    # [B, H]
      h_prev = h_prev_list[t]              # [B, H]
      dlogits_t = dlogits[:, t, :]         # [B, V]

      # Output layer gradients
      dWhy += h_t.T @ dlogits_t            # [H, V]
      db_y += dlogits_t.sum(dim=0)         # [V]

      # Hidden state gradients
      dh = dlogits_t @ W_hy.T + dh_next    # [B, H]
      da = (1 - h_t**2) * dh               # [B, H]

      # Parameter gradients
      dWxh += x_t.T @ da                   # [E, H]
      dWhh += h_prev.T @ da                # [H, H]
      db_h += da.sum(dim=0)                # [H]

      # Embedding gradients
      dx_t = da @ W_xh.T                   # [B, E]
      for b in range(B):
          token_id = xb[b, t]
          dW_emb[token_id] +=   dx_t[b]

      # Propagate gradient to previous timestep
      dh_next = da @ W_hh.T           # [B, H]

  return loss, dWxh , dWhh , dWhy , db_h , db_y , dW_emb 


def generate_text(start_token=0, max_length=200):
    idx = torch.tensor([[start_token]], device=device)  # [B=1, T=1]
    h = torch.zeros(1, hidden_size, device=device)
    out = []

    for _ in range(max_length):
        x_t = W_emb[idx[:, -1]]
        h = torch.tanh(x_t @ W_xh + h @ W_hh + b_h)
        logits = h @ W_hy + b_y
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        out.append(next_token.item())
        idx = torch.cat([idx, next_token.unsqueeze(1)], dim=1)

    return ''.join([idx_to_char[i] for i in out])


# ---------- Training -----------------
initial_lr = 0.01
epochs = 6000
smooth_loss = None
beta = 0.95
losses = []
grad_norms = []
total_norms = []
smooth_losses = []
val_losses =[]

params = [W_xh, W_hh, W_hy, b_h, b_y, W_emb]

optimizer = optim.Adam(params, lr=0.001)

for epoch in range(epochs):
    X, y = get_batch("train")

    # 1. Forward pass
    logits, cache = rnn_forward(X)

    # 2. Backward pass (loss + grads)
    optimizer.zero_grad()
    loss, dWxh, dWhh, dWhy, db_h, db_y, dW_emb = calculate_loss_backward(logits, y, cache, X)
    losses.append(loss.item())

    if smooth_loss is None:
        smooth_loss = loss
    else:
        smooth_loss = beta * smooth_loss + (1 - beta) * loss

    smooth_losses.append(smooth_loss.item())

    # 3. Gradient clipping
    grads = [dWxh, dWhh, dWhy, db_h, db_y, dW_emb]
    total_norm = clip_gradients(grads, max_norm=0.5)
    grads  = normalize_gradients(grads)
    total_norms.append(total_norm.item())

    for param, grad in zip(params, grads):
        param.grad = grad

    # 4. Parameter update
    optimizer.step()

    # validation 
    with torch.no_grad():
        if epoch % 100 == 0:
          xb_val, yb_val = get_batch("val")
          val_logits, cache_val = rnn_forward(xb_val)
          val_loss, *_ = calculate_loss_backward(val_logits, yb_val, cache_val, xb_val)
          val_losses.append(val_loss.item())
          print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Val Loss: {val_loss.item()}")
          print(generate_text())
          print(100*"-")


save_training_results(losses,smooth_losses,val_losses)
