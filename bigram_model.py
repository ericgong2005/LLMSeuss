import torch
import torch.nn as nn
from torch.nn import functional as F

import re
import matplotlib.pyplot as plt
from datetime import datetime

# hyperparameters
batch_size = 32
block_size = 8
iterations = 10000
iteration_checkpoint = 100
learning_rate = 1e-3
device = "mps" if torch.backends.mps.is_available() else 'cpu' # For MacOS GPU acceleration
eval_iters = 10

torch.manual_seed(1234)

print(f"Using {device}\n\n")

# Import the text for training the model
with open('seuss_works.txt', 'r', encoding='utf-8') as f:
    training_text = f.read()

pattern = r'(\w+|[^\w\s]|\s|\n)'
split = re.findall(pattern, training_text)
formatted = []
for word in split:
    if word.istitle():
        formatted.append("<C>")
        formatted.append(word.lower())
    elif word.isupper():
        formatted.append("<A>")
        formatted.append(word.lower())
    else:
        formatted.append(word)

unique = set(formatted)

training_data = formatted
tokens = unique

# Defining an Encoder and Decoder
sorted_tokens = sorted(tokens)
encode_dict = {element: idx for idx, element in enumerate(sorted_tokens)}
decode_dict = {idx: element for idx, element in enumerate(sorted_tokens)}

def encoder(text: list[str]) -> list[int] :
    return [encode_dict[element] for element in text]

def decoder(code: list[int]) -> list[str] :
    return [decode_dict[element] for element in code]

def full_decode(code: list[str]) -> str :
    status = 0
    final = ""
    for element in code:
        if element == "<C>":
            status = 1
        elif element == "<A>":
            status = 2
        else:
            if status == 0:
                final += element
            elif status == 1:
                final += element.capitalize()
            else:
                final += element.upper()
            status = 0
    return final

# Training and Validation Split
data = torch.tensor(encoder(training_data), dtype=torch.long)
n = int(0.9*len(data))
training = data[:n]
validation = data[n:]
vocab_size = len(tokens)

# data loading
def get_batch(type: str):
    # generate a batch of inputs x and targets y
    data = training if split == 'training' else validation
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['training', 'validation']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Bigram Language Model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (Batch x Time) tensors
        logits = self.token_embedding_table(idx) # (Batch x Time x Channels)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (Batch x Time) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last Time step
            logits = logits[:, -1, :] # becomes (Batch x Channels)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (Batch x Channels)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (Batch x 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (Batch x Time + 1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

train_losses = []
val_losses = []
iteration_steps = []

for iter in range(iterations):

    # Periodically evaluate the loss on train and val sets
    if iter % iteration_checkpoint == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['training']:.4f}, val loss {losses['validation']:.4f}")
        train_losses.append(losses["training"])
        val_losses.append(losses["validation"])
        iteration_steps.append(iter)

    # sample a batch of data
    xb, yb = get_batch('training')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)

generation = full_decode(decoder(m.generate(context, max_new_tokens=500)[0].tolist()))

current_time = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
with open(f"Bigram_model_generations/bigram_generation_{current_time}.txt", "w") as f:
    f.write(generation)

# Plot the Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(iteration_steps, train_losses, label='Training Loss')
plt.plot(iteration_steps, val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
hyperparameters_text = (f"Hyperparameters:\n"
                        f"Batch size: {batch_size}\n"
                        f"Block size: {block_size}\n"
                        f"Iterations: {iterations}\n"
                        f"Learning rate: {learning_rate}\n"
                        f"Device: {device}\n")
plt.text(0.95, 0.95, hyperparameters_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
plt.savefig(f"Bigram_model_loss_plots/bigram_loss_{current_time}.png")
plt.close()