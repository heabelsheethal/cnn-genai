# helper_lib/assignment5_llm/llm_finetune_gpt2.py

# helper_lib/assignment5_llm/llm_finetune_gpt2.py

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from datasets import load_dataset
from tqdm import tqdm

# -----------------------------------
# Dataset Wrapper (SQuAD → GPT-2)
# -----------------------------------

class SquadQADataset(Dataset):
    def __init__(self, split="train", tokenizer=None, max_len=384):
        # self.data = load_dataset("rajpurkar/squad", split=split)
        # self.data = load_dataset("rajpurkar/squad", split=split)
        # self.data = full.select(range(5000)) 
        full = load_dataset("rajpurkar/squad", split=split)
        self.data = full.select(range(7000))   
        # full = load_dataset("rajpurkar/squad", split=split)
        # self.data = full[:5000]   # old-style subset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        q = item["question"]
        a = item["answers"]["text"][0]

        text = (
            f"Question: {q}\n"
            f"That is a great question. {a} Let me know if you have any other questions."
        )

        tokens = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return tokens.input_ids.squeeze(), tokens.attention_mask.squeeze()


'''
class SquadQADataset(Dataset):
    def __init__(self, split="train", tokenizer=None, max_len=384):
        full = load_dataset("rajpurkar/squad", split=split)
        self.data = full.select(range(5000))      # FAST + GOOD RESULTS
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        q = item["question"]
        a = item["answers"]["text"][0]

        text = (
            f"Question: {q}\n"
            f"That is a great question. {a} Let me know if you have any other questions."
        )

        tokens = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return tokens.input_ids.squeeze(), tokens.attention_mask.squeeze()

'''
# -----------------------------------
# FINE-TUNE MODEL
# -----------------------------------
def train():
    print("Loading tokenizer + model...")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    model.config.pad_token_id = model.config.eos_token_id

    print("Loading SQuAD dataset (this may take ~10 sec)...")
    train_dataset = SquadQADataset(split="train", tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    print("Starting fine-tuning...")

    for epoch in range(1):
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=120)

        for i, (input_ids, attention_mask) in enumerate(progress):
            # if i > 5000:   # stop early for speed
            #    break

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar with live loss
            progress.set_postfix(loss=f"{loss.item():.4f}")

    print("Saving fine-tuned GPT-2...")
    model.save_pretrained("./finetuned_gpt2")
    tokenizer.save_pretrained("./finetuned_gpt2")

    print("Fine-tuned GPT-2 saved to ./finetuned_gpt2/")

if __name__ == "__main__":
    train()