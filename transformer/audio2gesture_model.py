import os
import json
import math
import torch
import torchaudio
from pathlib import Path
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModel, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from contextlib import nullcontext
from transformers import Wav2Vec2Processor, Wav2Vec2Model

AUDIO_MODEL_NAME   = "facebook/wav2vec2-base-960h"   # any HF audio encoder w/ 768-d states
SAMPLE_RATE        = 16_000                          # must match encoder
CODEBOOK_SIZE      = 256                           # VQ-VAE codebook length
MAX_DEC_LEN        = 256                         # max motion-token length
BATCH_SIZE         = 64
NUM_EPOCHS         = 1
LR                 = 5e-5
WARMUP_STEPS       = 1_000
DEVICE             = "cuda"

PAD_ID, BOS_ID, EOS_ID = CODEBOOK_SIZE, CODEBOOK_SIZE + 1, CODEBOOK_SIZE + 2
VOCAB_SIZE        = CODEBOOK_SIZE + 3                # +PAD, BOS, EOS

# ---------------------------------------------------------------
# 2️⃣  Dataset
#     Assumes a simple JSON-lines manifest: one example per line:
#     {"audio":"/abs/path/sample.wav", "tokens":[12,87,63,...]}
# ---------------------------------------------------------------

class AudioMotionDataset(Dataset):
    def __init__(self, manifest_path: str):
        self.items = [json.loads(l) for l in Path(f"./{manifest_path}").read_text().splitlines()]
        self.processor = Wav2Vec2Processor.from_pretrained(AUDIO_MODEL_NAME)

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        folder = "../data/wav2clips/"

        waveform, sample_rate = torchaudio.load(folder + item["audioname"])

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        
        # Process audio
        wav_inputs = self.processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
        tokens = torch.tensor(item["preds"], dtype=torch.long)
        return wav_inputs["input_values"].squeeze(), tokens.squeeze()

def collate_fn(batch):
    """Pad audio & tokens independently."""
    wavs, toks = zip(*batch)

    wav_lens   = [len(w) for w in wavs]
    toks_lens  = [len(t) for t in toks]

    # --- audio padding ---
    max_wav = max(wav_lens)
    wav_pad = torch.zeros(len(batch), max_wav) # B, MAX_W

    for i, w in enumerate(wavs):
        wav_pad[i, :w.size(0)] = w

    # --- token padding & teacher-forcing shift ---
    max_tok = max(toks_lens) + 1          # +1 for BOS/EOS shift
    dec_in  = torch.full((len(batch), max_tok), PAD_ID, dtype=torch.long)
    labels  = torch.full((len(batch), max_tok), PAD_ID, dtype=torch.long)

    for i, t in enumerate(toks):
        L = t.size(0)
        dec_in[i, 0]   = BOS_ID           # BOS
        dec_in[i, 1:L+1] = t
        labels [i, :L]  = t
        labels [i, L]   = EOS_ID          # EOS

    enc_pad_mask = torch.arange(max_wav).expand(len(batch), -1) >= torch.tensor(wav_lens).unsqueeze(1)
    dec_pad_mask = dec_in == PAD_ID
    return wav_pad, dec_in, labels, enc_pad_mask, dec_pad_mask

# ---------------------------------------------------------------
# 3️⃣  Model
# ---------------------------------------------------------------
class Audio2Motion(nn.Module):
    def __init__(self,
                 vocab_size: int = VOCAB_SIZE,
                 d_model: int   = 1024,
                 n_layers: int  = 8,
                 n_heads: int   = 8,
                 ff_dim: int    = 4096,
                 max_len:  int  = MAX_DEC_LEN):
        super().__init__()

        self.encoder = Wav2Vec2Model.from_pretrained(AUDIO_MODEL_NAME)
        for p in self.encoder.parameters(): p.requires_grad = False
        
        # token & position embedding
        self.tok_emb  = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model))

        #Audio Projector
        self.audio_head = nn.Linear(768, d_model)

        # decoder
        dec_layer = nn.TransformerDecoderLayer(d_model, n_heads, ff_dim, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, n_layers)

        # LM head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.uniform_(self.lm_head.weight, a=-0.02, b=0.02)

        print("Init Completed")

    @staticmethod
    def _causal_mask(size: int, device):
        return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), 1)

    def forward(self,
                wav: Tensor,
                tgt_in: Tensor,
                enc_pad_mask: Tensor,
                tgt_pad_mask: Tensor):
        # wav, dec_in, enc_mask, dec_mask
        """
        wav:        (B, T_audio)
        tgt_in:     (B, T_dec)  teacher-forcing tokens incl. BOS
        *_pad_mask: bool masks with True where PAD
        """

        with torch.no_grad():
            memory = self.encoder(wav, attention_mask=~enc_pad_mask).last_hidden_state  # (B,T_enc,768)
        
        memory = self.audio_head(memory)

        seq_len = tgt_in.size(1)
        tgt = self.tok_emb(tgt_in) + self.pos_emb[:, :seq_len, :]
        causal = self._causal_mask(seq_len, tgt.device)

        dec_out = self.decoder(
            tgt,
            memory,
            tgt_mask = causal,
            tgt_key_padding_mask = tgt_pad_mask,
            memory_key_padding_mask  = None
        )  
        logits = self.lm_head(dec_out) 
        return logits            # logits

# ---------------------------------------------------------------
# 4️⃣  Train & Evaluate
# ---------------------------------------------------------------
def train_loop(train_loader, model, optimizer, scheduler, scaler = None):
    total_loss = 0
    # wav_pad, dec_in, labels, enc_pad_mask, dec_pad_mask
    for wav, dec_in, labels, enc_mask, dec_mask in tqdm(train_loader):
        wav, dec_in, labels = wav.to(DEVICE), dec_in.to(DEVICE), labels.to(DEVICE)
        enc_mask, dec_mask  = enc_mask.to(DEVICE), dec_mask.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        logits = model(wav, dec_in, enc_mask, dec_mask) 

        loss = F.cross_entropy(
            logits.transpose(1,2),                        # (B,V,T)
            labels,
            ignore_index=PAD_ID
        )

        print(loss)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   

        optimizer.step()

        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# ---------------------------------------------------------------
# 5️⃣  Autoregressive Generation
# ---------------------------------------------------------------
@torch.no_grad()
def generate_motion(model: Audio2Motion, wav_path: str, top_k: int = 0, top_p: float = 0.95,
                    max_len: int = 480):
    
    wav, sr = torchaudio.load(wav_path)
    if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            wav = resampler(wav)
                    # (1,T)

    wav = wav.to(DEVICE)
    memory = model.encoder(wav)["last_hidden_state"]               # (1,T_enc,768)
    memory = model.audio_head(memory)
    tokens = torch.tensor([[BOS_ID]], device=DEVICE)
    for _ in range(max_len):
        tgt_mask = model._causal_mask(tokens.size(1), tokens.device)
        dec_out  = model.decoder(
            model.tok_emb(tokens) + model.pos_emb[:, :tokens.size(1), :],
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask = None
        )
        logits = model.lm_head(dec_out[:, -1])                      # (1,V)

        # --- sampling ---
        if top_k > 0:
            logits_topk, idx = torch.topk(logits, top_k)
            probs = F.softmax(logits_topk, -1)
            next_tok = idx[0, torch.multinomial(probs, 1)]
        else:
            probs = F.softmax(logits, -1)
            next_tok = torch.multinomial(probs, 1)[0]
        tokens = torch.cat([tokens, next_tok.unsqueeze(0)], dim=1)
        if next_tok.item() == EOS_ID: break

    return tokens[0, 1:].tolist()                                  # strip BOS

# ---------------------------------------------------------------
# 6️⃣  Main
# ---------------------------------------------------------------
def main(manifest_train: str):
    ds  = AudioMotionDataset(manifest_train)
    dl  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                     collate_fn=collate_fn, num_workers=0)
            
    model = Audio2Motion().to(DEVICE)

    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2, betas=(0.9,0.98))
    scheduler  = get_linear_schedule_with_warmup(optimizer,
                                                 num_warmup_steps=WARMUP_STEPS,
                                                 num_training_steps = NUM_EPOCHS * len(dl))

    for epoch in range(NUM_EPOCHS):
        loss = train_loop(dl, model, optimizer, scheduler)
        print(f"[Epoch {epoch+1}] loss = {loss:.4f}")

    torch.save(model.state_dict(), './saved_models/a2t_model.pth')

    # quick qualitative check

    folder = "../data/wav2clips/"
    sample_path = ds.items[0]["audioname"]
    pred_tokens = generate_motion(model, folder + sample_path)
    print("Generated token ids:", pred_tokens)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_manifest", required=True, help="JSONL list of audio+token pairs")
    args = ap.parse_args()
    main(args.train_manifest)
