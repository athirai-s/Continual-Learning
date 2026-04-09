"""Debug: check what probes look like and what the model generates."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from casf_dataset_api import TemporalWikiDataset

MODEL_PATH = "/scratch1/ashanmug/models/Llama-3.2-3B-Instruct"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
model.to(device)

# Check aug_sep changed probes
dataset = TemporalWikiDataset(period="aug_sep")
dataset.load("changed")
probes = dataset.get_probes("changed")

print(f"Total probes: {len(probes)}")
print(f"\nFirst 10 probes - prompt, expected answer, and model output:")
print("=" * 80)

for i, probe in enumerate(probes[:10]):
    encoded = tokenizer(
        probe.prompt,
        truncation=True,
        max_length=512,
        padding="do_not_pad",
        return_tensors="pt",
    )
    batch = {k: v.to(device) for k, v in encoded.items()}
    prompt_len = batch["input_ids"].shape[1]

    with torch.no_grad():
        output_short = model.generate(**batch, max_new_tokens=8, pad_token_id=tokenizer.eos_token_id)
        output_long = model.generate(**batch, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id)

    gen_short = tokenizer.decode(output_short[0][prompt_len:], skip_special_tokens=True)
    gen_long = tokenizer.decode(output_long[0][prompt_len:], skip_special_tokens=True)

    print(f"\n[{i}] Prompt:    {probe.prompt}")
    print(f"    Expected:  {probe.ground_truth}")
    print(f"    Got (8):   {gen_short}")
    print(f"    Got (32):  {gen_long}")
    print(f"    Relation:  {probe.relation}")
    print(f"    Subject:   {probe.subject}")
    gt_in_short = probe.ground_truth.lower() in gen_short.lower()
    gt_in_long = probe.ground_truth.lower() in gen_long.lower()
    print(f"    Contains(8)?  {gt_in_short}")
    print(f"    Contains(32)? {gt_in_long}")