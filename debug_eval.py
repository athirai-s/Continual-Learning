"""Debug: compare raw vs instruct prompting on TemporalWiki probes."""

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


def generate(prompt, max_tokens=16):
    encoded = tokenizer(prompt, truncation=True, max_length=512, padding="do_not_pad", return_tensors="pt")
    batch = {k: v.to(device) for k, v in encoded.items()}
    prompt_len = batch["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**batch, max_new_tokens=max_tokens, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)


def make_instruct_prompt(cloze_prompt):
    """Wrap a cloze prompt in Llama 3 instruct chat format."""
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Answer in one or two words only. {cloze_prompt}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


print(f"\nFirst 15 probes - raw vs instruct prompting:")
print("=" * 80)

raw_contains = 0
instruct_contains = 0

for i, probe in enumerate(probes[:15]):
    raw_out = generate(probe.prompt, max_tokens=16)
    instruct_prompt = make_instruct_prompt(probe.prompt)
    instruct_out = generate(instruct_prompt, max_tokens=16)

    gt = probe.ground_truth.lower()
    raw_hit = gt in raw_out.lower()
    instruct_hit = gt in instruct_out.lower()
    if raw_hit:
        raw_contains += 1
    if instruct_hit:
        instruct_contains += 1

    print(f"\n[{i}] Cloze:      {probe.prompt}")
    print(f"    Expected:   {probe.ground_truth}")
    print(f"    Raw (16):   {raw_out.strip()[:80]}")
    print(f"    Instruct:   {instruct_out.strip()[:80]}")
    print(f"    Raw hit?    {raw_hit}")
    print(f"    Instruct?   {instruct_hit}")

print(f"\n{'='*80}")
print(f"Raw contains:      {raw_contains}/15")
print(f"Instruct contains: {instruct_contains}/15")