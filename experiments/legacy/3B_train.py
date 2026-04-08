import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
from casf_dataset_api import TSQADataset, TGQADataset

MODEL_ID = "/scratch1/ashanmug/models/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


class ModelWrapper:
    def __init__(self, model, tokenizer, max_new_tokens=64):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()


wrapper = ModelWrapper(model, tokenizer)


print("TSQA:")
tsqa = TSQADataset()
tsqa.load("train")
tsqa_probes = tsqa.get_probes()
clean_probes = [p for p in tsqa_probes if not p.metadata.get("is_hard_negative")]
perturbed_probes = [p for p in tsqa_probes if p.metadata.get("is_hard_negative")]

print("TGQA:")
tgqa = TGQADataset()
tgqa.load("train")
tgqa_probes = tgqa.get_probes()

print(f"TS-QA clean: {len(clean_probes)}")
print(f"TS-QA perturbed: {len(perturbed_probes)}")
print(f"TGQA cloze: {len(tgqa_probes)}")


#Evaluation 
def eval_all(wrapper, tag=""):
    print(f"\n{'='*60}")
    print(f"  {tag}")
    print(f"{'='*60}")
    for name, probes in [
        ("TS-QA clean", clean_probes),
        ("TS-QA perturbed", perturbed_probes),
        ("TGQA cloze", tgqa_probes),
    ]:
        sample = probes[:200]
        exact, contains, f1s = 0, 0, []
        for p in sample:
            output = wrapper.generate(p.prompt)
            gt = p.ground_truth.lower()
            out_lower = output.lower()
            if out_lower == gt:
                exact += 1
            if gt in out_lower:
                contains += 1
            pred_tok = set(out_lower.split())
            gt_tok = set(gt.split())
            if pred_tok and gt_tok:
                prec = len(pred_tok & gt_tok) / len(pred_tok)
                rec = len(pred_tok & gt_tok) / len(gt_tok)
                f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0)
            else:
                f1s.append(0)
        n = len(sample)
        print(f"  {name:20s}  n={n}  exact={exact/n:.3f}  contains={contains/n:.3f}  f1={sum(f1s)/n:.5f}")


def probes_to_hf(probes):
    return Dataset.from_dict({"text": [f"{p.prompt} {p.ground_truth}" for p in probes]})


# ── Baseline evaluation (already recorded) ──
# BASELINE: TS-QA clean  contains=0.355 f1=0.096 | TS-QA perturbed contains=1.000 f1=0.000 | TGQA cloze contains=0.000 f1=0.035
# eval_all(wrapper, "BASELINE")


datasets_seq = [
    ("tsqa_clean", clean_probes),
    ("tsqa_perturbed", perturbed_probes),
    ("tgqa_cloze", tgqa_probes),
]

for name, probes in datasets_seq:
    print(f"\nTraining on {name} ({len(probes)} probes)")
    train_ds = probes_to_hf(probes)

    args = TrainingArguments(
        output_dir=f"./checkpoints/{name}",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
    )
    trainer.train()
    model.save_pretrained(f"./checkpoints/{name}/final")

    eval_all(wrapper, f"After {name}")

print("\nDone!")
