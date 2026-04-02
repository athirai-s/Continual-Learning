import torch

from casf_dataset_api import Probe, TemporalEvaluator
from training.evaluation_runner import GenerationAdapter


class FakeTokenizer:
    model_max_length = 16
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, truncation=True, max_length=16, padding=None, return_tensors=None):
        _ = text, truncation, max_length, padding
        payload = {
            "input_ids": torch.tensor([[10, 11, 12]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }
        return payload if return_tensors == "pt" else {k: v.tolist() for k, v in payload.items()}

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        mapping = {
            10: "Question:",
            11: "Ada Example",
            12: "Answer:",
            13: "unused",
            1: "",
        }
        parts = []
        for token_id in token_ids:
            if skip_special_tokens and token_id == self.eos_token_id:
                continue
            parts.append(mapping.get(token_id, str(token_id)))
        return " ".join(part for part in parts if part).strip()


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def generate(self, input_ids, attention_mask, max_new_tokens, pad_token_id, eos_token_id):
        _ = attention_mask, max_new_tokens, pad_token_id
        eos = torch.tensor([[eos_token_id]], dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, eos], dim=1)


def test_generation_adapter_scores_only_generated_continuation():
    adapter = GenerationAdapter(FakeModel(), FakeTokenizer())
    evaluator = TemporalEvaluator()
    probe = Probe(
        prompt="Question: Ada Example\n\nContext: Ada Example is in the prompt.\n\nAnswer:",
        ground_truth="Ada Example",
        relation="qa",
        subject="Who leads Exampleland?",
        current_value="Ada Example",
        source="synthetic",
    )

    exact, f1 = evaluator.score_probe(adapter, probe)

    assert exact is False
    assert f1 == 0.0
