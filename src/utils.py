import torch
from transformers import LogitsProcessor

# Define a custom LogitsProcessor
class ConstrainedOutputLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, valid_outputs):
        self.valid_token_ids = [
            tokenizer.convert_tokens_to_ids(c) for c in valid_outputs
        ]

    def __call__(self, input_ids, scores):
        mask = torch.ones_like(scores) * float('-inf')
        for valid_token_id in self.valid_token_ids:
            mask[:, valid_token_id] = scores[:, valid_token_id]
        return mask