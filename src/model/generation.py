"""Greedy generation for periodic validation, using the model's K/V cache."""

import torch


@torch.inference_mode()
def greedy_generate(model, encoder_input_ids, encoder_attention_mask, max_new_tokens):
    """Return generated IDs without BOS. The model must be in eval mode.

    Finished rows emit EOS while other rows continue. Stop when every row has
    reached EOS or the token limit is reached; never feed generated PAD/BOS.
    """
    memory = model.encode(encoder_input_ids, encoder_attention_mask)
    token_ids = torch.full(
        (encoder_input_ids.size(0), 1),
        model.bos_id,
        dtype=torch.long,
        device=encoder_input_ids.device,
    )
    finished = torch.zeros(
        encoder_input_ids.size(0), dtype=torch.bool, device=encoder_input_ids.device
    )
    cache = None
    generated = []
    for _ in range(max_new_tokens):
        logits, cache = model.decode_step(
            token_ids, memory, encoder_attention_mask, cache
        )
        logits[:, [model.pad_id, model.bos_id]] = float("-inf")
        next_token = logits.argmax(-1).masked_fill(finished, model.eos_id)
        generated.append(next_token)
        finished |= next_token.eq(model.eos_id)
        if finished.all():
            break
        token_ids = next_token[:, None]
    return torch.stack(generated, dim=1)
