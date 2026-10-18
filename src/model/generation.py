"""Greedy and beam generation with a shared encoder output and K/V caches."""

import torch


def generate(
    model,
    encoder_input_ids,
    encoder_attention_mask,
    max_new_tokens,
    num_beams=1,
    length_penalty=1.0,
):
    if max_new_tokens < 1 or num_beams < 1 or length_penalty < 0:
        raise ValueError("Token/beam counts must be positive; length_penalty >= 0.")
    if num_beams == 1:
        return greedy_generate(
            model, encoder_input_ids, encoder_attention_mask, max_new_tokens
        )
    return beam_generate(
        model,
        encoder_input_ids,
        encoder_attention_mask,
        max_new_tokens,
        num_beams,
        length_penalty,
    )


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


@torch.inference_mode()
def beam_generate(
    model,
    encoder_input_ids,
    encoder_attention_mask,
    max_new_tokens,
    num_beams=3,
    length_penalty=1.0,
):
    """Return one sequence per question, ranked by log P / length**penalty.

    Finished hypotheses are kept separately from active beams. EOS contributes
    to the length. The stopping bound assumes nonnegative length_penalty.
    """
    batch_size = encoder_input_ids.size(0)
    device = encoder_input_ids.device
    memory = model.encode(encoder_input_ids, encoder_attention_mask)
    memory = memory.repeat_interleave(num_beams, dim=0)
    if encoder_attention_mask is not None:
        encoder_attention_mask = encoder_attention_mask.repeat_interleave(
            num_beams, dim=0
        )
    token_ids = torch.full(
        (batch_size * num_beams, 1), model.bos_id, dtype=torch.long, device=device
    )
    scores = torch.full((batch_size, num_beams), -torch.inf, device=device)
    scores[:, 0] = 0
    sequences = token_ids.new_empty((batch_size, num_beams, 0))
    best_sequences = token_ids.new_full((batch_size, max_new_tokens), model.eos_id)
    best_scores = scores.new_full((batch_size,), -torch.inf)
    rows = torch.arange(batch_size, device=device)
    offsets = rows[:, None] * num_beams
    cache = None

    for step in range(max_new_tokens):
        logits, cache = model.decode_step(
            token_ids, memory, encoder_attention_mask, cache
        )
        logits[:, [model.pad_id, model.bos_id]] = -torch.inf
        log_probs = logits.float().log_softmax(-1)
        vocab_size = logits.size(-1)
        candidates = scores[:, :, None] + log_probs.view(
            batch_size, num_beams, vocab_size
        )
        top_scores, top_ids = candidates.flatten(1).topk(2 * num_beams, dim=-1)
        parents, tokens = top_ids // vocab_size, top_ids % vocab_size

        # Only EOS among the best `num_beams` candidates can finish a beam.
        completed = top_scores[:, :num_beams] / (step + 1) ** length_penalty
        completed = completed.masked_fill(
            tokens[:, :num_beams].ne(model.eos_id), -torch.inf
        )
        completed_scores, completed_indices = completed.max(-1)
        completed_parents = parents[rows, completed_indices]
        completed_sequences = torch.cat(
            (sequences[rows, completed_parents], tokens[rows, completed_indices, None]),
            dim=-1,
        )
        improved = completed_scores > best_scores
        best_sequences[:, : step + 1] = torch.where(
            improved[:, None], completed_sequences, best_sequences[:, : step + 1]
        )
        best_scores = torch.maximum(best_scores, completed_scores)

        active = top_scores.masked_fill(tokens.eq(model.eos_id), -torch.inf)
        scores, active_indices = active.topk(num_beams, dim=-1)
        parents = parents.gather(1, active_indices)
        tokens = tokens.gather(1, active_indices)
        sequences = torch.cat(
            (
                sequences.gather(1, parents[:, :, None].expand_as(sequences)),
                tokens[:, :, None],
            ),
            dim=-1,
        )
        cache_indices = (parents + offsets).flatten()
        cache = [
            tuple(
                tuple(tensor.index_select(0, cache_indices) for tensor in attention)
                for attention in layer
            )
            for layer in cache
        ]
        token_ids = tokens.reshape(-1, 1)

        # Future log probabilities cannot increase; the longest allowed length
        # gives an optimistic bound for normalized scores of active hypotheses.
        done = best_scores >= scores[:, 0] / max_new_tokens**length_penalty
        scores = scores.masked_fill(done[:, None], -torch.inf)
        if done.all():
            break

    final_scores, final_indices = (scores / (step + 1) ** length_penalty).max(-1)
    return torch.where(
        (final_scores > best_scores)[:, None],
        sequences[rows, final_indices],
        best_sequences[:, : step + 1],
    )
