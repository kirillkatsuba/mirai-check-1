import random
from collections.abc import Callable, Iterable
from typing import NamedTuple
import math
from lalamo.data.lalamo_completions import LalamoCompletion
from lalamo.speculator.common import Speculator


class SpeculatorTrainingEvent(NamedTuple):
    trained_sequences: int
    trained_tokens: int


def train_speculator(
    speculator: Speculator,
    traces: Iterable[LalamoCompletion],
    tokens_to_train: int | None = None,
    progress_callback: Callable[[SpeculatorTrainingEvent], None] | None = None,
) -> None:
    trained_tokens = 0

    for trained_sequences, trace in enumerate(traces, start=1):
        if tokens_to_train is not None and trained_tokens + len(trace.completion_token_ids) > tokens_to_train:
            end = tokens_to_train - trained_tokens
        else:
            end = None
        token_ids = trace.completion_token_ids[:end]
        token_logits = trace.completion_token_logits[:end]

        speculator.train(token_ids, token_logits)

        trained_tokens += len(token_ids)

        if progress_callback is not None:
            progress_callback(SpeculatorTrainingEvent(trained_sequences, trained_tokens))

        if tokens_to_train is not None and trained_tokens >= tokens_to_train:
            break


def test_speculator(
    speculator: Speculator,
    sequence: Iterable[int] = [],
    max_completion_length: int = 32,
    temperature: float = 1.0,
) -> list[int]:
    sequence = list(sequence)
    
    for _ in range(max_completion_length):
        # Получаем скоры (Gumbel scores для Gumbel-версии или Probs для NGram)
        scores = speculator.probs(sequence)
        if not scores:
            break

        # Проверка: это Gumbel Scores или обычные вероятности?
        # Gumbel-скоры часто отрицательные и не суммируются в 1.
        is_gumbel = any(v < 0 for v in scores.values()) or abs(sum(scores.values()) - 1.0) > 0.01

        if is_gumbel:
            # Математически, выбор из Gumbel-распределения — это Argmax от скоров.
            # Если мы хотим «разнообразия» в тесте, используем Softmax-семплирование.
            
            keys = list(scores.keys())
            vals = list(scores.values())
            
            # Применяем Softmax для превращения скоров в вероятности для семплирования
            max_val = max(vals)
            exp_vals = [math.exp((v - max_val) / temperature) for v in vals]
            total = sum(exp_vals)
            weights = [v / total for v in exp_vals]
            
            selected = random.choices(keys, weights=weights, k=1)[0]
        else:
            # Стандартная логика для N-Gram (вероятности)
            keys = list(scores.keys())
            weights = list(scores.values())
            if sum(weights) <= 0:
                break
            selected = random.choices(keys, weights=weights, k=1)[0]
            
        sequence.append(selected)
        
    return sequence