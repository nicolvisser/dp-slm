from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from .config import ELLAVModelArgs
from .tokenizer import ELLAVTokenizer
from .transformer import TransformerModel


class ELLAVGARModel(nn.Module):
    """ELLA-V GAR model to predict first RVQ level"""

    def __init__(self, model_args: ELLAVModelArgs, tokenizer: ELLAVTokenizer):
        super().__init__()
        self.model_args = model_args
        self.tokenizer = tokenizer

        self.token_embedding = nn.Embedding(
            num_embeddings=self.tokenizer.vocab_size,
            embedding_dim=model_args.transformer.dim,
        )

        self.transformer = TransformerModel(args=model_args.transformer)

        self.prediction_head = nn.Linear(
            in_features=model_args.transformer.dim,
            out_features=self.tokenizer.vocab_size,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self, src_ids: torch.Tensor, q_seqlen: List[int], return_last: bool = False
    ) -> torch.Tensor:
        x = self.token_embedding(input=src_ids)  # (T, D)
        x = self.transformer(embeddings=x, q_seqlen=q_seqlen)  # (T, D)
        if return_last:
            # Get the last token for each sequence in the batch
            last_indices = torch.tensor(
                [sum(q_seqlen[: i + 1]) - 1 for i in range(len(q_seqlen))],
                device=x.device,
            )
            x = x[last_indices]  # (B, D)
        x = self.prediction_head(input=x)  # (T, E) or (B, E)
        return (
            x.squeeze(0) if return_last and len(q_seqlen) == 1 else x
        )  # (E,) if return_last and batch=1 else (T, E) or (B, E)

    @property
    def inference_mask(self) -> torch.Tensor:
        """Mask for zeroing out unwanted tokens during inference. Lazily initialized."""
        if not hasattr(self, "_inference_mask"):
            self._inference_mask = self.tokenizer.inference_zero_out_ids.to(self.device)
        elif self._inference_mask.device != self.device:
            self._inference_mask = self._inference_mask.to(self.device)
        return self._inference_mask

    @torch.inference_mode
    def generate(
        self,
        phoneme_tokens_lists: List[List[str]],
        top_p: float = 0.8,
        temperature: float = 1.0,
        max_phoneme_duration: int = 0.4,
        show_progress: bool = False,
    ) -> torch.Tensor:
        return generate(
            phoneme_tokens_lists,
            model=self,
            top_p=top_p,
            temperature=temperature,
            max_phoneme_duration=max_phoneme_duration,
            show_progress=show_progress,
        )

    @classmethod
    def from_model_dir(cls, model_dir: str):
        model_dir = Path(model_dir)
        tokenizer_path: Path = model_dir / "tokenizer.json"
        checkpoint_path: Path = model_dir / "model.pt"
        model_args_path: Path = model_dir / "model_args.json"
        assert tokenizer_path.exists(), tokenizer_path
        assert checkpoint_path.exists(), checkpoint_path
        assert model_args_path.exists(), model_args_path
        model = cls(
            model_args=ELLAVModelArgs.load(model_args_path),
            tokenizer=ELLAVTokenizer.load_json(tokenizer_path),
        )
        model.load_state_dict(torch.load(checkpoint_path))
        return model


class ELLAVInferenceHandler:
    def __init__(
        self,
        phoneme_ids: List[int],
        model: ELLAVGARModel,
        top_p: float,
        max_acoustic_tokens_per_phoneme: int,
        show_progress: bool = False,
    ):
        # save references
        self.model = model
        self.tokenizer = model.tokenizer
        # save params
        self.top_p = top_p
        self.max_acoustic_tokens_per_phoneme = max_acoustic_tokens_per_phoneme

        # copy phoneme ids and remove ids not to sandwich (e.g. ["SIL"] for TTS models)
        self.phoneme_ids = [
            id for id in phoneme_ids if id not in self.tokenizer.ids_not_to_sandwich
        ]

        # initialize progress bar
        if show_progress:
            self.pbar = tqdm(
                total=len(self.phoneme_ids), desc="Running ELLA-V GAR Model"
            )

        # initialize a few states:
        self.ids = []  # the ids to process
        self.current_idx = (
            0  # the index of the item in the self.ids list to process next
        )
        self.current_phoneme_idx = (
            -1
        )  # the index of the phoneme in the self.phoneme_ids list that we are currently predicting acoustic tokens for
        self.num_acoustic_tokens_per_phoneme = [
            0 for _ in range(len(self.phoneme_ids))
        ]  # how many acoustic tokens have been predicted for each phoneme in self.phoneme_ids
        self.is_waiting_to_sample = (
            False  # do we need to sample the next token, or is it deterministic?
        )
        self.is_finished = False  # are we done with inference?

    @property
    def num_phoneme_ids(self):
        return len(self.phoneme_ids)

    @property
    def num_ids(self):
        return len(self.ids)

    def step(self, sampled_id: Optional[int] = None):
        if self.current_idx < self.num_phoneme_ids:
            # add phoneme in global advance section
            self.ids.append(self.phoneme_ids[self.current_idx])
            self.current_idx += 1
            self.is_waiting_to_sample = False

        elif self.current_idx == self.num_phoneme_ids:
            # force the BOS token after the global advance section
            self.ids.append(self.tokenizer.BOS_id)
            self.current_idx += 1
            self.is_waiting_to_sample = False

        elif self.current_idx == self.num_ids:
            if self.ids[-1] in [self.tokenizer.BOS_id, self.tokenizer.EOP_id]:
                if self.current_phoneme_idx < self.num_phoneme_ids - 1:
                    # force next phoneme
                    self.current_phoneme_idx += 1
                    self.ids.append(self.phoneme_ids[self.current_phoneme_idx])
                    self.current_idx += 1
                    self.is_waiting_to_sample = True
                    if hasattr(self, "pbar"):
                        # Update progress bar
                        self.pbar.update(1)
                else:
                    # force the EOS token
                    self.ids.append(self.tokenizer.EOS_id)
                    self.current_idx += 1
                    self.is_waiting_to_sample = False
                    self.is_finished = True
                    if hasattr(self, "pbar"):
                        # Close progress bar when finished
                        self.pbar.close()
            else:
                if (
                    self.num_acoustic_tokens_per_phoneme[self.current_phoneme_idx]
                    < self.max_acoustic_tokens_per_phoneme
                ):
                    # CONSUME THE SAMPLED ID HERE
                    self.ids.append(sampled_id)
                    self.current_idx += 1
                    self.num_acoustic_tokens_per_phoneme[self.current_phoneme_idx] += 1
                    self.is_waiting_to_sample = True
                else:
                    # force the EOP token
                    self.ids.append(self.tokenizer.EOP_id)
                    self.current_idx += 1
                    self.num_acoustic_tokens_per_phoneme[self.current_phoneme_idx] = 0
                    self.is_waiting_to_sample = False
        else:
            raise ValueError("Invalid state. Something went wrong.")


def generate(
    phoneme_tokens_lists: List[List[str]],
    model: ELLAVGARModel,
    top_p: float = 0.8,
    temperature: float = 1.0,
    max_phoneme_duration: int = 0.4,
    show_progress: bool = False,
):
    assert isinstance(phoneme_tokens_lists, list), "phoneme_tokens_lists must be a list"
    assert all(
        isinstance(phoneme_list, list) for phoneme_list in phoneme_tokens_lists
    ), "phoneme_tokens_lists must be a list of lists"
    assert all(
        isinstance(phoneme, str)
        for phoneme_list in phoneme_tokens_lists
        for phoneme in phoneme_list
    ), "phoneme_tokens_lists must be a list of lists of strings"

    phoneme_ids_lists: List[List[int]] = [
        model.tokenizer.encode_infer(phoneme_list)
        for phoneme_list in phoneme_tokens_lists
    ]

    max_acoustic_tokens_per_phoneme = int(
        max_phoneme_duration * model.tokenizer.phoneme_token_rate
    )

    handlers = [
        ELLAVInferenceHandler(
            phoneme_ids,
            model,
            top_p=top_p,
            max_acoustic_tokens_per_phoneme=max_acoustic_tokens_per_phoneme,
            show_progress=show_progress,
        )
        for phoneme_ids in phoneme_ids_lists
    ]

    while not all(handler.is_finished for handler in handlers):
        # only consider unfinished handlers
        unfinished_handlers = [
            handler for handler in handlers if not handler.is_finished
        ]

        # handle all deterministic steps until we need to sample a token for each unfinished handler
        for handler in unfinished_handlers:
            while not handler.is_waiting_to_sample:
                handler.step()

        # now forward the ids from each unfinished handler to the model and sample
        input_ids = [handler.ids for handler in unfinished_handlers]
        input_ids_tensor = torch.tensor(
            sum(input_ids, []), device=model.device, dtype=torch.long
        )
        seq_lens = [len(ids) for ids in input_ids]
        logits = model.forward(
            src_ids=input_ids_tensor,
            q_seqlen=seq_lens,
            return_last=True,
        )  # (B, V)
        logits[..., model.inference_mask] = float("-inf")  # zero out some tokens
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)  # (B, V)
            next_ids = sample_top_p(probs, top_p)
        else:
            next_ids = torch.argmax(logits, dim=-1).unsqueeze(0)

        # now take a step and provide the sampled ids to the handlers
        for handler, id in zip(unfinished_handlers, next_ids):
            handler.step(id.item())

    generated_ids_lists: List[List[int]] = [handler.ids for handler in handlers]

    generated_codec_ids = [model.tokenizer.decode(ids) for ids in generated_ids_lists]
    return generated_codec_ids


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Source: https://github.com/mistralai/mistral-inference"""

    assert 0 <= p <= 1

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)
