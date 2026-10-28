"""LLM2Vec text encoder (Meta-Llama-3-8B-Instruct with the LLM2Vec MNTP and supervised adapters) for captions and
joint descriptions.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


class LLM2VecEncoder:
    """LLM2Vec = base decoder LLM + MNTP LoRA + (optionally) a supervised LoRA. Captions and joint descriptions
    are encoded as documents, i.e. without an instruction.
    """

    def __init__(self,
                 base: str = "meta-llama/Meta-Llama-3-8B-Instruct",
                 mntp: str = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
                 supervised: str | None =
                 "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
                 device: str = "cuda", dtype=torch.bfloat16, max_length: int = 64):
        try:
            import torch.distributed.tensor
        except Exception:
            pass
        from llm2vec import LLM2Vec

        self.l2v = LLM2Vec.from_pretrained(
            mntp,
            peft_model_name_or_path=supervised,
            device_map=device,
            torch_dtype=dtype,
            pooling_mode="mean",
            max_length=max_length,
        )
        self.tok = self.l2v.tokenizer
        self.dim = int(self.l2v.model.config.hidden_size)
        self.max_length = int(max_length)

    @torch.no_grad()
    def encode(self, texts: Sequence[str], bs: int = 64) -> np.ndarray:
        """Mean-pooled embeddings [N, dim] float32."""
        e = self.l2v.encode(list(texts), batch_size=bs)
        e = e.float().cpu().numpy() if torch.is_tensor(e) else np.asarray(e, np.float32)
        return e.astype(np.float32)

    @torch.no_grad()
    def encode_tokens(self, texts: Sequence[str], bs: int = 1):
        """Per-token hidden states of each string's sentence span, packed to the front, and their mask:
        H [N, max_length, dim] float32, M [N, max_length] bool.

        Strings take the library's document path (`_convert_to_str("", text)`), so the mask marks the sentence tokens and
        excludes the chat template. Keep bs=1: llm2vec pads on the left and Llama's position ids do not discount the
        padding, so in a batch a string's states depend on its neighbours.
        """
        from llm2vec.llm2vec import batch_to_device
        dev = next(self.l2v.model.parameters()).device
        L, D = self.max_length, self.dim
        H = np.zeros((len(texts), L, D), dtype=np.float32)
        M = np.zeros((len(texts), L), dtype=bool)

        for s in range(0, len(texts), bs):
            chunk = [self.l2v.prepare_for_tokenization(self.l2v._convert_to_str("", t))
                     for t in texts[s:s + bs]]
            feat = batch_to_device(self.l2v.tokenize(chunk), dev)
            embed_mask = feat.pop("embed_mask")
            h = self.l2v.model(**feat).last_hidden_state.float().cpu()
            emb_b = embed_mask.bool().cpu()
            att_b = feat["attention_mask"].bool().cpu()
            if bool((emb_b & ~att_b).any()):
                raise RuntimeError(
                    "encode_tokens: embed_mask marks padded positions — llm2vec's "
                    "tokenize() contract is violated for this batch; refusing to pool")
            pool_mask = emb_b
            ids = feat["input_ids"].cpu()
            for j in range(h.shape[0]):
                mask_j = pool_mask[j]
                doc_ids = self.tok(texts[s + j], add_special_tokens=False,
                                   return_tensors=None)["input_ids"]
                eot = self.tok.convert_tokens_to_ids("<|eot_id|>")
                expect = doc_ids + [eot]
                got_ids = ids[j][mask_j].tolist()
                if got_ids != expect[:len(got_ids)]:
                    raise RuntimeError(
                        f"encode_tokens: masked ids are not a prefix of "
                        f"document+EOT for {texts[s + j][:80]!r}; the span is "
                        f"misaligned, not merely truncated")
                if len(got_ids) < len(expect) and ids.shape[1] < self.max_length:
                    raise RuntimeError(
                        f"encode_tokens: document lost tokens "
                        f"({len(got_ids)}/{len(expect)}) without hitting the "
                        f"tokenizer cap {self.max_length} — a non-truncation "
                        f"defect for {texts[s + j][:80]!r}")
                nz = mask_j.nonzero().flatten()
                if len(nz) and (int(nz[-1]) - int(nz[0]) + 1) != len(nz):
                    raise RuntimeError(
                        f"encode_tokens: non-contiguous document mask for "
                        f"{texts[s + j][:80]!r}")
                sel = h[j][mask_j]
                n = sel.shape[0]
                if n > L:
                    raise RuntimeError(
                        f"encode_tokens: {n} document tokens exceed the return "
                        f"buffer L={L}; impossible under one tokenizer cap — "
                        f"caller passed inconsistent max_length")
                if n:
                    H[s + j, :n] = sel.numpy()
                    M[s + j, :n] = True
        return H, M

    @torch.no_grad()
    def encoded_token_lengths(self, texts: Sequence[str]) -> np.ndarray:
        """Sentence-span token count per string, without running the model."""
        out = np.zeros(len(texts), dtype=np.int64)
        for i, t in enumerate(texts):
            wrapped = self.l2v.prepare_for_tokenization(self.l2v._convert_to_str("", t))
            feat = self.l2v.tokenize([wrapped])
            out[i] = int((feat["embed_mask"].bool() & feat["attention_mask"].bool()).sum())
        return out
