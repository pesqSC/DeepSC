import re
import json

import sentencepiece as spm

import torch
import torch.nn.functional as F

from models.tx_model import Transmitter
from utils.model_utils import (
    power_normalize,
    subsequent_mask,
    send_through_channel,
    normalize_text,
)

class SeqtoTextBPE:
    """
    Convert token IDs back into readable text.

    Supports both the old word-level vocabulary and SentencePiece BPE.

    For BPE, pass:
        tokenizer_model="data/train/europarl_bpe/tokenizer_bpe.model"
    """

    DEFAULT_SKIP_TOKENS = {
        "<PAD>",
        "<START>",
        "<EN>",
        "<PT>",
        "<ES>",
        "<FR>",
    }

    def __init__(
        self,
        vocb_dictionary,
        end_idx,
        skip_tokens=None,
        unk_token="<UNK>",
        tokenizer_model=None,
    ):
        if not isinstance(vocb_dictionary, dict):
            raise TypeError(
                "vocb_dictionary must be a dictionary."
            )

        self.reverse_word_map = {
            int(idx): token
            for token, idx in vocb_dictionary.items()
        }

        self.end_idx = int(end_idx)

        if skip_tokens is None:
            self.skip_tokens = set(self.DEFAULT_SKIP_TOKENS)
        else:
            self.skip_tokens = set(skip_tokens)

        self.unk_token = unk_token
        self.tokenizer = None

        if tokenizer_model is not None:
            try:
                import sentencepiece as spm
            except ImportError as exc:
                raise ImportError(
                    "SentencePiece is required for BPE decoding. "
                    "Install it with: uv add sentencepiece"
                ) from exc

            self.tokenizer = spm.SentencePieceProcessor(
                model_file=tokenizer_model
            )

            if self.tokenizer.eos_id() != self.end_idx:
                raise ValueError(
                    "Tokenizer <END> ID does not match end_idx: "
                    f"{self.tokenizer.eos_id()} != {self.end_idx}"
                )

        self.skip_ids = {
            int(idx)
            for token, idx in vocb_dictionary.items()
            if token in self.skip_tokens
        }
        self.unk_idx = vocb_dictionary.get(self.unk_token)
        if self.unk_idx is not None:
            self.unk_idx = int(self.unk_idx)

    def _clean_ids(self, list_of_indices, keep_unknown=True):
        cleaned = []

        for idx in list_of_indices:
            if hasattr(idx, "item"):
                idx = idx.item()

            idx = int(idx)

            if idx == self.end_idx:
                break

            if idx in self.skip_ids:
                continue

            if idx not in self.reverse_word_map:
                if keep_unknown and self.unk_idx is not None:
                    cleaned.append(self.unk_idx)
                continue

            cleaned.append(idx)

        return cleaned

    def sequence_to_tokens(
        self,
        list_of_indices,
        keep_unknown=True,
    ):
        cleaned_ids = self._clean_ids(
            list_of_indices,
            keep_unknown=keep_unknown,
        )

        return [
            self.reverse_word_map.get(idx, self.unk_token)
            for idx in cleaned_ids
        ]

    def sequence_to_text(
        self,
        list_of_indices,
        keep_unknown=True,
    ):
        cleaned_ids = self._clean_ids(
            list_of_indices,
            keep_unknown=keep_unknown,
        )

        # Correct path for SentencePiece BPE.
        if self.tokenizer is not None:
            return self.tokenizer.decode(cleaned_ids).strip()

        # Backward-compatible word-level path.
        tokens = [
            self.reverse_word_map.get(idx, self.unk_token)
            for idx in cleaned_ids
        ]

        text = " ".join(tokens)

        text = re.sub(
            r"\s+([.,!?;:%])",
            r"\1",
            text,
        )
        text = re.sub(
            r"([\(\[\{])\s+",
            r"\1",
            text,
        )
        text = re.sub(
            r"\s+([\)\]\}])",
            r"\1",
            text,
        )
        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        return text


def text_to_indices_bpe(text, sp, token_to_idx, max_length, lang_token="<EN>"):
    """
    Convert text to indices for BPE.

    Args:
        text: The text to convert.
        sp: The SentencePiece processor.
        token_to_idx: The token to index mapping.
        max_length: The maximum length of the sequence.
        lang_token: The language token.
    """
    text = normalize_text(text)

    content_ids = sp.encode(
        text,
        out_type=int,
        add_bos=False,
        add_eos=False,
    )

    # <START> + <LANG> + content + <END>
    max_content = max_length - 3
    if len(content_ids) > max_content:
        print(
            f"[Warning] {len(content_ids)} content tokens -> "
            f"truncated to {max_content}"
        )
        content_ids = content_ids[:max_content]

    seq = [
        token_to_idx["<START>"],
        token_to_idx[lang_token],
        *content_ids,
        token_to_idx["<END>"],
    ]

    return text, seq

def show_sequence(name, ids, sp, pad_idx):
    """
    Show the sequence of IDs and pieces.
    """
    ids = [int(x) for x in ids if int(x) != pad_idx]
    pieces = [sp.id_to_piece(x) for x in ids]
    print("\n" + name)
    print("IDs    :", ids)
    print("Pieces :", pieces)


def load_bpe(vocab_path, tokenizer_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    token_to_idx = vocab["token_to_idx"]
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)

    expected = {
        "<PAD>": 0,
        "<START>": 1,
        "<END>": 2,
        "<UNK>": 3,
        "<EN>": 4,
        "<PT>": 5,
        "<ES>": 6,
        "<FR>": 7,
    }

    print("=" * 70)
    print("BPE CHECK")
    print("=" * 70)
    print("SentencePiece vocab :", sp.get_piece_size())
    print("JSON vocab          :", len(token_to_idx))

    if sp.get_piece_size() != len(token_to_idx):
        raise ValueError("SentencePiece and JSON vocabulary sizes differ.")

    for token, expected_id in expected.items():
        json_id = token_to_idx[token]
        sp_id = sp.piece_to_id(token)
        print(f"{token:<8} JSON={json_id:<5} SP={sp_id:<5}")
        if json_id != expected_id or sp_id != expected_id:
            raise ValueError(f"Wrong ID for {token}")

    return token_to_idx, sp

@torch.no_grad()
def transmitter(src: torch.Tensor,src_mask: torch.Tensor, model: Transmitter, padding_idx: int, device: torch.device)-> torch.Tensor:
    """
    Transmit the source signal through the model.

    Args:
        src: The source signal with shape [batch_size, seq_len].
        src_mask: The source mask with shape [batch_size, 1, seq_len].
        model: The model to transmit the signal.
        padding_idx: The padding index.
        device: The device to transmit the signal on.
    """
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = power_normalize(channel_enc_output)
    return Tx_sig


@torch.no_grad()
def greedy_decode(
    Rx_sig,
    model,
    src,
    src_mask,
    max_len,
    start_idx,
    padding_idx,
    end_idx,
    target_language_idx,
    channel,
    snr,
    device,
):
    """
    Greedy decoding para DeepSC + BPE.

    A sequência é iniciada como:

        <START> <LANG>

    Exemplo EN -> PT:

        <START> <PT> ...

    Parameters
    ----------
    model:
        Modelo DeepSC completo.

    src:
        Source token IDs.
        Shape: [batch, src_len]

    n_var:
        Noise std.

    max_len:
        Comprimento máximo TOTAL da sequência gerada,
        incluindo:
            <START>
            <LANG>
            conteúdo
            <END>

    padding_idx:
        ID de <PAD>.

    start_symbol:
        ID de <START>.

    end_symbol:
        ID de <END>.

    target_language_symbol:
        ID da língua desejada.
        Ex:
            <EN> para reconstrução EN
            <PT> para EN -> PT

    channel:
        "AWGN", "Rayleigh" ou "Rician".

    snr:
        Signal-to-noise ratio in dB.

    device:
        Device to run the model on.
    """
    batch_size = Rx_sig.size(0)
    # Rx_sig = send_through_channel(tx_sig, channel, snr)

    # CHANNEL DECODER
    memory = model.channel_decoder(Rx_sig)

    # <START> <LANG>
    start_tokens = torch.full(
        (batch_size, 1),
        start_idx,
        dtype=src.dtype,
        device=device,
    )

    language_tokens = torch.full(
        (batch_size, 1),
        target_language_idx,
        dtype=src.dtype,
        device=device,
    )

    outputs = torch.cat(
        [
            start_tokens,
            language_tokens,
        ],
        dim=1,
    )

    # Marca exemplos que já produziram <END>
    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=device,
    )

    # Já temos dois tokens:
    #
    # <START> <LANG>
    #
    for _ in range(max_len - 2):

        # Look-ahead mask
        trg_len = outputs.size(1)

        look_ahead_mask = subsequent_mask(
            trg_len
        ).type(torch.float32).to(device)

        # Durante geração não deveria existir PAD dentro de outputs,
        # mas mantemos a máscara por segurança.
        trg_padding_mask = (
            (outputs == padding_idx)
            .unsqueeze(-2)
            .float()
        )

        combined_mask = torch.maximum(
            trg_padding_mask,
            look_ahead_mask,
        )

        # DECODER
        #
        # IMPORTANTE:
        # src_mask entra no cross-attention.
        dec_output = model.decoder(
            outputs,
            memory,
            combined_mask,
            src_mask,
        )

        # Output projection
        logits = model.dense(
            dec_output
        )

        # Somente o último passo
        next_token_logits = logits[:, -1, :]

        next_word = torch.argmax(
            next_token_logits,
            dim=-1,
        )

        # Quem já terminou continua produzindo END
        next_word = torch.where(
            finished,
            torch.full_like(
                next_word,
                end_idx,
            ),
            next_word,
        )

        outputs = torch.cat([outputs, next_word.unsqueeze(1)], dim=1)

        # Atualiza estado finished
        finished = finished | (
            next_word == end_idx
        )

        # Todos terminaram
        if finished.all():
            break

    return outputs


@torch.no_grad()
def beam_decode(
    Rx_sig,
    model,
    src,
    src_mask,
    max_len,
    start_idx,
    padding_idx,
    end_idx,
    target_language_idx,
    channel,
    snr,
    device,
    beam_size=5,
    length_penalty=0.7,
    forbidden_token_ids=None,
    return_all_beams=False,
):
    """
    Beam-search decoding for DeepSC + BPE.

    Initial sequence:

        <START> <LANG>

    Example EN -> PT:

        <START> <PT> ...

    Parameters
    ----------
    Rx_sig:
        Channel-encoder output.
        Shape usually:
            [batch, src_len, channel_dim]

    model:
        DeepSC receiver / complete model containing:
            model.channel_decoder
            model.decoder
            model.dense

    src:
        Source token IDs.
        Shape:
            [batch, src_len]

    src_mask:
        Source padding mask used by decoder cross-attention.

    max_len:
        Maximum TOTAL sequence length, including:
            <START>
            <LANG>
            content
            <END>

    start_idx:
        ID of <START>.

    padding_idx:
        ID of <PAD>.

    end_idx:
        ID of <END>.

    target_language_idx:
        Language control token.
        Examples:
            <EN>
            <PT>

    channel:
        "AWGN", "Rayleigh", "Rician", etc.

    snr:
        SNR in dB.

    device:
        torch device.

    beam_size:
        Number of beams.

    length_penalty:
        Controls preference for longer sequences.

        0.0:
            no length normalization

        ~0.6-1.0:
            usually reasonable for translation

    forbidden_token_ids:
        Optional iterable of token IDs that cannot be generated.

        Example:
            [
                padding_idx,
                start_idx,
                en_idx,
                pt_idx,
                es_idx,
                fr_idx,
            ]

        Do NOT include end_idx.

    return_all_beams:
        If False:
            returns best sequence [B, L]

        If True:
            returns:
                best_sequences,
                all_beams,
                normalized_scores

    Returns
    -------
    best_sequences:
        Tensor [batch, generated_length]
    """

    model.eval()

    batch_size = Rx_sig.size(0)

    if beam_size < 1:
        raise ValueError("beam_size must be >= 1.")

    if max_len < 3:
        raise ValueError(
            "max_len must allow at least "
            "<START> <LANG> <END>."
        )

    # [B, src_len, d_model]
    memory = model.channel_decoder(
        Rx_sig
    )

    # =====================================================
    # 2. INITIAL PREFIX
    #
    # <START> <LANG>
    # =====================================================

    prefix = torch.tensor(
        [
            start_idx,
            target_language_idx,
        ],
        dtype=src.dtype,
        device=device,
    )

    # [B, beam, 2]
    sequences = (
        prefix
        .view(1, 1, 2)
        .expand(
            batch_size,
            beam_size,
            2,
        )
        .clone()
    )

    # =====================================================
    # 3. BEAM SCORES
    # =====================================================
    #
    # At the beginning only beam 0 is valid.
    #
    # Beam 0 -> score = 0
    # Others -> -inf
    # =====================================================

    beam_scores = torch.full(
        (batch_size, beam_size),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )

    beam_scores[:, 0] = 0.0

    # Whether each beam has already produced <END>
    finished = torch.zeros(
        batch_size,
        beam_size,
        dtype=torch.bool,
        device=device,
    )

    # Store the sequence length where EOS was first emitted.
    #
    # Default = max_len for beams that never emit EOS.
    sequence_lengths = torch.full(
        (batch_size, beam_size),
        max_len,
        dtype=torch.long,
        device=device,
    )

    # =====================================================
    # 4. EXPAND ENCODER MEMORY TO BEAMS
    # =====================================================

    # Original:
    # [B, src_len, d_model]
    #
    # Expanded:
    # [B*beam, src_len, d_model]

    memory_beam = memory.repeat_interleave(
        beam_size,
        dim=0,
    )

    # src_mask can usually be:
    #
    # [B, 1, src_len]
    #
    # or
    #
    # [B, 1, 1, src_len]
    #
    # repeat_interleave on dim 0 works for either.

    if src_mask is not None:
        src_mask_beam = (
            src_mask.repeat_interleave(
                beam_size,
                dim=0,
            )
        )
    else:
        src_mask_beam = None

    # =====================================================
    # 5. FORBIDDEN TOKENS
    # =====================================================

    if forbidden_token_ids is None:
        forbidden_token_ids = []

    forbidden_token_ids = set(
        int(x)
        for x in forbidden_token_ids
    )

    # Never forbid EOS accidentally.
    forbidden_token_ids.discard(
        end_idx
    )

    # =====================================================
    # 6. AUTOREGRESSIVE BEAM SEARCH
    # =====================================================

    for step in range(max_len - 2):

        current_len = sequences.size(-1)

        # -------------------------------------------------
        # Flatten beam dimension
        #
        # [B, beam, L]
        #       ->
        # [B*beam, L]
        # -------------------------------------------------

        flat_sequences = sequences.reshape(
            batch_size * beam_size,
            current_len,
        )

        # -------------------------------------------------
        # TARGET MASK
        # -------------------------------------------------

        look_ahead_mask = subsequent_mask(
            current_len
        ).to(
            device=device,
            dtype=torch.float32,
        )

        trg_padding_mask = (
            (flat_sequences == padding_idx)
            .unsqueeze(-2)
            .float()
        )

        combined_mask = torch.maximum(
            trg_padding_mask,
            look_ahead_mask,
        )

        # -------------------------------------------------
        # DECODER
        # -------------------------------------------------

        dec_output = model.decoder(
            flat_sequences,
            memory_beam,
            combined_mask,
            src_mask_beam,
        )

        # [B*beam, L, vocab]
        logits = model.dense(
            dec_output
        )

        # Last autoregressive position only
        #
        # [B*beam, vocab]
        next_token_logits = logits[
            :,
            -1,
            :,
        ]

        vocab_size = (
            next_token_logits.size(-1)
        )

        # -------------------------------------------------
        # Block invalid/special tokens if requested
        # -------------------------------------------------

        if forbidden_token_ids:

            valid_forbidden = [
                token_id
                for token_id
                in forbidden_token_ids
                if 0 <= token_id < vocab_size
            ]

            if valid_forbidden:

                next_token_logits[
                    :,
                    valid_forbidden,
                ] = float("-inf")

        # -------------------------------------------------
        # Convert to log probabilities
        # -------------------------------------------------

        log_probs = F.log_softmax(
            next_token_logits,
            dim=-1,
        )

        # Restore:
        #
        # [B, beam, vocab]
        log_probs = log_probs.view(
            batch_size,
            beam_size,
            vocab_size,
        )

        # =================================================
        # FINISHED BEAMS
        # =================================================
        #
        # Once a beam has produced END, force it to keep
        # producing END without changing its score.
        #
        # This keeps tensor shapes simple.
        # =================================================

        if finished.any():

            log_probs = log_probs.masked_fill(
                finished.unsqueeze(-1),
                float("-inf"),
            )

            # END gets log-probability 0 for completed beams.
            end_scores = log_probs[
                :,
                :,
                end_idx,
            ]

            end_scores = torch.where(
                finished,
                torch.zeros_like(
                    end_scores
                ),
                end_scores,
            )

            log_probs[
                :,
                :,
                end_idx,
            ] = end_scores

        # =================================================
        # COMBINE OLD BEAM SCORES + NEW TOKEN SCORES
        # =================================================

        candidate_scores = (
            beam_scores.unsqueeze(-1)
            +
            log_probs
        )

        # [B, beam * vocab]
        candidate_scores = (
            candidate_scores.view(
                batch_size,
                -1,
            )
        )

        # =================================================
        # SELECT TOP-K CANDIDATES
        # =================================================

        top_scores, top_indices = torch.topk(
            candidate_scores,
            k=beam_size,
            dim=-1,
        )

        # Determine:
        #
        # which previous beam?
        # which vocabulary token?
        #

        source_beam = (
            top_indices // vocab_size
        )

        next_tokens = (
            top_indices % vocab_size
        )

        # =================================================
        # GATHER PREVIOUS SEQUENCES
        # =================================================

        gather_index = (
            source_beam
            .unsqueeze(-1)
            .expand(
                -1,
                -1,
                current_len,
            )
        )

        selected_sequences = torch.gather(
            sequences,
            dim=1,
            index=gather_index,
        )

        # Append new tokens
        sequences = torch.cat(
            [
                selected_sequences,
                next_tokens.unsqueeze(-1),
            ],
            dim=-1,
        )

        # =================================================
        # UPDATE FINISHED STATUS
        # =================================================

        previous_finished = torch.gather(
            finished,
            dim=1,
            index=source_beam,
        )

        previous_lengths = torch.gather(
            sequence_lengths,
            dim=1,
            index=source_beam,
        )

        just_finished = (
            (~previous_finished)
            &
            (next_tokens == end_idx)
        )

        new_length = sequences.size(-1)

        sequence_lengths = torch.where(
            just_finished,
            torch.full_like(
                previous_lengths,
                new_length,
            ),
            previous_lengths,
        )

        finished = (
            previous_finished
            |
            (next_tokens == end_idx)
        )

        beam_scores = top_scores

        # =================================================
        # STOP IF ALL BEAMS FOR ALL SAMPLES FINISHED
        # =================================================

        if finished.all():
            break

    # =====================================================
    # 7. FINAL LENGTH NORMALIZATION
    # =====================================================

    actual_length = sequences.size(-1)

    # Beams that never generated EOS use actual generated len.
    effective_lengths = torch.where(
        finished,
        sequence_lengths,
        torch.full_like(
            sequence_lengths,
            actual_length,
        ),
    )

    if length_penalty > 0:

        # GNMT-style length penalty
        #
        # lp = ((5 + length) / 6)^alpha

        length_norm = (
            (
                5.0
                +
                effective_lengths.float()
            )
            / 6.0
        ).pow(
            length_penalty
        )

        normalized_scores = (
            beam_scores
            /
            length_norm
        )

    else:

        normalized_scores = (
            beam_scores
        )

    # =====================================================
    # 8. SELECT BEST BEAM FOR EACH BATCH ITEM
    # =====================================================

    best_beam_idx = torch.argmax(
        normalized_scores,
        dim=-1,
    )

    batch_indices = torch.arange(
        batch_size,
        device=device,
    )

    best_sequences = sequences[
        batch_indices,
        best_beam_idx,
    ]

    # =====================================================
    # 9. OPTIONAL ALL-BEAM OUTPUT
    # =====================================================

    if return_all_beams:
        return (
            best_sequences,
            sequences,
            normalized_scores,
        )

    return best_sequences