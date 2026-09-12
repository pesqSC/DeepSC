import re
import json

import sentencepiece as spm

import torch
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
    tx_sig,
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
    batch_size = tx_sig.size(0)
    Rx_sig = send_through_channel(tx_sig, channel, snr)

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
    max_len,
    start_symbol,
    language_symbol,
    end_symbol,
    padding_idx,
    device,
    beam_size=5,
    length_penalty=0.7,
    idx_to_token=None,
    debug=False,
)-> torch.Tensor:
    """
    Beam-search decoding for Student / LoRA receiver.

    Prefix:
        <START> <LANGUAGE>

    Example EN -> PT:
        <START> <PT>

    Parameters
    ----------
    beam_size : int
        Number of candidate sequences retained at each step.

    length_penalty : float
        Penalizes overly short sequences.
        0.0 = no length normalization.
        Typical values: 0.6 - 1.0.

    debug : bool
        Print current beam candidates.
    """

    # Channel decoding
    memory = model.channel_decoder(Rx_sig)

    batch_size = src.size(0)

    if batch_size != 1:
        raise ValueError(
            "This beam-search implementation currently "
            "supports batch_size=1."
        )

    # Initial prefix:
    # <START> <LANGUAGE>
    initial_sequence = torch.tensor(
        [[start_symbol, language_symbol]],
        dtype=src.dtype,
        device=device,
    )

    # Each beam:
    # {
    #   "tokens": Tensor [1, T],
    #   "score": cumulative log probability,
    #   "finished": bool
    # }
    beams = [
        {
            "tokens": initial_sequence,
            "score": 0.0,
            "finished": False,
        }
    ]

    # Length-normalized score
    def normalized_score(beam):
        tokens = beam["tokens"]

        # Ignore <START> and language token
        generated_length = max(
            tokens.size(1) - 2,
            1,
        )

        if length_penalty <= 0:
            return beam["score"]

        penalty = (
            (5.0 + generated_length) / 6.0
        ) ** length_penalty

        return beam["score"] / penalty

    # Beam-search loop
    for step in range(max_len - 2):

        candidates = []

        for beam in beams:

            # Finished beams are kept as-is
            if beam["finished"]:
                candidates.append(beam)
                continue

            outputs = beam["tokens"]

            # Target padding mask
            trg_mask = (
                outputs == padding_idx
            ).unsqueeze(-2).float().to(device)

            # Causal mask
            look_ahead_mask = subsequent_mask(
                outputs.size(1)
            ).float().to(device)

            combined_mask = torch.max(
                trg_mask,
                look_ahead_mask,
            )

            # Decoder
            dec_output = model.decoder(
                outputs,
                memory,
                combined_mask,
                None,
            )

            logits = model.dense(dec_output)

            # Only final position matters
            next_logits = logits[:, -1, :]

            log_probs = torch.log_softmax(
                next_logits,
                dim=-1,
            )

            # Top-K next tokens
            top_log_probs, top_ids = torch.topk(
                log_probs[0],
                k=beam_size,
            )

            for log_prob, token_id in zip(
                top_log_probs.tolist(),
                top_ids.tolist(),
            ):

                next_token = torch.tensor(
                    [[token_id]],
                    dtype=src.dtype,
                    device=device,
                )

                new_tokens = torch.cat(
                    [
                        outputs,
                        next_token,
                    ],
                    dim=1,
                )

                candidates.append(
                    {
                        "tokens": new_tokens,
                        "score": (
                            beam["score"]
                            + float(log_prob)
                        ),
                        "finished": (
                            token_id == end_symbol
                        ),
                    }
                )

        # Rank candidates
        candidates.sort(
            key=normalized_score,
            reverse=True,
        )

        beams = candidates[:beam_size]

        # Optional debugging
        if debug:

            print(
                f"\n{'=' * 60}"
            )
            print(
                f"Beam step {step + 1}"
            )
            print(
                f"{'=' * 60}"
            )

            for rank, beam in enumerate(
                beams,
                start=1,
            ):

                tokens = (
                    beam["tokens"][0]
                    .detach()
                    .cpu()
                    .tolist()
                )

                if idx_to_token is not None:
                    words = [
                        idx_to_token.get(
                            int(token),
                            "<UNKNOWN>",
                        )
                        for token in tokens
                    ]

                    text = " ".join(words)

                else:
                    text = str(tokens)

                print(
                    f"{rank:2d}. "
                    f"score={beam['score']:.4f} "
                    f"norm={normalized_score(beam):.4f} "
                    f"finished={beam['finished']}"
                )

                print(
                    f"    {text}"
                )

            # Stop when all beams finished
            if all(
                beam["finished"]
                for beam in beams
            ):
                break

        # Choose best beam
        finished_beams = [
            beam
            for beam in beams
            if beam["finished"]
        ]

        if finished_beams:
            best_beam = max(
                finished_beams,
                key=normalized_score,
            )
        else:
            best_beam = max(
                beams,
                key=normalized_score,
            )

        return best_beam["tokens"]