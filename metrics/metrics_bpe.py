"""
metrics_bpe.py

Evaluation metrics for DeepSC BPE outputs.

Metrics:
- BLEU
- chrF
- Semantic similarity (multilingual SentenceTransformer cosine similarity)
- ROUGE-L F1
- Exact-match rate
- BPE token accuracy

Recommended packages:
    uv add sacrebleu rouge-score sentence-transformers sentencepiece pandas numpy

For EN reconstruction:
    reference = original English sentence
    hypothesis = Teacher EN or Student EN output

For EN -> PT translation:
    reference = ground-truth Portuguese target
    hypothesis = Student PT output
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import sacrebleu
import sentencepiece as spm
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer


def normalize_text(text: str) -> str:
    """
    Normalize text before exact-match and metric evaluation.

    Keeps accents and Unicode intact.
    """
    text = unicodedata.normalize("NFKC", str(text))
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


@dataclass
class MetricResult:
    bleu: float
    chrf: float
    semantic_similarity: float
    rouge_l: float
    exact_match_rate: float
    token_accuracy: float

    def as_dict(self) -> dict:
        return {
            "bleu": self.bleu,
            "chrf": self.chrf,
            "semantic_similarity": self.semantic_similarity,
            "rouge_l": self.rouge_l,
            "exact_match_rate": self.exact_match_rate,
            "token_accuracy": self.token_accuracy,
        }


class DeepSCMetrics:
    """
    Metric evaluator for DeepSC / SentencePiece BPE outputs.

    Parameters
    ----------
    sp:
        SentencePiece BPE model.

    semantic_model:
        Multilingual SentenceTransformer model.
        This model supports EN/PT/ES/FR much better than an English-only model.

    device:
        "cuda", "cpu", or None. None lets SentenceTransformer choose.
    """

    def __init__(
        self,
        sp,
        semantic_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device: Optional[str] = None,
    ):
        self.sp = sp
        self.semantic_model = SentenceTransformer(
            semantic_model,
            device=device,
        )

        self.rouge = rouge_scorer.RougeScorer(
            ["rougeL"],
            use_stemmer=True,
        )

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    @staticmethod
    def bleu(reference: str, hypothesis: str) -> float:
        """
        Sentence BLEU in [0, 100].
        """
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)

        return float(
            sacrebleu.sentence_bleu(
                hypothesis,
                [reference],
                smooth_method="exp",
            ).score
        )

    @staticmethod
    def chrf(reference: str, hypothesis: str) -> float:
        """
        Sentence chrF in [0, 100].
        """
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)

        return float(
            sacrebleu.sentence_chrf(
                hypothesis,
                [reference],
                word_order=2,  # chrF++
            ).score
        )

    def semantic_similarity(
        self,
        reference: str,
        hypothesis: str,
    ) -> float:
        """
        Cosine similarity between multilingual sentence embeddings.

        Usually approximately [-1, 1], although normal text is commonly > 0.
        """
        texts = [
            normalize_text(reference),
            normalize_text(hypothesis),
        ]

        embeddings = self.semantic_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Embeddings are normalized, so dot product == cosine similarity.
        return float(np.dot(embeddings[0], embeddings[1]))

    def rouge_l(
        self,
        reference: str,
        hypothesis: str,
    ) -> float:
        """
        ROUGE-L F1 in [0, 1].
        """
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)

        score = self.rouge.score(reference, hypothesis)["rougeL"]
        return float(score.fmeasure)

    @staticmethod
    def exact_match(
        reference: str,
        hypothesis: str,
    ) -> float:
        """
        1.0 if normalized strings are identical, otherwise 0.0.
        """
        return float(
            normalize_text(reference) == normalize_text(hypothesis)
        )

    def token_accuracy(
        self,
        reference: str,
        hypothesis: str,
    ) -> float:
        """
        Position-wise BPE token accuracy in [0, 1].

        Both decoded strings are re-encoded with the SAME SentencePiece model.
        Extra/missing tokens are penalized.

        Example:
            ref: [10, 20, 30]
            hyp: [10, 20, 40, 50]

            correct positions = 2
            denominator = 4
            accuracy = 0.5

        This is intentionally stricter than BLEU/chrF.
        """
        ref_ids = self.sp.encode(
            normalize_text(reference),
            out_type=int,
        )
        hyp_ids = self.sp.encode(
            normalize_text(hypothesis),
            out_type=int,
        )

        denom = max(len(ref_ids), len(hyp_ids))

        if denom == 0:
            return 1.0

        matches = sum(
            int(ref_id == hyp_id)
            for ref_id, hyp_id in zip(ref_ids, hyp_ids)
        )

        return float(matches / denom)

    # ------------------------------------------------------------------
    # One sentence
    # ------------------------------------------------------------------

    def evaluate_sentence(
        self,
        reference: str,
        hypothesis: str,
    ) -> MetricResult:
        return MetricResult(
            bleu=self.bleu(reference, hypothesis),
            chrf=self.chrf(reference, hypothesis),
            semantic_similarity=self.semantic_similarity(
                reference,
                hypothesis,
            ),
            rouge_l=self.rouge_l(reference, hypothesis),
            exact_match_rate=self.exact_match(
                reference,
                hypothesis,
            ),
            token_accuracy=self.token_accuracy(
                reference,
                hypothesis,
            ),
        )

    # ------------------------------------------------------------------
    # Corpus / dataset
    # ------------------------------------------------------------------

    def evaluate_corpus(
        self,
        references: Sequence[str],
        hypotheses: Sequence[str],
    ) -> MetricResult:
        """
        Evaluate an entire dataset.

        BLEU and chrF are calculated as CORPUS metrics.
        Semantic similarity, ROUGE-L, exact match and token accuracy
        are averaged across samples.
        """
        if len(references) != len(hypotheses):
            raise ValueError(
                "references and hypotheses must have the same length."
            )

        if len(references) == 0:
            raise ValueError("Cannot evaluate an empty corpus.")

        refs = [normalize_text(x) for x in references]
        hyps = [normalize_text(x) for x in hypotheses]

        # Proper corpus BLEU.
        bleu = float(
            sacrebleu.corpus_bleu(
                hyps,
                [refs],
            ).score
        )

        # Proper corpus chrF++.
        chrf = float(
            sacrebleu.corpus_chrf(
                hyps,
                [refs],
                word_order=2,
            ).score
        )

        # Encode all sentences in two batches instead of one-by-one.
        ref_emb = self.semantic_model.encode(
            refs,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        hyp_emb = self.semantic_model.encode(
            hyps,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        semantic_scores = np.sum(
            ref_emb * hyp_emb,
            axis=1,
        )

        rouge_scores = [
            self.rouge_l(ref, hyp)
            for ref, hyp in zip(refs, hyps)
        ]

        exact_scores = [
            self.exact_match(ref, hyp)
            for ref, hyp in zip(refs, hyps)
        ]

        token_scores = [
            self.token_accuracy(ref, hyp)
            for ref, hyp in zip(refs, hyps)
        ]

        return MetricResult(
            bleu=bleu,
            chrf=chrf,
            semantic_similarity=float(np.mean(semantic_scores)),
            rouge_l=float(np.mean(rouge_scores)),
            exact_match_rate=float(np.mean(exact_scores)),
            token_accuracy=float(np.mean(token_scores)),
        )

    # ------------------------------------------------------------------
    # Per-example dataframe
    # ------------------------------------------------------------------

    def evaluate_examples(
        self,
        references: Sequence[str],
        hypotheses: Sequence[str],
        snrs: Optional[Sequence[float]] = None,
        system_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Return one row per sentence.

        Useful for:
        - finding bad examples
        - saving JSON/CSV
        - plotting distributions
        - grouping by SNR
        """
        if len(references) != len(hypotheses):
            raise ValueError(
                "references and hypotheses must have the same length."
            )

        if snrs is not None and len(snrs) != len(references):
            raise ValueError(
                "snrs must have the same length as references."
            )

        rows = []

        for i, (ref, hyp) in enumerate(
            zip(references, hypotheses)
        ):
            metrics = self.evaluate_sentence(ref, hyp)

            row = {
                "sample_id": i,
                "reference": ref,
                "hypothesis": hyp,
                **metrics.as_dict(),
            }

            if snrs is not None:
                row["snr"] = float(snrs[i])

            if system_name is not None:
                row["system"] = system_name

            rows.append(row)

        return pd.DataFrame(rows)


def summarize_by_snr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average sentence-level metrics for every SNR.

    Expected columns:
        snr
        bleu
        chrf
        semantic_similarity
        rouge_l
        exact_match_rate
        token_accuracy
    """
    required = {
        "snr",
        "bleu",
        "chrf",
        "semantic_similarity",
        "rouge_l",
        "exact_match_rate",
        "token_accuracy",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    group_cols = ["snr"]

    if "system" in df.columns:
        group_cols.append("system")

    metric_cols = [
        "bleu",
        "chrf",
        "semantic_similarity",
        "rouge_l",
        "exact_match_rate",
        "token_accuracy",
    ]

    summary = (
        df.groupby(group_cols, as_index=False)[metric_cols]
        .mean()
        .sort_values(group_cols)
        .reset_index(drop=True)
    )

    return summary


if __name__ == "__main__":
    # --------------------------------------------------------------
    # Small example
    # --------------------------------------------------------------

    evaluator = DeepSCMetrics(
        tokenizer_model=(
            "./data/train/europarl_bpe/tokenizer_bpe.model"
        ),
        device="cuda",  # use "cpu" if needed
    )

    original_en = (
        "the european union is a political and economic union ."
    )

    teacher_en = (
        "the european union is a political and economic union ."
    )

    student_en = (
        "the european union is a political and economic union ."
    )

    print("\nTeacher EN")
    print(
        evaluator.evaluate_sentence(
            original_en,
            teacher_en,
        ).as_dict()
    )

    print("\nStudent EN")
    print(
        evaluator.evaluate_sentence(
            original_en,
            student_en,
        ).as_dict()
    )

    # --------------------------------------------------------------
    # IMPORTANT FOR PT
    # --------------------------------------------------------------
    #
    # Do NOT compare Student PT against original_en.
    #
    # Use the real Portuguese target:
    #
    # target_pt = "a união europeia é uma união política e económica ."
    # student_pt = "o parlamento europeu é um problema fundamental ."
    #
    # print(
    #     evaluator.evaluate_sentence(
    #         target_pt,
    #         student_pt,
    #     ).as_dict()
    # )
