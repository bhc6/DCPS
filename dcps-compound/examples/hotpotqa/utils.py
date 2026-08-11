"""HotpotQA reproduction utilities — strict paper replication.

Paper setup:
- Multi-hop program based on HoVerMultiHop (generate_query per hop, last hop answers).
- BM25 retrieval over Wikipedia 2017 abstracts.
- Per-predictor textual feedback identifying remaining documents to retrieve.
- 150 train / 300 val / 300 test.
"""

import random
import re
import string
from collections import Counter
from pathlib import Path

import bm25s
import dspy
import orjson
import Stemmer
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Normalisation helpers (from HotpotQA official evaluation script)
# ---------------------------------------------------------------------------


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction_str: str, ground_truth_str: str) -> float:
    """Token-level F1 between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction_str).split()
    gold_tokens = normalize_answer(ground_truth_str).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match_score(prediction_str: str, ground_truth_str: str) -> float:
    """Exact match after normalisation."""
    return float(normalize_answer(prediction_str) == normalize_answer(ground_truth_str))


# ---------------------------------------------------------------------------
# BM25 Retriever over Wikipedia 2017 abstracts
# ---------------------------------------------------------------------------

_retriever = None
_corpus = None
_stemmer = None


def init_retriever(corpus_path: str = "wiki.abstracts.2017.jsonl"):
    """Build (or re-use) a BM25 index over the Wikipedia abstracts corpus.

    Downloads the corpus automatically if not present.
    """
    global _retriever, _corpus, _stemmer

    if _retriever is not None:
        return

    corpus_file = Path(corpus_path)
    if not corpus_file.exists():
        from dspy.utils import download

        print("Downloading Wikipedia 2017 abstracts (~500 MB compressed)...")
        download("https://huggingface.co/dspy/cache/resolve/main/wiki.abstracts.2017.tar.gz")
        import tarfile

        with tarfile.open("wiki.abstracts.2017.tar.gz", "r:gz") as tar:
            tar.extractall()

    print("Loading corpus...")
    _corpus = []
    with open(corpus_file, "rb") as f:
        for line in f:
            item = orjson.loads(line)
            _corpus.append(f"{item['title']} | {' '.join(item['text'])}")
    print(f"Corpus loaded: {len(_corpus)} documents")

    print("Building BM25 index (2-3 min)...")
    _stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(_corpus, stopwords="en", stemmer=_stemmer)
    _retriever = bm25s.BM25(k1=0.9, b=0.4)
    _retriever.index(corpus_tokens)
    print("BM25 index ready.")


def search(query: str, k: int = 5) -> list[str]:
    """Search the Wikipedia corpus via BM25."""
    assert _retriever is not None, "Call init_retriever() first."
    tokens = bm25s.tokenize(query, stopwords="en", stemmer=_stemmer, show_progress=False)
    results, scores = _retriever.retrieve(tokens, k=k, n_threads=1, show_progress=False)
    return [_corpus[doc] for doc in results[0]]


# ---------------------------------------------------------------------------
# Multi-hop DSPy program (HoVerMultiHop-style, last hop answers)
# ---------------------------------------------------------------------------

NUM_HOPS = 3
NUM_DOCS = 5


class MultiHopQA(dspy.Module):
    """Multi-hop QA program following the HoVerMultiHop structure.

    - Hops 1..(N-1): generate a search query, retrieve docs, take notes.
    - Hop N (last): answer the question using accumulated notes + context.
    """

    def __init__(self, num_hops: int = NUM_HOPS, num_docs: int = NUM_DOCS):
        super().__init__()
        self.num_hops = num_hops
        self.num_docs = num_docs
        self.generate_query = dspy.ChainOfThought("question, notes -> query")
        self.append_notes = dspy.ChainOfThought(
            "question, notes, context -> new_notes: list[str]"
        )
        self.answer_question = dspy.ChainOfThought(
            "question, notes, context -> answer"
        )

    def forward(self, question: str):
        notes: list[str] = []
        context = ""

        for hop_idx in range(self.num_hops):
            query = self.generate_query(question=question, notes=notes).query
            retrieved = search(query, k=self.num_docs)
            context = "\n\n".join(retrieved)

            if hop_idx < self.num_hops - 1:
                prediction = self.append_notes(
                    question=question, notes=notes, context=context
                )
                notes.extend(prediction.new_notes)
            else:
                prediction = self.answer_question(
                    question=question, notes=notes, context=context
                )
                return dspy.Prediction(
                    answer=prediction.answer,
                    notes=notes,
                )

        return dspy.Prediction(answer="", notes=notes)


# ---------------------------------------------------------------------------
# GEPA feedback metric (per-predictor aware)
# ---------------------------------------------------------------------------


def hotpotqa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA-compatible metric with per-predictor feedback.

    When ``pred_name`` is provided, returns predictor-specific feedback
    identifying what documents remain to be retrieved at that stage.
    """
    gold_answer = gold.answer
    pred_answer = getattr(pred, "answer", "")

    f1 = f1_score(pred_answer, gold_answer)
    em = exact_match_score(pred_answer, gold_answer)

    # --- Per-predictor feedback (for GEPA reflection) ---
    if pred_name == "generate_query":
        if f1 >= 0.8:
            feedback = (
                f"Good query generation. The final answer '{pred_answer}' achieved F1={f1:.2f} "
                f"against gold '{gold_answer}'. The queries successfully guided retrieval."
            )
        else:
            feedback = (
                f"Query generation needs improvement. Final answer '{pred_answer}' got F1={f1:.2f} "
                f"against gold '{gold_answer}'. The generated queries may not have retrieved "
                f"the right supporting documents. Try formulating queries that target the "
                f"specific entities and relationships needed to answer: '{gold.question}'"
            )
        return dspy.Prediction(score=f1, feedback=feedback)

    if pred_name == "append_notes":
        if f1 >= 0.8:
            feedback = (
                f"Good note-taking. The accumulated notes helped reach F1={f1:.2f}. "
                f"Key information was captured from retrieved documents."
            )
        else:
            feedback = (
                f"Note-taking needs improvement. Final F1={f1:.2f}. "
                f"The notes may have missed critical information. For the question "
                f"'{gold.question}', ensure notes capture entities, dates, and "
                f"relationships that bridge multiple documents."
            )
        return dspy.Prediction(score=f1, feedback=feedback)

    if pred_name == "answer_question":
        if em == 1.0:
            feedback = (
                f"Correct! Answer '{pred_answer}' exactly matches '{gold_answer}'."
            )
        elif f1 > 0.5:
            feedback = (
                f"Partially correct. Answer '{pred_answer}' has F1={f1:.2f} against "
                f"'{gold_answer}'. Extract a more precise short answer from the context."
            )
        else:
            feedback = (
                f"Incorrect. Answer '{pred_answer}' has F1={f1:.2f}. "
                f"The correct answer is '{gold_answer}'. Focus on synthesizing "
                f"information from the notes and context to produce a short factual answer."
            )
        return dspy.Prediction(score=f1, feedback=feedback)

    # --- Program-level feedback (default) ---
    if em == 1.0:
        feedback = f"Correct! '{pred_answer}' matches '{gold_answer}'."
    elif f1 > 0.5:
        feedback = (
            f"Partially correct. '{pred_answer}' has F1={f1:.2f} against '{gold_answer}'. "
            f"Improve query generation to retrieve better supporting documents."
        )
    else:
        feedback = (
            f"Incorrect. '{pred_answer}' has F1={f1:.2f}. Correct: '{gold_answer}'. "
            f"The multi-hop queries may not have retrieved the right documents."
        )

    return dspy.Prediction(score=f1, feedback=feedback)


# ---------------------------------------------------------------------------
# Dataset loading (paper: 150 train / 300 val / 300 test)
# ---------------------------------------------------------------------------

TRAIN_SIZE = 150
VAL_SIZE = 300
TEST_SIZE = 300


def load_hotpotqa_dataset(
    train_size: int = TRAIN_SIZE,
    val_size: int = VAL_SIZE,
    test_size: int = TEST_SIZE,
):
    """Load HotpotQA distractor dataset and split into train/val/test."""
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")

    indices = list(range(len(dataset)))
    random.Random(42).shuffle(indices)

    total_needed = train_size + val_size + test_size
    selected = indices[:total_needed]

    train_examples = []
    val_examples = []
    test_examples = []

    for i, idx in enumerate(selected):
        item = dataset[idx]

        ex = dspy.Example(
            question=item["question"],
            answer=item["answer"],
        ).with_inputs("question")

        if i < train_size:
            train_examples.append(ex)
        elif i < train_size + val_size:
            val_examples.append(ex)
        else:
            test_examples.append(ex)

    return train_examples, val_examples, test_examples
