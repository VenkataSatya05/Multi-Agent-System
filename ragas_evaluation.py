from typing import List, Dict

from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.evaluation import (
    EvaluationDataset,
    LangchainEmbeddingsWrapper,
    LangchainLLMWrapper,
    evaluate,
)
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall

from graph import build_graph

SAMPLES: List[Dict[str, str]] = [
    {
        "user_input": "Give me what exactly is sector of a circle mentioned in the textbook.",
        "reference": (
            "A sector of a circle is the region bounded by two radii and the included arc. "
            "Its area equals q/360 * π r^2 when q is the central angle in degrees."
        ),
    },
    {
        "user_input": "What is the area formula for a sector of a circle?",
        "reference": (
            "The area of a sector with central angle q is (q / 360) * π r^2."
        ),
    },
    {
        "user_input": "How much area does a 90 degree sector of a circle cover?",
        "reference": (
            "A 90 degree sector covers one quarter of the circle, so its area is (1/4) * π r^2."
        ),
    },
]


def build_evaluation_dataset(app):
    samples = []
    for sample in SAMPLES:
        question = sample["user_input"]
        state = app.invoke(
            {
                "question": question,
                "context": [],
                "answer": None,
                "verdict": None,
                "justification": None,
                "verification_context": [],
                "retry_count": 0,
            }
        )

        samples.append(
            {
                "user_input": question,
                "retrieved_contexts": state["context"],
                "response": state["answer"],
                "reference": sample["reference"],
            }
        )

    return EvaluationDataset.from_list(samples)


def main():
    app = build_graph()
    dataset = build_evaluation_dataset(app)

    llm = LangchainLLMWrapper(ChatOllama(model="qwen2.5:3b", temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

    print("Running RAGAS evaluation on textbook QA samples...")
    result = evaluate(
        dataset,
        metrics=[context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
        show_progress=False,
    )

    print("\nEvaluation results:")
    for metric_name, metric_value in result._repr_dict.items():
        print(f"- {metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
