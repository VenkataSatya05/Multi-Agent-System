from dotenv import load_dotenv
load_dotenv()
from ingest import build_vector_store
from graph import build_graph

# Run once to build the index
# build_vector_store("your_textbook.pdf")

app = build_graph()

result = app.invoke({
    "question": "give me what Exactly is sector of a Circle mentioned in the textbook.",
    "context": [],
    "answer": None,
    "verdict": None,
    "justification": None,
    "verification_context": [],
    "retry_count": 0
})

print("Answer:       ", result["answer"])
print("Verdict:      ", result["verdict"])
print("Justification:", result["justification"])