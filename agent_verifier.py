from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from state import AgentState

VERIFY_PROMPT = """You are a strict answer verifier for a textbook QA system.

Given:
- The original question
- A candidate answer
- Retrieved passages from the textbook

Your job: determine if the answer is CORRECT, PARTIALLY_CORRECT, or INCORRECT
based solely on the textbook passages. Provide a one-sentence justification.

Respond in this exact format:
VERDICT: <CORRECT|PARTIALLY_CORRECT|INCORRECT>
JUSTIFICATION: <your reasoning>"""

def make_verifier_agent(vectorstore_path: str):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    store = FAISS.load_local(vectorstore_path, embeddings,
                             allow_dangerous_deserialization=True)
    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5, "fetch_k": 10}
    )
    llm = ChatOllama(model="qwen2.5:3b", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", VERIFY_PROMPT),
        ("human",
         "Question: {question}\nCandidate Answer: {answer}\n\nContext:\n{context}")
    ])
    chain = prompt | llm

    def verifier_node(state: AgentState) -> AgentState:
        if state.get("context"):
            v_context = state["context"]
        else:
            docs = retriever.invoke(state["question"])
            v_context = [d.page_content for d in docs]

        response = chain.invoke({
            "question": state["question"],
            "answer": state["answer"],
            "context": "\n\n".join(v_context)
        })

        lines = response.content.strip().splitlines()
        verdict = "INCORRECT"
        justification = ""
        for line in lines:
            if line.startswith("VERDICT:"):
                verdict = line.replace("VERDICT:", "").strip()
            elif line.startswith("JUSTIFICATION:"):
                justification = line.replace("JUSTIFICATION:", "").strip()

        return {**state,
                "verification_context": v_context,
                "verdict": verdict,
                "justification": justification}

    return verifier_node