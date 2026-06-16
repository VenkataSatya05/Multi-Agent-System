import re
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from state import AgentState

SYSTEM_PROMPT = """You are a textbook assistant. Answer the question
using ONLY the provided context. If the context is insufficient, say so.
Do not hallucinate facts not present in the context."""

def make_qa_agent(vectorstore_path: str):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    store = FAISS.load_local(vectorstore_path, embeddings,
                             allow_dangerous_deserialization=True)
    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5, "fetch_k": 10}
    )
    llm = ChatOllama(model="qwen2.5:3b", temperature=0)

    # Keep a local copy of all documents for exact keyword fallback.
    all_docs = list(store.docstore._dict.values())
    all_docs_lc = [d.page_content.lower() for d in all_docs]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    chain = prompt | llm

    def _extract_keywords(text: str):
        tokens = re.findall(r"[a-zA-Z0-9']{4,}", text.lower())
        stopwords = {
            'what', 'which', 'when', 'where', 'who', 'how', 'the', 'a', 'an',
            'is', 'are', 'of', 'in', 'to', 'for', 'and', 'or', 'on', 'that',
            'this', 'these', 'those', 'with', 'by', 'from', 'as', 'have', 'has',
            'was', 'were', 'do', 'does', 'did', 'be', 'been', 'their', 'its',
            'your', 'our', 'into', 'about', 'between', 'through', 'give',
            'exactly', 'mentioned', 'mention', 'textbook', 'provide', 'details',
            'answer', 'meaning', 'means', 'whatsoever', 'that'
        }
        return [token for token in tokens if token not in stopwords]

    def _extract_topic_phrase(query: str):
        query_lower = query.lower()
        keywords = [kw for kw in _extract_keywords(query) if len(kw) >= 4]
        if len(keywords) < 2:
            return None
        first = keywords[0]
        last = keywords[-1]
        start = query_lower.find(first)
        end = query_lower.rfind(last)
        if start != -1 and end != -1 and end > start:
            phrase = query_lower[start:end + len(last)].strip()
            if len(phrase) >= 8:
                return phrase
        return None

    def _exact_doc_search(query: str):
        query_lower = query.lower()
        if len(query_lower) > 3:
            exact_hits = [doc for doc, text in zip(all_docs, all_docs_lc)
                          if query_lower in text]
            if exact_hits:
                return exact_hits[:5]

        topic_phrase = _extract_topic_phrase(query)
        if topic_phrase:
            phrase_hits = [doc for doc, text in zip(all_docs, all_docs_lc)
                           if topic_phrase in text]
            if phrase_hits:
                return phrase_hits[:5]

        keywords = [kw for kw in _extract_keywords(query) if len(kw) >= 4]
        if not keywords:
            return []

        # Prefer documents that contain all meaningful query terms.
        all_hits = [doc for doc, text in zip(all_docs, all_docs_lc)
                    if all(keyword in text for keyword in keywords)]
        if all_hits:
            return all_hits[:5]

        # If none contain all terms, prefer documents containing the most terms.
        scored = []
        for doc, text in zip(all_docs, all_docs_lc):
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:5]]

    def qa_node(state: AgentState) -> AgentState:
        question = state["question"]

        if state.get("justification"):
            question = f"{question}\n\nNote: A previous answer was wrong. Hint: {state['justification']}"

        docs = retriever.invoke(question)
        context = [d.page_content for d in docs]
        query_lower = question.lower()
        if query_lower not in "\n\n".join(context).lower():
            fallback_docs = _exact_doc_search(question)
            if fallback_docs:
                docs = fallback_docs
                context = [d.page_content for d in docs]

        response = chain.invoke({
            "context": "\n\n".join(context),
            "question": question
        })
        return {**state, "context": context, "answer": response.content,
                "retry_count": state.get("retry_count", 0) + 1}

    return qa_node