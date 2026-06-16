from typing import TypedDict, Optional, List  # ← this line was missing

class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: Optional[str]
    verdict: Optional[str]
    justification: Optional[str]
    verification_context: List[str]
    retry_count: int