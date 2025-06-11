from typing import List, TypedDict, Optional


class APIRequest(TypedDict):
    session_id: Optional[str]
    prompt: str
    num_samples: int
    max_output_len: Optional[int]
    ignore_eos: bool


class APIResponse(TypedDict):
    token_ids: List[int]
    output_text: str
    output_len: int
    token_latencies: List[float]
