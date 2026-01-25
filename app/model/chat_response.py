from pydantic import BaseModel
from typing import Optional, Literal

class ChatResponse(BaseModel):
    response_status: Literal["success", "failed"]
    user_message: str
    model_response: str
    timestamp: str
    feedback: Optional[Literal[0, 1]] = None
