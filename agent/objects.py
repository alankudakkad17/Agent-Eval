from pydantic import BaseModel, Field

class Answer(BaseModel):
    answer:str = Field(description="the answer that is needed for the user question")
    conversation:bool = Field(description="This is an indicator to know if the conversation" \
    "should still continue or no. if the agent has made the decision about the refund this should be False" \
    "if the conversation still need more talk then it's True", default=True)

class PurchaseItem(BaseModel):
    name: str = Field(description="name of the item puchased")


class answer(Answer):  # noqa: N801 - lowercase on purpose
    """Final reply to the customer. Call this tool to finish.

    The class name is the tool name the agent sees. gpt-oss models always call
    it in lowercase, and Groq rejects calls to tool names that don't exist.
    """
