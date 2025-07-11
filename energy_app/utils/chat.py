from groq import Groq
from ..settings import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)

def chat(messages: list[dict], model: str = "llama-3.1-8b-instant",
         temperature: float = 0.4, max_tokens: int = 768):
    """Return the assistant's full response text."""
    resp = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        top_p=1,
        stream=False,
    )
    return resp.choices[0].message.content.strip()
