import os
from groq import Groq


class GroqLLM:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"   # ✅ FIXED MODEL

    def invoke(self, prompt: str):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        class Result:
            def __init__(self, content):
                self.content = content

        return Result(response.choices[0].message.content)


def get_llm():
    return GroqLLM()
