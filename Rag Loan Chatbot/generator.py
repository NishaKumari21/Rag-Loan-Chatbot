import openai
from transformers import pipeline
import os

class Generator:
    def __init__(self, use_openai=True):
        self.use_openai = use_openai
        if not use_openai:
            self.generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")

    def generate_response(self, context, query):
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        if self.use_openai:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            return response['choices'][0]['message']['content'].strip()
        else:
            result = self.generator(prompt, max_new_tokens=200, do_sample=True)
            return result[0]['generated_text']