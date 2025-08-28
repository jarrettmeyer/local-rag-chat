import os
from typing import Optional

import ollama

from .database import Database


class ChatClient:
    def __init__(self, db: Database):
        self.db = db
        self.embedding_model_name = "nomic-embed-text"

    def build_prompt(self, system_prompt, context_chunks, user_question):
        context_str = "\n\n".join(context_chunks)
        return f"{system_prompt}\n\nContext:\n{context_str}\n\nUser Question: {user_question}\n\nAnswer:"

    def chat_loop(self, user_id: str):
        if not self.is_ollama_healthy():
            print("Ollama endpoint is not available. Exiting.")
            return

        system_prompt = self.load_system_prompt()

        print("\nType your question (or /q to quit):", end="\n\n")

        while True:
            question = input("You: ").strip()

            # Is the user trying to exit?
            if question.lower() in {"/q", "/quit"}:
                print("Goodbye!")
                break

            context_chunks = self.get_relevant_context(question, user_id)
            if not context_chunks:
                print(
                    "No documents available for this user or no relevant context found."
                )
                continue

            prompt = self.build_prompt(system_prompt, context_chunks, question)
            model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
            response = ollama.generate(model=model, prompt=prompt)
            answer = response.get("response")

            print(f"\nAI: {answer}", end="\n\n")

    def get_embedding(self, text: str) -> list:
        """Get an embedding for the given text using the Ollama SDK."""
        response = ollama.embeddings(model=self.embedding_model_name, prompt=text)
        return response["embedding"]

    def get_relevant_context(self, question: str, user_id: str, top_k: int = 5):
        question_embedding = self.get_embedding(question)
        return self.db.get_relevant_chunks_by_embedding(
            question_embedding, user_id, top_k
        )

    def is_ollama_healthy(self) -> bool:
        """Check if the Ollama endpoint is functioning."""
        try:
            # Try a simple model listing call
            ollama.list()
            return True
        except Exception as e:
            print(f"Ollama endpoint health check failed: {e}")
            return False

    def load_system_prompt(self):
        prompt_path = os.path.join(
            os.path.dirname(__file__), "./prompts/system_prompt.md"
        )
        with open(prompt_path, "r") as f:
            return f.read()
