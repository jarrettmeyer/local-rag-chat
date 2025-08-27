import os
import sys
import ollama
from .database import Database
from .document_processor import DocumentProcessor


class ChatClient:


    def load_system_prompt(self):
        prompt_path = os.path.join(os.path.dirname(__file__), '../../prompts/system_prompt.md')
        with open(prompt_path, 'r') as f:
            return f.read()

    def get_relevant_context(self, db: Database, question: str, top_k: int=5):
        processor = DocumentProcessor()
        question_embedding = processor.get_embedding(question)
        return db.get_relevant_chunks_by_embedding(question_embedding, top_k)

    def build_prompt(self, system_prompt, context_chunks, user_question):
        context_str = "\n\n".join(context_chunks)
        return f"{system_prompt}\n\nContext:\n{context_str}\n\nUser Question: {user_question}\n\nAnswer:"

    def chat_loop(self, db: Database):
        system_prompt = self.load_system_prompt()
        print("Type your question (or /q to quit):")
        while True:
                user_input = input("You: ").strip()
                if user_input.lower() in {"/q", "/quit"}:
                    print("Goodbye!")
                    break
                context_chunks = self.get_relevant_context(db, user_input)
                prompt = self.build_prompt(system_prompt, context_chunks, user_input)
                model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
                response = ollama.generate(model=model, prompt=prompt)
                answer = response.get("response")
                print(f"AI: {answer}")
