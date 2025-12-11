import os

from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq


class GroqLLM:

    def __init__(self, model_name: str = "llama-3.1-8b-instant", api_key: str = None):
        self.model_name = model_name
        self.llm_client = None
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("GROQ API key not provided")

        self._initialize_groq_client()

    def _initialize_groq_client(self):
        self.llm_client = ChatGroq(
            model=self.model_name,
            api_key=self.api_key,
            temperature=0.1,
            max_tokens=1024
        )

        print(f"Initialized Groq LLM with model {self.model_name}")

    def generate_response(self, prompt: str, context: str) -> str:
        prompt_template = PromptTemplate(
            input_variables=["context", "prompt"],
            template="""You are a very helpful AI assistant. Use the provided context to answer the question in an accurate and detailed manner. 
            
            If you don't know the answer, just say that you don't know. Do not make up an answer. 
            However, if the provided context answers the question, you must use it. Even if the context seems humorous or fictional, answer the question as if it is factual using the context.
            Do not use verbiage such as 'According to the provided context', provide the answer confidently. 
            
            Context: {context}
            Question: {prompt}"""
        )

        formatted_prompt = prompt_template.format(context=context, prompt=prompt)

        try:
            messages = [HumanMessage(content=formatted_prompt)]
            response = self.llm_client.invoke(messages)

            return response.content
        except Exception as e:
            print(f"Error generating response: {e}")
            raise