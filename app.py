import os
import json
import threading
import requests
from pathlib import Path
from typing import Optional, List, Dict

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from pypdf import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv(override=True)

# --- Pydantic Models ---

class RecordUserDetailsArgs(BaseModel):
    email: str = Field(..., description="The email address provided by the user.")
    name: str = Field(default="Not provided", description="The user's name.")
    notes: str = Field(default="None", description="Brief context of the conversation.")

class RecordUnknownQuestionArgs(BaseModel):
    question: str = Field(..., description="The specific question the assistant couldn't answer.")

# --- Tool Utilities ---

def pydantic_to_tool(name: str, description: str, model: type[BaseModel]) -> dict:
    """Convert a Pydantic model into an OpenAI tool with full Strict Mode compliance."""
    schema = model.model_json_schema()
    
    # 1. ALL properties must be listed as 'required' for Strict Mode
    if "properties" in schema:
        schema["required"] = list(schema["properties"].keys())
    
    # 2. additionalProperties must be set to False
    schema["additionalProperties"] = False
    
    # 3. Remove keys that are forbidden in Strict Mode (title, default)
    schema.pop("title", None)
    if "properties" in schema:
        for prop in schema["properties"].values():
            if isinstance(prop, dict):
                prop.pop("title", None)
                # 'default' is not allowed in the schema for Structured Outputs
                prop.pop("default", None) 

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema,
            "strict": True
        },
    }

# --- Shared Logic ---

def push_notification(text: str):
    """Sends a push notification via Pushover in a background thread."""
    def send():
        try:
            requests.post(
                "https://api.pushover.net/1/messages.json",
                data={
                    "token": os.getenv("PUSHOVER_TOKEN"),
                    "user": os.getenv("PUSHOVER_USER"),
                    "message": text,
                },
                timeout=5,
            )
        except Exception as e:
            print(f"Notification error: {e}")
    threading.Thread(target=send, daemon=True).start()

# --- Main Assistant Class ---

class PersonalAssistant:
    def __init__(self):
        # Configuration
        self.openai_model = "gpt-4o-mini" # Faster and cheaper for personal sites
        self.groq_model = "llama-3.1-8b-instant"
        self.user_name = "Mohit Kumar"
        
        # Clients
        self.client = OpenAI()
        self.groq = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )

        # Load Knowledge Base
        print("Initializing Embeddings & FAISS...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = FAISS.load_local(
            "faiss_index", self.embeddings, allow_dangerous_deserialization=True
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        
        # Registry
        self.tools = [
            pydantic_to_tool("record_user_details", "Save user contact info.", RecordUserDetailsArgs),
            pydantic_to_tool("record_unknown_question", "Log questions you can't answer.", RecordUnknownQuestionArgs)
        ]

    def _get_standalone_question(self, message: str, history: List[Dict]) -> str:
        if not history:
            return message
        
        # We only care about the last few turns for context
        context_window = history[-4:] 
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in context_window])
        
        prompt = f"""
        Re-write the following USER MESSAGE into a standalone search query for a vector database.
        The goal is to find information in a resume. 
        If the user uses pronouns (it, they, that, more) or asks follow-up questions, 
        replace them with the actual subject discussed in the history.

        HISTORY:
        {history_str}

        USER MESSAGE: {message}

        STANDALONE QUERY:"""
        
        res = self.groq.chat.completions.create(
            model=self.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0 # Keep it predictable
        )
        return res.choices[0].message.content

    def _execute_tool(self, tool_call) -> str:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        
        print(f"-> Executing Tool: {name}")
        
        if name == "record_user_details":
            push_notification(f"New Lead: {args.get('name')} ({args.get('email')})\nNotes: {args.get('notes')}")
            return json.dumps({"status": "success", "message": "Details recorded."})
        
        if name == "record_unknown_question":
            push_notification(f"Unknown Query: {args.get('question')}")
            return json.dumps({"status": "logged", "message": "I will make sure Mohit sees this."})
            
        return json.dumps({"error": "Unknown tool"})

    def build_system_prompt(self, context: str) -> str:
        return (
            f"You are {self.user_name}, an expert Data Scientist. You are speaking through your AI persona.\n\n"
        f"CHARACTER GUIDELINES:\n"
        f"- Be helpful, professional, and enthusiastic about your work.\n"
        f"- You have deep expertise in Text2SQL, RAG, and Agentic workflows.\n\n"
        f"KNOWLEDGE RULES:\n"
        f"- Use the CONTEXT below to answer the user's question.\n"
        f"- If the information is not in the context, but is a general professional question (e.g., 'What is your email?'), provide the answer (mohit.in@outlook.com). if linkedin, provide the answer (https://www.linkedin.com/in/mohitkumar-in/).\n"
        f"- If the user asks for technical details NOT in the context, say: 'I haven't added specific details about that project to my index yet, but I'd be happy to discuss it via email.'\n\n"
        f"- If the user asks for github, provide the answer (https://github.com/mohitktanwar-in).\n\n"
        f"- Gently Nudge the user to get in touch with you via email or linkedin or share their contact details, but do not force them to share their contact details.\n\n"
        f"CONTEXT FROM RESUME:\n"
        f"---START---\n{context}\n---END---"
        )

    def chat(self, message: str, history: List[Dict]):
        # 1. Query Contextualization
        standalone_query = self._get_standalone_question(message, history)
        
        # 2. RAG Retrieval
        docs = self.retriever.invoke(standalone_query)
        context = "\n".join([d.page_content for d in docs])
        
        # 3. Prepare Messages
        messages = [{"role": "system", "content": self.build_system_prompt(context)}]
        # Append history (Gradio format is already OpenAI compatible)
        messages.extend(history)
        messages.append({"role": "user", "content": message})

        # 4. Agentic Loop (Handle Tool Calls)
        while True:
            response = self.client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )
            
            msg = response.choices[0].message
            if not msg.tool_calls:
                break
            
            # Add assistant's tool call to history
            messages.append(msg)
            
            # Execute and add tool results
            for tool_call in msg.tool_calls:
                result_content = self._execute_tool(tool_call)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": result_content
                })

        # 5. Final Streaming Output
        stream = self.client.chat.completions.create(
            model=self.openai_model,
            messages=messages,
            stream=True
        )
        
        partial_text = ""
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                partial_text += content
                yield partial_text

# --- Launch ---

if __name__ == "__main__":
    if not Path("faiss_index").exists():
        print("Error: faiss_index not found. Run your indexing script first.")
    else:
        assistant = PersonalAssistant()
        demo = gr.ChatInterface(
            fn=assistant.chat,
            type="messages",
            title=f"Chat with {assistant.user_name}",
            description="Ask me about my experience, projects, or contact details.",
            examples=["Tell me about your AI projects", "How can I contact you?"]
        )
        demo.launch()