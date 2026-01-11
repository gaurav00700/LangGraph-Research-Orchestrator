"""
Agent Tools Configuration and Implementation.

This module defines the external tools available to the LangGraph workers, 
including web search (Hacker News, Arxiv), file system operations, 
Vector DB interactions, and MCP-based persistent memory.
"""

import json
import httpx
import os
from fpdf import FPDF
import subprocess
import logging
import arxiv
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langsmith import traceable

# --- Configuration ---
HN_API_BASE = "https://hn.algolia.com/api/v1"
MCP_SERVER_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mcp_server", "db_server.py"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "output"))
VECTOR_DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "vector_db"))

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if not os.path.exists(VECTOR_DB_DIR):
    os.makedirs(VECTOR_DB_DIR)

# --- MCP Client Implementation ---
class MCPClient:
    """
    A client for interacting with a local Model Context Protocol (MCP) server.
    
    The client starts a Python-based MCP server as a subprocess and communicates 
    via JSON-RPC over stdin/stdout.
    """
    def __init__(self, script_path: str):
        self.script_path = script_path
        self.process = None
        self._start_server()

    def _start_server(self):
        """Starts the MCP server subprocess."""
        self.process = subprocess.Popen(
            ["python", self.script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

    @traceable(run_type="tool", name="mcp_call_tool")
    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Sends a JSON-RPC request to the MCP server."""
        if self.process is None or self.process.poll() is not None:
            self._start_server()
        
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            },
            "id": 1
        }
        
        try:
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            
            response_line = self.process.stdout.readline()
            if not response_line:
                return "Error: No response from MCP server."
            
            response = json.loads(response_line)
            if "error" in response:
                return f"MCP Error: {response['error']['message']}"
            
            # Extract content from MCP standard response structure
            # result: { content: [ { type: 'text', text: '...' } ] }
            content = response.get("result", {}).get("content", [])
            if content and len(content) > 0:
                return content[0].get("text", "")
            return "Success"
            
        except Exception as e:
            return f"RPC Connection Error: {str(e)}"

# Global MCP Client Instance
mcp_client = MCPClient(MCP_SERVER_SCRIPT)

# --- Tools ---

@tool
def search_hn(query: str) -> str:
    """Searches Hacker News for stories related to the query."""
    try:
        url = f"{HN_API_BASE}/search"
        params = {"query": query, "tags": "story", "hitsPerPage": 5}
        # In testing we might want to mock httpx, but for this exercise we hit the real API
        # unless running in a restricted env.
        response = httpx.get(url, params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        
        hits = data.get("hits", [])
        if not hits:
            return "No results found on Hacker News."
        
        results = []
        for hit in hits:
            title = hit.get("title", "No Title")
            url = hit.get("url", "No URL")
            points = hit.get("points", 0)
            object_id = hit.get("objectID")
            results.append(f"- [HN] [{object_id}] {title} ({points} pts): {url}")
            
        return "\n".join(results)
    except Exception as e:
        return f"Error searching HN: {str(e)}"

@tool
def get_arxiv_details(pdf_url: str) -> str:
    """Fetches details (title, summary) of an Arxiv paper given its PDF URL."""
    try:
        # Extract ID from URL (e.g., https://arxiv.org/pdf/2410.09151v2 -> 2410.09151v2)
        paper_id = pdf_url.split("/")[-1].replace(".pdf", "")
        
        client = arxiv.Client()
        search = arxiv.Search(id_list=[paper_id])
        results = list(client.results(search))
        
        if not results:
            return "Paper not found."
            
        paper = results[0]
        return f"Title: {paper.title}\n\nAbstract: {paper.summary}"
    except Exception as e:
        return f"Error fetching details: {str(e)}"

@tool
def search_arxiv(query: str) -> str:
    """Searches Arxiv for research papers related to the query."""
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for r in client.results(search):
            results.append(f"- [Arxiv] {r.title} (Published: {r.published.year}): {r.pdf_url}")
            
        if not results:
            return "No results found on Arxiv."
            
        return "\n".join(results)
    except Exception as e:
        return f"Error searching Arxiv: {str(e)}"

@tool
def read_url(url: str) -> str:
    """Fetches text content from a general URL (non-Arxiv)."""
    try:
        # Arxiv Support: Convert PDF/Abs link to ar5iv (HTML version) for full extraction
        if "arxiv.org" in url:
            paper_id = url.split("/")[-1].replace(".pdf", "")
            # Try to hit ar5iv.org for HTML content
            url = f"https://ar5iv.org/abs/{paper_id}"
            
        headers = {"User-Agent": "Mozilla/5.0"}
        response = httpx.get(url, headers=headers, timeout=10.0, follow_redirects=True)
        response.raise_for_status()
        
        text = response.text
        # Very basic HTML stripping logic
        import re
        clean_text = re.sub('<[^<]+?>', '', text)
        clean_text = " ".join(clean_text.split())
        
        # If it was an Arxiv paper, denote that we fetched the full text
        source_note = " (Full paper via ar5iv)" if "ar5iv.org" in url else ""
        return f"CONTENT{source_note}:\n{clean_text[:4000]}..." # Increased truncation for full papers
    except Exception as e:
        return f"Error reading URL: {str(e)}"

@tool
def summarize_item(text: str) -> str:
    """Summarizes text (Simulated using simulated extraction)."""
    # In a real app, this would use an LLM. Here we just take the first 200 chars.
    return f"SUMMARY: {text[:200]}..."

@tool
def save_session_message(session_id: str, role: str, content: str) -> str:
    """Saves a conversation turn to the session history database."""
    client = MCPClient(MCP_SERVER_SCRIPT)
    return client.call_tool("add_session_message", {"session_id": session_id, "role": role, "content": content})

@tool
def get_session_history_tool(session_id: str) -> str:
    """Retrieves the full conversation history for a given session."""
    client = MCPClient(MCP_SERVER_SCRIPT)
    return client.call_tool("get_session_history", {"session_id": session_id})

@tool
def save_knowledge(content: str, category: str = "general") -> str:
    """Saves a piece of information to the Knowledge Base (MCP)."""
    return mcp_client.call_tool("add_memory", {"content": content, "category": category})

@tool
def search_knowledge(query: str) -> str:
    """Searches the Knowledge Base (MCP) for saved items."""
    return mcp_client.call_tool("search_memory", {"query": query})

@tool
def save_file(filename: str, content: str) -> str:
    """Saves content to a file in the output directory."""
    try:
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File saved successfully to {filepath}"
    except Exception as e:
        return f"Error saving file: {str(e)}"

@tool
def save_as_pdf(filename: str, content: str) -> str:
    """Saves text content as a PDF file in the output directory."""
    try:
        print(f"DEBUG: Generating PDF: {filename}")
        
        if not filename.endswith(".pdf"):
            filename += ".pdf"
            
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Mapping for common unicode characters that default FPDF fonts don't support.
        # FPDF standard fonts (Helvetica/Times) only support Latin-1 by default.
        replacements = {
            '\u2013': '-', # en dash
            '\u2014': '--', # em dash
            '\u2018': "'", # left single quote
            '\u2019': "'", # right single quote
            '\u201c': '"', # left double quote
            '\u201d': '"', # right double quote
            '\u2022': '*', # bullet
            '\u2026': '...', # ellipsis
            '\u00a0': ' ', # non-breaking space
        }
        
        clean_content = content
        for char, sub in replacements.items():
            clean_content = clean_content.replace(char, sub)
            
        # Final safety: strip anything that still isn't in latin-1 (replace with ?)
        clean_content = clean_content.encode("latin-1", "replace").decode("latin-1")
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12) # Standard Helvetica font
        
        # Split content by lines and add to PDF using word-wrapping (multi_cell)
        for line in clean_content.split("\n"):
            # Multi_cell handles word wrapping
            # 'text' is preferred in fpdf2, but 'txt' is for backward compat
            pdf.multi_cell(0, 10, text=line)
            
        pdf.output(filepath)
        print(f"DEBUG: PDF saved successfully: {filepath}")
        return f"PDF saved successfully to {filepath}"
    except Exception as e:
        return f"Error saving PDF: {str(e)}"

@tool
def index_content(content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Adds content to the persistent vector store (ChromaDB) for future similarity search.
    Use this to 'memorize' found information for other agents.
    """
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings,
            collection_name="research_notes"
        )
        
        # Add a unique ID if not provided
        import uuid
        doc_id = str(uuid.uuid4())
        
        if metadata is None:
            metadata = {}
        metadata["id"] = doc_id
        
        vectorstore.add_texts(
            texts=[content],
            metadatas=[metadata],
            ids=[doc_id]
        )
        return f"Successfully indexed content into vector store (ID: {doc_id})"
    except Exception as e:
        return f"Error indexing content: {str(e)}"

@tool
def search_vector_store(query: str, k: int = 3) -> str:
    """Searches the persistent vector store (ChromaDB) using similarity search.
    Returns the top 'k' relevant snippets.
    """
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings,
            collection_name="research_notes"
        )
        
        results = vectorstore.similarity_search(query, k=k)
        if not results:
            return "No relevant information found in vector store."
            
        formatted_results = []
        for i, doc in enumerate(results):
            # Try to get ID from metadata
            doc_id = doc.metadata.get("id", "Unknown")
            formatted_results.append(f"Result {i+1} (ID: {doc_id}):\n{doc.page_content}\nMetadata: {doc.metadata}")
            
        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Error searching vector store: {str(e)}"
