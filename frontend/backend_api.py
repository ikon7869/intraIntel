import requests
import json

def upload_document_to_backend(fastapi_base_url: str, uploaded_file):
    """
    Sends a PDF document to the FastAPI backend for upload and indexing.
    
    Args:
        fastapi_base_url (str): The base URL of the FastAPI backend.
        uploaded_file (streamlit.runtime.uploaded_file_manager.UploadedFile): The file object from Streamlit's file_uploader.
        
    Returns:
        dict: The JSON response from the backend on success.
        
    Raises:
        requests.exceptions.RequestException: If the HTTP request fails.
        json.JSONDecodeError: If the response is not valid JSON.
    """
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    
    response = requests.post(f"{fastapi_base_url}/upload-document/", files=files)
    response.raise_for_status()
    
    return response.json()

def chat_with_backend_agent(fastapi_base_url: str, session_id: str, query: str, restricted_mode: bool):
    """
    Sends a chat query to the FastAPI backend's agent.
    
    Args:
        fastapi_base_url (str): The base URL of the FastAPI backend.
        session_id (str): Unique ID for the current chat session.
        query (str): The user's chat message.
        restricted_mode (bool): Flag indicating if web search is disabled.
        
    Returns:
        tuple: (agent_response_text: str, trace_events: list)
        
    Raises:
        requests.exceptions.RequestException: If the HTTP request fails.
        json.JSONDecodeError: If the response is not valid JSON.
    """
    payload = {
        "session_id": session_id,
        "query": query,
        "restricted_mode": restricted_mode 
    }
    
    response = requests.post(f"{fastapi_base_url}/chat/", json=payload, stream=False)
    response.raise_for_status()
    
    data = response.json()
    agent_response = data.get("response", "Sorry, I couldn't get a response from the agent.")
    trace_events = data.get("trace_events", [])
    
    return agent_response, trace_events