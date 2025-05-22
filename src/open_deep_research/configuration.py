import os
from enum import Enum
from dataclasses import dataclass, fields
from typing import Any, Optional, Dict
from dotenv import load_dotenv

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from dataclasses import dataclass

load_dotenv()

DEFAULT_REPORT_STRUCTURE = """Use this structure to create a report on the user-provided topic:

1. Introduction (no research needed)
   - Brief overview of the topic area

2. Main Body Sections:
   - Each section should focus on a sub-topic of the user-provided topic
   
3. Conclusion
   - Aim for 1 structural element (either a list of table) that distills the main body sections 
   - Provide a concise summary of the report"""

class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    EXA = "exa"
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    LINKUP = "linkup"
    DUCKDUCKGO = "duckduckgo"
    GOOGLESEARCH = "googlesearch"

@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the chatbot."""
    # Common configuration
    report_structure: str = DEFAULT_REPORT_STRUCTURE # Defaults to the default report structure
    search_api: SearchAPI = SearchAPI.TAVILY # Default to TAVILY
    search_api_config: Optional[Dict[str, Any]] = None
    
    # Graph-specific configuration
    number_of_queries: int = os.environ.get('NUMBER_OF_QUERIES', 2) # Number of search queries to generate per iteration
    max_search_depth: int = os.environ.get('MAX_SEARCH_DEPTH', 2) # Maximum number of reflection + search iterations
    planner_provider: str = os.environ.get('PLANNER_PROVIDER', 'google_genai')  # Defaults to Anthropic as provider
    planner_model: str = os.environ.get('PLANNER_MODEL', 'gemini-2.0-flash') # Defaults to claude-3-7-sonnet-latest
    planner_model_kwargs: Optional[Dict[str, Any]] = None # kwargs for planner_model
    writer_provider: str = os.environ.get('WRITER_PROVIDER', 'google_genai') # Defaults to Anthropic as provider
    writer_model: str = os.environ.get('WRITER_MODEL', 'gemini-2.0-flash') # Defaults to claude-3-5-sonnet-latest
    writer_model_kwargs: Optional[Dict[str, Any]] = None # kwargs for writer_model
    search_api: SearchAPI = SearchAPI.TAVILY # Default to TAVILY
    search_api_config: Optional[Dict[str, Any]] = None 
    
    # Multi-agent specific configuration
    supervisor_provider: str = os.environ.get('SUPERVISOR_PROVIDER', 'google_genai')
    supervisor_model: str = os.environ.get('SUPERVISOR_MODEL', 'gemini-2.0-flash')
    supervisor_model_kwargs: Optional[Dict[str, Any]] = None
    researcher_provider: str = os.environ.get('RESEARCHER_PROVIDER', 'google_genai')
    researcher_model: str = os.environ.get('RESEARCHER_MODEL', 'gemini-2.0-flash') 
    researcher_model_kwargs: Optional[Dict[str, Any]] = None

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})
