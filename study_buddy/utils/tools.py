import os
import requests
from langchain_core.tools import tool, Tool

from langchain_community.tools import YouTubeSearchTool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_community.tools.google_books import GoogleBooksQueryRun
from langchain_community.utilities.google_books import GoogleBooksAPIWrapper
from langchain_community.tools.google_scholar import GoogleScholarQueryRun
from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from e2b_code_interpreter import Sandbox
from pydantic import BaseModel, Field

from study_buddy.utils.vector_store import get_vector_store
from study_buddy.config import FAISS_INDEX_DIR

google_api_key = os.getenv("GOOGLE_API_KEY")


# ------------------------------------------------------------------------------
# Basic tools
@tool(response_format="content_and_artifact")
def retrieve_tool(query: str):
    """Retrieve information related to a query."""
    vector_store = get_vector_store(FAISS_INDEX_DIR)
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


web_search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True
)


class CustomGoogleBooksAPIWrapper(GoogleBooksAPIWrapper):
    """Custom Google Books API Wrapper to handle missing fields gracefully."""

    def _format(self, query: str, books: list) -> str:
        if not books:
            return f"Sorry, no books could be found for your query: {query}"

        results = [f"Here are {len(books)} suggestions for books related to '{query}':"]

        for i, book in enumerate(books, start=1):
            info = book.get("volumeInfo", {})
            title = info.get("title", "Title not available")
            authors = self._format_authors(info.get("authors", ["Unknown author"]))
            summary = info.get("description", "No description available")  # ✅ FIXED
            source = info.get("infoLink", "No source available")

            desc = f'{i}. "{title}" by {authors}: {summary}\n'
            desc += f"You can read more at {source}"
            results.append(desc)

        return "\n\n".join(results)


google_books_tool = GoogleBooksQueryRun(api_wrapper=CustomGoogleBooksAPIWrapper(google_books_api_key=google_api_key))

# ------------------------------------------------------------------------------
# Tools for learning support

youtube_search_tool = YouTubeSearchTool()

wolfram = WolframAlphaAPIWrapper()


def wolfram_tool(query: str):
    """Query Wolfram Alpha and return the response."""
    return wolfram.run(query)


wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# ------------------------------------------------------------------------------
# Advanced tools for scientific research support
arxive_tool = ArxivQueryRun()

google_scholar_tool = GoogleScholarQueryRun(api_wrapper=GoogleScholarAPIWrapper())


class CustomWikidataAPIWrapper:
    """Custom Wrapper for Wikidata API using SPARQL."""

    def run(self, query: str) -> str:
        """Search for an entity on Wikidata using SPARQL and return detailed info."""
        wikidata_sparql_url = "https://query.wikidata.org/sparql"
        headers = {"User-Agent": "MyWikidataBot/1.0 (myemail@example.com)"}  # Sostituisci con la tua email

        sparql_query = f"""
        SELECT ?item ?itemLabel ?description ?alias ?propertyLabel ?propertyValueLabel WHERE {{
          ?item ?label "{query}"@en.
          OPTIONAL {{ ?item skos:altLabel ?alias FILTER (LANG(?alias) = "en") }}
          OPTIONAL {{ ?item schema:description ?description FILTER (LANG(?description) = "en") }}
          OPTIONAL {{ ?item ?property ?propertyValue.
                     ?property wikibase:directClaim ?propClaim.
                     ?propClaim rdfs:label ?propertyLabel.
                     ?propertyValue rdfs:label ?propertyValueLabel.
                     FILTER(LANG(?propertyLabel) = "en" && LANG(?propertyValueLabel) = "en") }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """

        response = requests.get(wikidata_sparql_url, params={"query": sparql_query, "format": "json"}, headers=headers)

        if response.status_code != 200:
            return f"Error: Unable to fetch data from Wikidata (status code {response.status_code})"

        data = response.json()
        results = data.get("results", {}).get("bindings", [])

        if not results:
            return f"No Wikidata result found for '{query}'"

        entities = {}
        for result in results:
            entity_id = result["item"]["value"].split("/")[-1]  # Extract QID
            label = result.get("itemLabel", {}).get("value", "Unknown")
            description = result.get("description", {}).get("value", "No description available")
            alias = result.get("alias", {}).get("value", "")
            property_label = result.get("propertyLabel", {}).get("value", "")
            property_value = result.get("propertyValueLabel", {}).get("value", "")

            if entity_id not in entities:
                entities[entity_id] = {
                    "Label": label,
                    "Description": description,
                    "Aliases": set(),
                    "Properties": {}
                }

            if alias:
                entities[entity_id]["Aliases"].add(alias)

            if property_label and property_value:
                if property_label not in entities[entity_id]["Properties"]:
                    entities[entity_id]["Properties"][property_label] = set()
                entities[entity_id]["Properties"][property_label].add(property_value)

        # Format output
        formatted_results = []
        for entity_id, details in entities.items():
            result_str = f"Result {entity_id}:\nLabel: {details['Label']}\nDescription: {details['Description']}"
            if details["Aliases"]:
                result_str += f"\nAliases: {', '.join(details['Aliases'])}"
            for prop, values in details["Properties"].items():
                result_str += f"\n{prop.lower()}: {', '.join(values)}"
            formatted_results.append(result_str)

        return "\n\n".join(formatted_results)


wikidata_tool = Tool(
    name="Wikidata",
    description="Searches for an entity on Wikidata and returns detailed info.",
    func=CustomWikidataAPIWrapper().run
)


class CodeInterpreterInput(BaseModel):
    """Input schema for the Code Interpreter."""
    code: str = Field(description="Python code to execute.")


class CodeInterpreterFunctionTool:
    """Tool to execute Python code in a Jupyter environment using E2B."""

    tool_name: str = "code_interpreter"

    def __init__(self):
        self._initialize_sandbox()

    def _initialize_sandbox(self):
        """Initialize or restart the E2B sandbox."""
        if "E2B_API_KEY" not in os.environ:
            raise Exception("E2B API key is missing. Set the E2B_API_KEY environment variable.")
        self.code_interpreter = Sandbox()
        self.code_interpreter.set_timeout(60)  # Extend timeout to 60 seconds

    def close(self):
        """Kill the sandbox."""
        self.code_interpreter.kill()

    def call(self, parameters: dict):
        code = parameters.get("code", "")
        try:
            execution = self.code_interpreter.run_code(code)
            return {
                "results": execution.results,
                "stdout": execution.logs.stdout,
                "stderr": execution.logs.stderr,
                "error": execution.error,
            }
        except Exception as e:
            if "502 Bad Gateway" in str(e) or "sandbox timeout" in str(e).lower():
                print("Sandbox timed out. Restarting...")
                self._initialize_sandbox()  # Restart the sandbox
                return self.call(parameters)  # Retry execution
            else:
                raise e

    def langchain_call(self, code: str):
        return self.call({"code": code})

    def to_langchain_tool(self) -> Tool:
        return Tool(
            name=self.tool_name,
            description="Executes Python code and returns stdout, stderr, and results.",
            func=self.langchain_call,
            args_schema=CodeInterpreterInput
        )


execute_python_tool = CodeInterpreterFunctionTool().to_langchain_tool()


# ------------------------------------------------------------------------------
# Tools for emotional support and mental health


# ------------------------------------------------------------------------------
# Tools for accessibility and inclusivity

tools = [
    retrieve_tool,
    web_search_tool,
    arxive_tool,
    execute_python_tool,
    google_books_tool,
    google_scholar_tool,
    wikidata_tool,
    wikipedia_tool,
    wolfram_tool,
    youtube_search_tool
]
