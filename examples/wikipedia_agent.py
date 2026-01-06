"""Wikipedia Research Agent

Demonstrates:
- Tool calling with async execute functions
- Wikipedia API integration
- on_step_finish callback for logging
"""

import asyncio
import aiohttp
from ai_query import generate_text, google, tool, Field, step_count_is, StepFinishEvent


# --- Tools ---

@tool(description="Search Wikipedia for articles matching a query. Returns titles and snippets.")
async def search_wikipedia(
    query: str = Field(description="The search query")
) -> str:
    """Search Wikipedia and return article summaries."""
    print(f"  [Wikipedia] Searching: {query}")
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 3,
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                return f"Error: Wikipedia search failed (status {resp.status})"
            data = await resp.json()
            results = data.get("query", {}).get("search", [])
            if not results:
                return f"No Wikipedia articles found for '{query}'"

            summaries = []
            for r in results:
                title = r.get("title", "")
                snippet = r.get("snippet", "").replace("<span class=\"searchmatch\">", "").replace("</span>", "")
                summaries.append(f"- {title}: {snippet}...")

            return f"Found {len(results)} articles:\n" + "\n".join(summaries)


@tool(description="Get the full introduction/summary of a specific Wikipedia article by its exact title.")
async def get_wikipedia_summary(
    title: str = Field(description="The exact title of the Wikipedia article")
) -> str:
    """Get the full summary of a Wikipedia article by title."""
    print(f"  [Wikipedia] Getting summary: {title}")
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "titles": title,
        "format": "json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                return f"Error: Failed to get article (status {resp.status})"
            data = await resp.json()
            pages = data.get("query", {}).get("pages", {})
            for page_id, page in pages.items():
                if page_id == "-1":
                    return f"Article '{title}' not found."
                extract = page.get("extract", "No content available.")
                # Truncate if too long
                if len(extract) > 1500:
                    extract = extract[:1500] + "..."
                return f"Wikipedia article '{title}':\n\n{extract}"
            return "No article found."


# --- Callback ---

def log_step(event: StepFinishEvent):
    print(f"\n--- Step {event.step_number} ---")
    if event.step.tool_calls:
        for tc in event.step.tool_calls:
            print(f"Tool: {tc.name}({tc.arguments})")
    if event.step.text:
        print(f"Response: {event.step.text[:200]}...")


# --- Main ---

async def main():
    print("Wikipedia Research Agent")
    print("=" * 50)

    user_question = "What is the difference between machine learning and deep learning? Give me a detailed explanation."
    print(f"Question: {user_question}\n")

    result = await generate_text(
        model=google("gemini-2.0-flash"),
        system=(
            "You are a research assistant that uses Wikipedia to answer questions. "
            "First search for relevant articles, then get summaries of the most relevant ones, "
            "and finally synthesize a comprehensive answer."
        ),
        prompt=user_question,
        tools={
            "search_wikipedia": search_wikipedia,
            "get_wikipedia_summary": get_wikipedia_summary,
        },
        on_step_finish=log_step,
        stop_when=step_count_is(6),
    )

    print("\n" + "=" * 50)
    print("ANSWER")
    print("=" * 50)
    print(result.text)
    print(f"\nTotal tokens used: {result.usage.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
