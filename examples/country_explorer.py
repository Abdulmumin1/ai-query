"""Country Data Explorer Agent

Demonstrates:
- REST Countries API integration
- Multiple tools for exploring data
- on_step_start to modify messages before each call
"""

import asyncio
import aiohttp
from ai_query import generate_text, google, tool, step_count_is, StepStartEvent, StepFinishEvent


# --- Tools ---

async def search_country(name: str):
    """Search for a country by name."""
    print(f"  [API] Searching for country: {name}")

    async with aiohttp.ClientSession() as session:
        url = f"https://restcountries.com/v3.1/name/{name}"
        async with session.get(url) as resp:
            if resp.status == 404:
                return f"No country found matching '{name}'"
            if resp.status != 200:
                return f"Error: API returned status {resp.status}"

            data = await resp.json()
            if not data:
                return "No results found"

            # Return first match
            country = data[0]
            return {
                "name": country.get("name", {}).get("common"),
                "official_name": country.get("name", {}).get("official"),
                "capital": country.get("capital", ["N/A"])[0] if country.get("capital") else "N/A",
                "region": country.get("region"),
                "subregion": country.get("subregion"),
                "population": country.get("population"),
                "area_km2": country.get("area"),
                "languages": list(country.get("languages", {}).values()),
                "currencies": list(country.get("currencies", {}).keys()),
                "flag_emoji": country.get("flag"),
            }


async def get_countries_by_region(region: str):
    """Get all countries in a region."""
    print(f"  [API] Getting countries in region: {region}")

    async with aiohttp.ClientSession() as session:
        url = f"https://restcountries.com/v3.1/region/{region}"
        async with session.get(url) as resp:
            if resp.status != 200:
                return f"Error: Could not fetch region '{region}'"

            data = await resp.json()
            countries = []
            for c in data[:15]:  # Limit to 15
                countries.append({
                    "name": c.get("name", {}).get("common"),
                    "capital": c.get("capital", ["N/A"])[0] if c.get("capital") else "N/A",
                    "population": c.get("population"),
                })

            # Sort by population
            countries.sort(key=lambda x: x["population"], reverse=True)
            return {"region": region, "countries": countries, "total": len(data)}


async def compare_countries(country1: str, country2: str):
    """Compare two countries side by side."""
    print(f"  [API] Comparing {country1} vs {country2}")

    async with aiohttp.ClientSession() as session:
        results = {}
        for name in [country1, country2]:
            url = f"https://restcountries.com/v3.1/name/{name}"
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data:
                        c = data[0]
                        results[name] = {
                            "name": c.get("name", {}).get("common"),
                            "population": c.get("population"),
                            "area_km2": c.get("area"),
                            "region": c.get("region"),
                        }

        if len(results) != 2:
            return "Could not find one or both countries"

        return {"comparison": results}


search_tool = tool(
    description="Search for a country by name and get detailed information.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Country name to search for"}
        },
        "required": ["name"]
    },
    execute=search_country
)

region_tool = tool(
    description="Get a list of all countries in a region (e.g., 'Europe', 'Asia', 'Africa', 'Americas', 'Oceania').",
    parameters={
        "type": "object",
        "properties": {
            "region": {"type": "string", "description": "The region name"}
        },
        "required": ["region"]
    },
    execute=get_countries_by_region
)

compare_tool = tool(
    description="Compare two countries side by side (population, area, etc.).",
    parameters={
        "type": "object",
        "properties": {
            "country1": {"type": "string", "description": "First country name"},
            "country2": {"type": "string", "description": "Second country name"}
        },
        "required": ["country1", "country2"]
    },
    execute=compare_countries
)


# --- Callbacks ---

request_count = 0

def on_start(event: StepStartEvent):
    global request_count
    request_count += 1
    print(f"\n[Step {event.step_number}] Preparing request #{request_count}...")
    print(f"  Messages in context: {len(event.messages)}")


def on_finish(event: StepFinishEvent):
    print(f"[Step {event.step_number}] Complete")
    if event.step.tool_calls:
        for tc in event.step.tool_calls:
            print(f"  Tool: {tc.name}")
    print(f"  Tokens so far: {event.usage.total_tokens}")


# --- Main ---

async def main():
    print("Country Data Explorer")
    print("=" * 50)

    queries = [
        "What are the top 5 most populous countries in Europe?",
        "Compare Japan and Germany - which one is larger by area and population?",
        "Tell me about New Zealand - its capital, languages, and interesting facts.",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*50}")
        print(f"Query {i}: {query}")
        print("=" * 50)

        result = await generate_text(
            model=google("gemini-2.0-flash"),
            system="You are a geography expert. Use the available tools to look up accurate country data.",
            prompt=query,
            tools={
                "search_country": search_tool,
                "get_countries_by_region": region_tool,
                "compare_countries": compare_tool,
            },
            on_step_start=on_start,
            on_step_finish=on_finish,
            stop_when=step_count_is(4),
        )

        print(f"\nAnswer:\n{result.text}")


if __name__ == "__main__":
    asyncio.run(main())
