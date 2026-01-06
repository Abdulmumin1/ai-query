"""Example of a Real-World Agent using the Google provider."""

import asyncio
import json
import aiohttp
from ai_query import generate_text, google, tool, Field, step_count_is, StepFinishEvent


# --- Tools ---

@tool(description="Get the current price of a cryptocurrency (e.g., 'bitcoin', 'ethereum', 'solana') in USD.")
async def get_crypto_price(
    coin_id: str = Field(description="The ID of the coin on CoinGecko.", default="bitcoin")
) -> str:
    """Fetch live cryptocurrency prices from CoinGecko API."""
    print(f"  [Tool] Fetching price for: {coin_id}")
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                return f"Error: Failed to fetch price (Status {resp.status})"
            data = await resp.json()
            price = data.get(coin_id, {}).get("usd")
            return f"The current price of {coin_id} is ${price} USD." if price else f"Coin '{coin_id}' not found."


@tool(description="Get the current weather for a specific city.")
async def get_weather(
    location: str = Field(description="The name of the city.")
) -> str:
    """Fetch current weather using Open-Meteo API."""
    print(f"  [Tool] Fetching weather for: {location}")
    geocode_url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json&limit=1"
    headers = {"User-Agent": "ai-query-example/1.0"}

    async with aiohttp.ClientSession() as session:
        async with session.get(geocode_url, headers=headers) as resp:
            if resp.status != 200:
                return "Error: Geocoding failed."
            geo_data = await resp.json()
            if not geo_data:
                return f"Location '{location}' not found."

            lat = geo_data[0]["lat"]
            lon = geo_data[0]["lon"]

            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
            async with session.get(weather_url) as w_resp:
                if w_resp.status != 200:
                    return "Error: Weather API failed."
                w_data = await w_resp.json()
                current = w_data.get("current_weather", {})
                temp = current.get("temperature")
                wind = current.get("windspeed")
                return f"Current weather in {location}: {temp}°C with wind speed {wind} km/h."


# --- Callbacks ---

def on_step_finish(event: StepFinishEvent):
    """Log the progress of each step in a premium way."""
    print(f"\n--- Step {event.step_number} ---")
    
    if event.step.text:
        print(f"Thought: {event.step.text.strip()}")
        
    if event.step.tool_calls:
        for tc in event.step.tool_calls:
            args = json.dumps(tc.arguments)
            print(f"Action: Call tool '{tc.name}' with arguments {args}")
            
    if event.step.tool_results:
        for tr in event.step.tool_results:
            print(f"Observation: {tr.result}")
            
    if event.usage:
        print(f"Usage: {event.usage.total_tokens} tokens so far")


# --- Main ---

async def main():
    print("Initializing Market & Environment Agent (powered by Google Gemini-2.0-Flash)...")
    
    model = google("gemini-2.0-flash")
    
    system_prompt = (
        "You are a helpful assistant that can fetch live market data and weather information. "
        "Use the tools provided to answer the user's questions with real-time data. "
        "If you need to check multiple things, do them in sequence."
    )
    
    user_prompt = "What is the current price of Bitcoin, and what is the weather like in London right now?"

    try:
        result = await generate_text(
            model=model,
            system=system_prompt,
            prompt=user_prompt,
            tools={
                "get_crypto_price": get_crypto_price,
                "get_weather": get_weather,
            },
            on_step_finish=on_step_finish,
            stop_when=step_count_is(5)
        )

        print("\n" + "="*50)
        print("FINAL ANSWER")
        print("="*50)
        print(result.text)
        print("="*50)
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure GOOGLE_API_KEY is set in your environment.")


if __name__ == "__main__":
    asyncio.run(main())