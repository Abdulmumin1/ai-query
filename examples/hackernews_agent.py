"""Hacker News Reader Agent

Demonstrates:
- Fetching real data from Hacker News API
- Multiple related tools (list stories, get details, get comments)
- Streaming with tool calling
"""

import asyncio
import aiohttp
from ai_query import stream_text, google, tool, Field, step_count_is


# --- Hacker News API Tools ---

@tool(description="Get the current top stories from Hacker News. Returns title, score, author, and comment count.")
async def get_top_stories(
    limit: int = Field(description="Number of stories to fetch (default 5, max 10)", default=5)
) -> str:
    """Get the top stories from Hacker News."""
    print(f"  [HN] Fetching top {limit} stories...")

    async with aiohttp.ClientSession() as session:
        # Get top story IDs
        async with session.get("https://hacker-news.firebaseio.com/v0/topstories.json") as resp:
            if resp.status != 200:
                return "Error: Failed to fetch top stories"
            story_ids = await resp.json()

        # Fetch details for top N stories
        stories = []
        for story_id in story_ids[:limit]:
            async with session.get(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json") as resp:
                if resp.status == 200:
                    story = await resp.json()
                    if story:
                        stories.append({
                            "id": story.get("id"),
                            "title": story.get("title"),
                            "score": story.get("score"),
                            "by": story.get("by"),
                            "url": story.get("url", "No URL"),
                            "comments": story.get("descendants", 0),
                        })

        if not stories:
            return "No stories found"

        result = "Top Hacker News Stories:\n"
        for i, s in enumerate(stories, 1):
            result += f"\n{i}. [{s['id']}] {s['title']}\n"
            result += f"   Score: {s['score']} | By: {s['by']} | Comments: {s['comments']}\n"

        return result


@tool(description="Get detailed information about a specific Hacker News story by its ID.")
async def get_story_details(
    story_id: int = Field(description="The Hacker News story ID")
) -> str:
    """Get detailed information about a specific story."""
    print(f"  [HN] Getting details for story {story_id}...")

    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json") as resp:
            if resp.status != 200:
                return f"Error: Failed to fetch story {story_id}"
            story = await resp.json()

            if not story:
                return f"Story {story_id} not found"

            result = f"Story: {story.get('title')}\n"
            result += f"Author: {story.get('by')}\n"
            result += f"Score: {story.get('score')} points\n"
            result += f"Time: {story.get('time')}\n"
            result += f"URL: {story.get('url', 'No URL (text post)')}\n"
            result += f"Comments: {story.get('descendants', 0)}\n"

            if story.get("text"):
                text = story["text"][:500]
                result += f"\nText: {text}..."

            return result


@tool(description="Get the top comments on a Hacker News story.")
async def get_top_comments(
    story_id: int = Field(description="The Hacker News story ID"),
    limit: int = Field(description="Number of comments to fetch (default 3)", default=3)
) -> str:
    """Get top comments for a story."""
    print(f"  [HN] Getting comments for story {story_id}...")

    async with aiohttp.ClientSession() as session:
        # Get story to find comment IDs
        async with session.get(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json") as resp:
            if resp.status != 200:
                return "Error: Failed to fetch story"
            story = await resp.json()

        comment_ids = story.get("kids", [])[:limit]
        if not comment_ids:
            return "No comments on this story"

        comments = []
        for cid in comment_ids:
            async with session.get(f"https://hacker-news.firebaseio.com/v0/item/{cid}.json") as resp:
                if resp.status == 200:
                    comment = await resp.json()
                    if comment and comment.get("text"):
                        # Clean up HTML
                        text = comment["text"].replace("<p>", "\n").replace("</p>", "")
                        text = text[:300] + "..." if len(text) > 300 else text
                        comments.append({
                            "by": comment.get("by"),
                            "text": text,
                        })

        if not comments:
            return "No readable comments found"

        result = f"Top {len(comments)} comments:\n"
        for c in comments:
            result += f"\n@{c['by']}:\n{c['text']}\n"

        return result


# --- Main ---

async def main():
    print("Hacker News Agent (Streaming)")
    print("=" * 50)

    prompt = (
        "What are the top 3 stories on Hacker News right now? "
        "Pick the most interesting one and show me what people are saying about it in the comments. "
        "Give me a brief summary of the discussion."
    )
    print(f"Request: {prompt}\n")
    print("-" * 50)
    print("Response (streaming):\n")

    result = stream_text(
        model=google("gemini-2.0-flash"),
        system="You are a tech news assistant that helps users explore Hacker News.",
        prompt=prompt,
        tools={
            "get_top_stories": get_top_stories,
            "get_story_details": get_story_details,
            "get_top_comments": get_top_comments,
        },
        stop_when=step_count_is(5),
    )

    # Stream the response
    async for chunk in result.text_stream:
        print(chunk, end="", flush=True)

    # Get final stats
    usage = await result.usage
    print(f"\n\n[Tokens: {usage.total_tokens}]")


if __name__ == "__main__":
    asyncio.run(main())
