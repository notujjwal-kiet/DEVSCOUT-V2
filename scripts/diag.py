import asyncio, time
from app.research.discovery import discover_sources
from app.llm.provider import GeminiProvider

async def main():
    p = GeminiProvider()
    t = time.perf_counter()
    out = await p.generate("Reply with the single word: ok")
    print("plain call:", round(time.perf_counter() - t, 1), "s ->", out[:40])
    for q in ["FastAPI path parameters official documentation",
              "FastAPI Pydantic request body tutorial"]:
        t = time.perf_counter()
        try:
            res = await discover_sources(query=q, max_results=8, provider=p)
            print(q, "->", len(res), "sources,", round(time.perf_counter() - t, 1), "s")
            for s in res:
                print("   ", s.url)
        except Exception as e:
            print(q, "-> ERROR", type(e).__name__, str(e)[:200])

asyncio.run(main())