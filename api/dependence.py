async def return_supported_sources():
    return {
        "sources": {
            "openai": "sdk",
            "aoai": "sdk",
            "llamafile": "sdk",
            "ollama": "request"
        }
    }