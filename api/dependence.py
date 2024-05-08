async def return_supported_sources():
    return {
        "sources": {
            "openai": "sdk",
            "aoai": "sdk",
            "llamafile": "request_oai",
            "ollama": "request_raw"
        }
    }