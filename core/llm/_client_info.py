SUPPORTED_SOURCES = {
    "openai": "openai_sdk",
    "aoai": "openai_sdk",
    "llamafile": "openai_sdk",
    "ollama": "openai_sdk",
    "groq": "openai_sdk"
}

def get_client_info(source: str):
    try:
        return SUPPORTED_SOURCES[source]
    except KeyError:
        raise ValueError(f"Unsupported source: {source}")
