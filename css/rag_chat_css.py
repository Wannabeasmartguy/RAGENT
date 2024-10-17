RAG_CHAT_USER_STYLE_ST_V37 = """
<style>
    .stChatMessage:has(.rag-chat-user) {
        flex-direction: row-reverse;
        // width: fit-content;
        // max-width: 85%;
        width: 90%;
        margin-left: auto;
        margin-right: 0;
        background-color: #E7F8FF;
        border-radius: 10px;
        padding: 20px;
    }
    .stChatMessage:has(.rag-chat-user) .stCodeBlock {
        text-align: left;
    }
</style>
"""

RAG_CHAT_USER_STYLE_ST_V39 = """
<style>
    .stChatMessage:has(.rag-chat-user) {
        flex-direction: row-reverse;
        // width: fit-content;
        // max-width: 85%;
        width: 90%;
        margin-left: auto;
        margin-right: 0;
        background-color: #E7F8FF;
        border-radius: 10px;
        padding: 20px;
    }
    .stChatMessage:has(.rag-chat-user) .stCode {
        text-align: left;
    }
</style>
"""

RAG_CHAT_ASSISTANT_STYLE_ST_V37 = """
<style>
    .stChatMessage:has(.rag-chat-assistant) {
        flex-direction: row;
        text-align: left;
        width: 90%;
        margin-left: 0;
        margin-right: auto;
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 20px;
    }
    .stChatMessage:has(.rag-chat-assistant) .stCodeBlock {
        text-align: left;
    }
</style>
"""

RAG_CHAT_ASSISTANT_STYLE_ST_V39 = """
<style>
    .stChatMessage:has(.rag-chat-assistant) {
        flex-direction: row;
        text-align: left;
        width: 90%;
        margin-left: 0;
        margin-right: auto;
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 20px;
    }
    .stChatMessage:has(.rag-chat-assistant) .stCode {
        text-align: left;
    }
</style>
"""