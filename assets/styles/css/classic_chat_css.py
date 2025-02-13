USER_CHAT_STYLE_ST_V37 = """
<style>
    .stChatMessage:has(.chat-user) {
        // flex-direction: row-reverse;
        // width: fit-content;
        // max-width: 85%;
        // width: 85%;
        // margin-left: auto;
        // margin-right: 0;
        background-color: #E7F8FF;
        border-radius: 10px;
        padding: 20px;
    }
    .stChatMessage:has(.chat-user) .stCodeBlock {
        text-align: left;
    }
</style>
"""

USER_CHAT_STYLE_ST_V39 = """
<style>
    .stChatMessage:has(.chat-user) {
        // flex-direction: row-reverse;
        // width: fit-content;
        // max-width: 85%;
        // width: 85%;
        // margin-left: auto;
        // margin-right: 0;
        background-color: #E7F8FF;
        border-radius: 10px;
        padding: 20px;
    }
    .stChatMessage:has(.chat-user) .stCode {
        text-align: left;
    }
</style>
"""

ASSISTANT_CHAT_STYLE = """
<style>
    /* 使用更具体的选择器 */
    div[data-testid="stChatMessage"] {
        flex-direction: row;
        text-align: left;
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 20px;
    }
    
    /* 确保样式优先级 */
    div[data-testid="stChatMessage"]:has(.chat-assistant) {
        flex-direction: row;
        text-align: left;
        background-color: #F3F4F6 !important;
        border-radius: 10px !important;
        padding: 20px !important;
    }
</style>
"""