CUSTOM_RADIO_STYLE = """
<style>
    div[role="radiogroup"] label {
        display: flex;
        align-items: center;
        padding: 10px;
        margin: 5px 0;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        width: 100%;
        height: 100%;
        background-color: white;
        transition: all 0.2s ease;
    }
        
    div[role="radiogroup"] label:hover {
        background-color: #f5f5f5;
        border-color: #c0c0c0;
        cursor: pointer;
    }
        
    div[role="radiogroup"] label:has(input[type="radio"]:checked) {
        background-color: #e8f0fe;
        border-color: #1967d2;
    }
        
    div[role="radiogroup"] {
        max-height: 100%;
        overflow-y: auto;
    }
</style>
"""