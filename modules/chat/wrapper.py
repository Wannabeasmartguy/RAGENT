from openai import Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk


def stream_with_reasoning_content_wrapper(stream: Stream[ChatCompletionChunk]):
    """Wrap a generator that yields ChatCompletionChunk objects into a function that yields reasoning_content and content."""
    # 初始化变量
    REASONING_BEGIN_MARKER = "<think>"
    REASONING_END_MARKER = "</think>"
    reasoning_content = ""
    content = ""
    reasoning_started = False  # 标记是否已经开始输出 reasoning_content

    for chunk in stream:
        delta = chunk.choices[0].delta
        new_reasoning_content = ""
        new_content = ""

        try:
            if delta.reasoning_content:
                new_reasoning_content = delta.reasoning_content
        except AttributeError:
            pass

        try:
            if delta.content:
                new_content = delta.content
        except AttributeError:
            pass

        # 更新累积内容
        reasoning_content += new_reasoning_content
        content += new_content

        # 优先输出 reasoning_content
        if new_reasoning_content:
            if not reasoning_started:
                # 如果是第一次输出 reasoning_content，添加开始标志
                yield REASONING_BEGIN_MARKER
                reasoning_started = True
            yield new_reasoning_content
        else:
            # 如果没有新的 reasoning_content，则输出 content
            if new_content:
                # 如果之前有 reasoning_content 输出，先添加结束标志
                if reasoning_started:
                    yield REASONING_END_MARKER
                    reasoning_started = False
                yield new_content

    # 如果循环结束后仍有未结束的 reasoning_content，添加结束标志
    if reasoning_started:
        yield REASONING_END_MARKER