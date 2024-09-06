# LLMBooster

LLMBooster is a Python library designed for efficient, parallel processing of large language model (LLM) requests. It supports both OpenAI and Anthropic APIs, offering features like prompt caching, structured outputs, and various response formats.

## Features

- Parallel processing of multiple LLM requests
- Support for OpenAI and Anthropic APIs
- Rate limiting to respect API constraints
- Efficient handling of token usage
- Prompt caching for improved performance (Anthropic)
- Multiple response formats: text, JSON, structured output, and tool-based responses
- Customizable request configurations
- Error handling and request retrying
- Conversation history management with caching support

## Installation

To install LLMBooster, clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/llmbooster.git
cd llmbooster
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the root directory with your API keys and model configurations:

```env:.env
#OPENAI Credentials
OPENAI_KEY=sk-xxxx
OPENAI_MODEL=gpt-4o-mini
OPENAI_CONTEXT_LENGTH=128000

#Anthropic credentials
ANTHROPIC_API_KEY=sk-xxxx
ANTHROPIC_CONTEXT_LENGTH=200000
ANTHROPIC_MODEL=claude-3-5-sonnet-20240620
```

## Usage

### Basic Usage

Here's a simple example of how to use LLMBooster for parallel processing:

```python
import asyncio
from dotenv import load_dotenv
from parallel_inference import ParallelAIUtilities
from message_models import LLMPromptContext, LLMConfig

async def main():
    load_dotenv()
    parallel_ai = ParallelAIUtilities()

    prompts = [
        LLMPromptContext(
            system_string="You are a helpful assistant.",
            new_message=f"Tell me a short joke about the number {i}.",
            llm_config=LLMConfig(client="openai", model="gpt-4o-mini", max_tokens=50)
        ) for i in range(10)
    ]

    results = await parallel_ai.run_parallel_ai_completion(prompts)

    for prompt, result in zip(prompts, results):
        print(f"Prompt: {prompt.new_message}")
        print(f"Response: {result.str_content}\n")

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Usage

LLMBooster supports various configuration options and features:

1. **Mixed API Requests**: Process requests for both OpenAI and Anthropic in a single batch:

```python
prompts = [
    LLMPromptContext(
        system_string="You are a helpful assistant.",
        new_message=f"Tell me a short joke about the letter {chr(65+i)}.",
        llm_config=LLMConfig(client="openai", model="gpt-4o-mini", max_tokens=50)
    ) for i in range(5)
] + [
    LLMPromptContext(
        system_string="You are a helpful assistant.",
        new_message=f"Tell me a short joke about the number {i}.",
        llm_config=LLMConfig(client="anthropic", model="claude-3-5-sonnet-20240620", max_tokens=50)
    ) for i in range(5)
]

results = await parallel_ai.run_parallel_ai_completion(prompts)
```

2. **Structured Output**: Request structured JSON responses:

```python
from message_models import StructuredTool

json_schema = {
    "type": "object",
    "properties": {
        "joke": {"type": "string"},
        "explanation": {"type": "string"}
    },
    "required": ["joke", "explanation"]
}

structured_tool = StructuredTool(
    json_schema=json_schema,
    schema_name="tell_joke",
    schema_description="Generate a programmer joke with explanation"
)

prompt = LLMPromptContext(
    system_string="You are a helpful assistant.",
    new_message="Tell me a programmer joke.",
    llm_config=LLMConfig(client="openai", model="gpt-4o-mini", response_format="structured_output"),
    structured_output=structured_tool
)

result = await parallel_ai.run_parallel_ai_completion([prompt])
print(result[0].json_object)
```

3. **Rate Limiting**: LLMBooster automatically handles rate limiting based on the configuration:

```python
from parallel_inference import RequestLimits

oai_limits = RequestLimits(max_requests_per_minute=60, max_tokens_per_minute=40000)
anthropic_limits = RequestLimits(max_requests_per_minute=50, max_tokens_per_minute=100000, provider="anthropic")

parallel_ai = ParallelAIUtilities(oai_request_limits=oai_limits, anthropic_request_limits=anthropic_limits)
```

4. **Growing History with Cache (Anthropic)**: Utilize Anthropic's prompt caching feature for efficient conversation history management:

```python
from message_models import LLMPromptContext, LLMConfig
from parallel_inference import ParallelAIUtilities

async def test_growing_history_with_cache():
    parallel_ai = ParallelAIUtilities()
    
    base_prompt = LLMPromptContext(
        system_string="You are a helpful AI assistant. Here's some context about prompt caching: <file_contents>...</file_contents>",
        new_message="Let's have a conversation about prompt caching. What is it and how does it work?",
        llm_config=LLMConfig(client="anthropic", model="claude-3-sonnet-20240307", use_cache=True, max_tokens=1000)
    )

    # First turn
    result1 = await parallel_ai.run_parallel_ai_completion([base_prompt])
    print("Turn 1 (System content cached)")
    print(f"Response: {result1[0].str_content}")
    print(f"Usage: {result1[0].usage}")

    # Second turn
    prompt2 = base_prompt.add_chat_turn_history(result1[0])
    prompt2.new_message = "How does prompt caching improve performance?"
    result2 = await parallel_ai.run_parallel_ai_completion([prompt2])
    print("\nTurn 2 (System content + partial history cached)")
    print(f"Response: {result2[0].str_content}")
    print(f"Usage: {result2[0].usage}")

    # Third turn
    prompt3 = prompt2.add_chat_turn_history(result2[0])
    prompt3.new_message = "What are some best practices for using prompt caching?"
    result3 = await parallel_ai.run_parallel_ai_completion([prompt3])
    print("\nTurn 3 (System content + more history cached)")
    print(f"Response: {result3[0].str_content}")
    print(f"Usage: {result3[0].usage}")

asyncio.run(test_growing_history_with_cache())
```

This example demonstrates how to use the `add_chat_turn_history` method to build a conversation while leveraging Anthropic's prompt caching feature. The `use_cache=True` parameter in the `LLMConfig` enables caching, which can significantly reduce token usage and improve response times in multi-turn conversations.

## Key Components

- `ParallelAIUtilities`: Main class for handling parallel LLM requests.
- `LLMPromptContext`: Represents a single LLM request, including prompt, configuration, and optional structured output settings.
- `LLMConfig`: Defines the configuration for an LLM request.
- `StructuredTool`: Defines the schema for structured output responses.

## Error Handling and Retrying

LLMBooster includes built-in error handling and request retrying. Failed requests are automatically retried up to a specified number of attempts, with exponential backoff.

## Performance Considerations

- Asynchronous processing maximizes throughput.
- Requests are batched by client (OpenAI or Anthropic) for efficient processing.
- Token usage is tracked to ensure compliance with rate limits.
- Anthropic's prompt caching feature can significantly improve performance in multi-turn conversations.

## Contributing

Contributions to LLMBooster are welcome! Please submit pull requests or open issues on the GitHub repository.

## License

LLMBooster is released under the MIT License. See the LICENSE file for details.

```1:22:LICENSE
MIT License

Copyright (c) 2024 Cynde

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

This README provides a comprehensive overview of the LLMBooster library, focusing on its parallel processing capabilities and usage examples. It covers the main features, installation, configuration, basic and advanced usage, key components, error handling, and performance considerations.