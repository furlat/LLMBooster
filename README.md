# LLMBooster

LLMBooster is a Python library designed for efficient, parallel processing of large language model (LLM) requests. It supports both OpenAI and Anthropic APIs, allowing for high-throughput, rate-limited interactions with these services.

## Features

- Parallel processing of multiple LLM requests
- Support for OpenAI and Anthropic APIs
- Rate limiting to respect API constraints
- Efficient handling of token usage
- Customizable request configurations
- Error handling and request retrying
- Structured output parsing

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
ANTHROPIC_MODEL=claude-3-5-sonnet-2024062

```

## Usage

### Basic Usage

Here's a simple example of how to use LLMBooster for parallel processing:

```python
import asyncio
from dotenv import load_dotenv
from parallel_inference import ParallelAIUtilities
from models import LLMPromptContext, LLMConfig

async def main():
    load_dotenv()
    parallel_ai = ParallelAIUtilities()

    prompts = [
        LLMPromptContext(
            system_string="You are a helpful assistant.",
            new_message=f"Tell me a short joke about the number {i}.",
            llm_config=LLMConfig(client="openai", model="gpt-4", max_tokens=50)
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

1. **Mixed API Requests**: You can process requests for both OpenAI and Anthropic in a single batch:

```python
prompts = [
    LLMPromptContext(
        system_string="You are a helpful assistant.",
        new_message=f"Tell me a short joke about the letter {chr(65+i)}.",
        llm_config=LLMConfig(client="openai", model="gpt-4", max_tokens=50)
    ) for i in range(5)
] + [
    LLMPromptContext(
        system_string="You are a helpful assistant.",
        new_message=f"Tell me a short joke about the number {i}.",
        llm_config=LLMConfig(client="anthropic", model="claude-3-sonnet", max_tokens=50)
    ) for i in range(5)
]

results = await parallel_ai.run_parallel_ai_completion(prompts)
```

2. **Structured Output**: You can request structured JSON responses:

```python
from models import StructuredTool

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
    llm_config=LLMConfig(client="openai", model="gpt-4", response_format="structured_output"),
    structured_output=structured_tool
)

result = await parallel_ai.run_parallel_ai_completion([prompt])
print(result[0].json_object)
```

3. **Rate Limiting**: LLMBooster automatically handles rate limiting based on the configuration:

```python
from models import RequestLimits

oai_limits = RequestLimits(max_requests_per_minute=60, max_tokens_per_minute=40000)
anthropic_limits = RequestLimits(max_requests_per_minute=50, max_tokens_per_minute=100000, provider="anthropic")

parallel_ai = ParallelAIUtilities(oai_request_limits=oai_limits, anthropic_request_limits=anthropic_limits)
```

## Key Components

### ParallelAIUtilities

The main class for handling parallel LLM requests. It manages the processing of requests for both OpenAI and Anthropic APIs.


```17:24:llmbooster/parallel_inference.py
class ParallelAIUtilities:
    def __init__(self, oai_request_limits: RequestLimits = RequestLimits(), anthropic_request_limits: RequestLimits = RequestLimits(provider="anthropic")):
        load_dotenv()
        self.openai_key = os.getenv("OPENAI_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.oai_request_limits = oai_request_limits
        self.anthropic_request_limits = anthropic_request_limits

```


### LLMPromptContext

Represents a single LLM request, including the prompt, configuration, and optional structured output settings.


```97:106:llmbooster/models.py
class LLMPromptContext(BaseModel):
    system_string: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None
    new_message: str
    prefill: str = Field(default="Here's the valid JSON object response:```json", description="prefill assistant response with an instruction")
    postfill: str = Field(default="\n\nPlease provide your response in JSON format.", description="postfill user response with an instruction")
    structured_output : Optional[StructuredTool] = None
    use_schema_instruction: bool = False
    llm_config: LLMConfig

```


### LLMConfig

Defines the configuration for an LLM request, including the client, model, and response format.


```86:91:llmbooster/models.py
class LLMConfig(BaseModel):
    client: Literal["openai", "azure_openai", "anthropic", "vllm"]
    model: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0
    response_format: Literal["json_beg", "text","json_object","structured_output","tool"] = "text"
```


### StructuredTool

Defines the schema for structured output responses.


```36:52:llmbooster/models.py
class StructuredTool(BaseModel):
    """ Supported type by OpenAI Structured Output:
    String, Number, Boolean, Integer, Object, Array, Enum, anyOf
    Root must be Object, not anyOf
    Not supported by OpenAI Structured Output: 
    For strings: minLength, maxLength, pattern, format
    For numbers: minimum, maximum, multipleOf
    For objects: patternProperties, unevaluatedProperties, propertyNames, minProperties, maxProperties
    For arrays: unevaluatedItems, contains, minContains, maxContains, minItems, maxItems, uniqueItems
    oai_reference: https://platform.openai.com/docs/guides/structured-outputs/how-to-use """

    json_schema: Optional[Dict[str, Any]] = None
    schema_name: str = Field(default = "generate_structured_output")
    schema_description: str = Field(default ="Generate a structured output based on the provided JSON schema.")
    instruction_string: str = Field(default = "Please follow this JSON schema for your response:")
    strict_schema: bool = True

```


## Error Handling and Retrying

LLMBooster includes built-in error handling and request retrying. Failed requests are automatically retried up to a specified number of attempts, with exponential backoff.

## Performance Considerations

- LLMBooster uses asynchronous processing to maximize throughput.
- Requests are batched by client (OpenAI or Anthropic) for efficient processing.
- Token usage is tracked to ensure compliance with rate limits.

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