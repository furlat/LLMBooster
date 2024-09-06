# LLMBooster System Notes

## Existing Prompt Caching Implementation

After a closer examination of the codebase, it's clear that there's already a significant implementation of prompt caching for Anthropic's API. Let's break down the key components:

### 1. msg_dict_to_anthropic Function

This function in `utils.py` is crucial for converting message dictionaries to Anthropic's format with prompt caching support:

## Key Components

1. ParallelAIUtilities
   - Handles parallel processing of AI requests
   - Supports OpenAI and Anthropic APIs
   - Manages rate limits and token usage

2. LLMPromptContext
   - Represents a single AI request
   - Configurable for different LLM providers and response formats

3. LLMConfig
   - Configures LLM-specific settings (client, model, tokens, temperature, etc.)

4. StructuredTool
   - Defines JSON schemas for structured outputs
   - Supports both OpenAI and Anthropic formats

5. LLMOutput
   - Processes and standardizes AI responses
   - Handles different response types (string, JSON, tool outputs)

## Key Functionalities

1. Parallel Processing
   - Asynchronously processes multiple AI requests
   - Respects rate limits and token quotas

2. Multi-Provider Support
   - Works with OpenAI and Anthropic APIs
   - Adapts requests and responses to provider-specific formats

3. Flexible Response Formats
   - Supports text, JSON, and structured outputs
   - Handles tool-based responses

4. Error Handling and Retries
   - Implements retry logic for failed requests
   - Logs errors and provides detailed output

5. Usage Tracking
   - Monitors token usage and request counts
   - Provides usage statistics for each request

## Usage Example
