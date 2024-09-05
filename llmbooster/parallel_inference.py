import asyncio
import json
from typing import List, Dict, Any, Tuple, Optional, Literal
from pydantic import BaseModel, Field
from models import LLMPromptContext, LLMOutput, LLMConfig
from oai_parallel import process_api_requests_from_file, OAIApiFromFileConfig
import os
from dotenv import load_dotenv
#Import time
import time

class RequestLimits(BaseModel):
    max_requests_per_minute: int = Field(default=50,description="The maximum number of requests per minute for the API")
    max_tokens_per_minute: int = Field(default=100000,description="The maximum number of tokens per minute for the API")
    provider: Literal["openai", "anthropic"] = Field(default="openai",description="The provider of the API")

class ParallelAIUtilities:
    def __init__(self, oai_request_limits: RequestLimits = RequestLimits(), anthropic_request_limits: RequestLimits = RequestLimits(provider="anthropic")):
        load_dotenv()
        self.openai_key = os.getenv("OPENAI_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.oai_request_limits = oai_request_limits
        self.anthropic_request_limits = anthropic_request_limits


    async def run_parallel_ai_completion(self, prompts: List[LLMPromptContext]) -> List[LLMOutput]:
        openai_prompts = [p for p in prompts if p.llm_config.client == "openai"]
        anthropic_prompts = [p for p in prompts if p.llm_config.client == "anthropic"]

        results = []
        if openai_prompts:
            results.extend(await self._run_openai_completion(openai_prompts))
        if anthropic_prompts:
            results.extend(await self._run_anthropic_completion(anthropic_prompts))

        return results

    async def _run_openai_completion(self, prompts: List[LLMPromptContext]) -> List[LLMOutput]:
        #human readable timestamp
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        requests_file = self._prepare_requests_file(prompts, "openai")
        results_file = f'openai_results_{timestamp}.jsonl'
        config = self._create_config(prompts[0], requests_file, results_file)
        if config:
            await process_api_requests_from_file(config)
            return self._parse_results_file(results_file, prompts)
        return []

    async def _run_anthropic_completion(self, prompts: List[LLMPromptContext]) -> List[LLMOutput]:
        requests_file = self._prepare_requests_file(prompts, "anthropic")
        #human readable timestamp
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        results_file = f'anthropic_results_{timestamp}.jsonl'
        config = self._create_config(prompts[0], requests_file, results_file)
        if config:
            await process_api_requests_from_file(config)
            return self._parse_results_file(results_file, prompts)
        return []

    def _prepare_requests_file(self, prompts: List[LLMPromptContext], client: str) -> str:
        #human readable timestamp
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        requests = []
        for prompt in prompts:
            request = self._convert_prompt_to_request(prompt, client)
            if request:
                requests.append(request)
        
        filename = f'{client}_requests_{timestamp}.jsonl'

        with open(filename, 'w') as f:
            for request in requests:
                json.dump(request, f)
                f.write('\n')
        return filename

    def _convert_prompt_to_request(self, prompt: LLMPromptContext, client: str) -> Optional[Dict[str, Any]]:
        if client == "openai":
            messages = prompt.oai_messages
            return {
                "model": prompt.llm_config.model,
                "messages": messages,
                "max_tokens": prompt.llm_config.max_tokens,
                "temperature": prompt.llm_config.temperature,
            }
        elif client == "anthropic":
            system_content, messages = prompt.anthropic_messages
            return {
                "model": prompt.llm_config.model,
                "max_tokens": prompt.llm_config.max_tokens,
                "temperature": prompt.llm_config.temperature,
                "messages": messages,
                "system": system_content if system_content else None
            }
        return None

    def _create_config(self, prompt: LLMPromptContext, requests_file: str, results_file: str) -> Optional[OAIApiFromFileConfig]:
        
        if prompt.llm_config.client == "openai" and self.openai_key:
            return OAIApiFromFileConfig(
                requests_filepath=requests_file,
                save_filepath=results_file,
                request_url="https://api.openai.com/v1/chat/completions",
                api_key=self.openai_key,
                max_requests_per_minute=self.oai_request_limits.max_requests_per_minute,
                max_tokens_per_minute=self.oai_request_limits.max_tokens_per_minute,
                token_encoding_name="cl100k_base",
                max_attempts=5,
                logging_level=20,
            )
        elif prompt.llm_config.client == "anthropic" and self.anthropic_key:
            return OAIApiFromFileConfig(
                requests_filepath=requests_file,
                save_filepath=results_file,
                request_url="https://api.anthropic.com/v1/messages",
                api_key=self.anthropic_key,
                max_requests_per_minute=self.anthropic_request_limits.max_requests_per_minute,
                max_tokens_per_minute=self.anthropic_request_limits.max_tokens_per_minute,
                token_encoding_name="cl100k_base",
                max_attempts=5,
                logging_level=20,
            )
        return None

    def _parse_results_file(self, filepath: str, original_prompts: List[LLMPromptContext]) -> List[LLMOutput]:
        results = []
        with open(filepath, 'r') as f:
            for line, original_prompt in zip(f, original_prompts):
                try:
                    result = json.loads(line)
                    print(f"Debug: Raw result from file: {result}")  # Debug print
                    llm_output = self._convert_result_to_llm_output(result, original_prompt)
                    results.append(llm_output)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON: {line}")
                    results.append(LLMOutput(raw_result={"error": "JSON decode error"}, completion_kwargs={}))
                except Exception as e:
                    print(f"Error processing result: {e}")
                    results.append(LLMOutput(raw_result={"error": str(e)}, completion_kwargs={}))
        return results

    def _convert_result_to_llm_output(self, result: List[Dict[str, Any]], original_prompt: LLMPromptContext) -> LLMOutput:
        request_data, response_data = result
        print(f"Debug: Converting result to LLMOutput")
        print(f"Debug: Request data: {request_data}")
        print(f"Debug: Response data: {response_data}")
        
        if original_prompt.llm_config.client == "openai":
            return LLMOutput(raw_result=response_data, completion_kwargs=request_data)
        elif original_prompt.llm_config.client == "anthropic":
            # Convert Anthropic response format to match LLMOutput expectations
            return LLMOutput(raw_result=response_data, completion_kwargs=request_data)
        else:
            print(f"Debug: Unexpected client type: {original_prompt.llm_config.client}")
            return LLMOutput(raw_result={"error": "Unexpected client type"}, completion_kwargs=request_data)

    async def run_parallel_ai_tool_completion(self, prompts: List[LLMPromptContext]) -> List[LLMOutput]:
        # Implement this method similar to run_parallel_ai_completion
        # but with tool-specific logic
        return []

