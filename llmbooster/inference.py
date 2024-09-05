import os
from dotenv import load_dotenv

#import together
from openai import OpenAI
from anthropic import Anthropic

from anthropic.types.message_create_params import ToolChoiceToolChoiceTool

from openai.types.chat import ChatCompletion

from typing import Dict, Any

from models import LLMOutput, LLMPromptContext






class AIUtilities:
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        
        # openai credentials
        self.openai_key = os.getenv("OPENAI_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL")
        # anthropic credentials
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL")
    

    def run_ai_completion(self, prompt: LLMPromptContext):
        if prompt.llm_config.client == "openai":
            assert self.openai_key is not None, "OpenAI API key is not set"
            client = OpenAI(api_key=self.openai_key)
            return self.run_openai_completion(client, prompt)
        
        elif prompt.llm_config.client == "anthropic":
            assert self.anthropic_api_key is not None, "Anthropic API key is not set"
            anthropic = Anthropic(api_key=self.anthropic_api_key)
            return self.run_anthropic_completion(anthropic, prompt)
        
        else:
            return "Invalid AI vendor"
    
    def run_openai_completion(self, client: OpenAI, prompt: LLMPromptContext):
        try:
            

            completion_kwargs: Dict[str, Any] = {
                "model": prompt.llm_config.model or self.openai_model,
                "messages": prompt.oai_messages,
                "max_tokens": prompt.llm_config.max_tokens,
                "temperature": prompt.llm_config.temperature,
                "response_format": prompt.oai_response_format,
            }
            

            response: ChatCompletion = client.chat.completions.create(**completion_kwargs)
            return LLMOutput(raw_result=response, completion_kwargs=completion_kwargs)
        except Exception as e:
            return LLMOutput(raw_result=f"Error: {str(e)}", completion_kwargs=completion_kwargs)

    def run_anthropic_completion(self, anthropic: Anthropic, prompt: LLMPromptContext):
        
        system_content, anthropic_messages = prompt.anthropic_messages
        model = prompt.llm_config.model or self.anthropic_model

        try:
            assert model is not None, "Model is not set"
            completion_kwargs = {
                "model": model,
                "messages": anthropic_messages,
                "max_tokens": prompt.llm_config.max_tokens,
                "temperature": prompt.llm_config.temperature,
                "system": system_content,
            }
            

            response = anthropic.beta.prompt_caching.messages.create(**completion_kwargs)
            return LLMOutput(raw_result=response, completion_kwargs=completion_kwargs)
        except Exception as e:
            return LLMOutput(raw_result=str(e), completion_kwargs=completion_kwargs)


        
    def run_ai_tool_completion(
        self,
        prompt: LLMPromptContext,
        
    ):
        if prompt.llm_config.client == "openai":
            return self.run_openai_tool_completion(prompt)
        elif prompt.llm_config.client == "anthropic":
            return self.run_anthropic_tool_completion(prompt)
        else:
            raise ValueError("Unsupported client for tool completion")

    def run_openai_tool_completion(
        self,
        prompt: LLMPromptContext,
        
    ):
        client = OpenAI(api_key=self.openai_key)
        
        
        try:
            assert prompt.structured_output is not None, "Tool is not set"
            
            completion_kwargs = {
                "model":  prompt.llm_config.model or self.openai_model,
                "messages": prompt.oai_messages,
                "max_tokens": prompt.llm_config.max_tokens,
                "temperature": prompt.llm_config.temperature,
            }

            tool = prompt.get_tool()
            if tool:
                completion_kwargs["tools"] = [tool]
                completion_kwargs["tool_choice"] = {"type": "function", "function": {"name": prompt.structured_output.schema_name}}
            
            response : ChatCompletion = client.chat.completions.create(**completion_kwargs)

            return LLMOutput(raw_result=response, completion_kwargs=completion_kwargs)
        except Exception as e:
            return LLMOutput(raw_result=str(e), completion_kwargs=completion_kwargs)

    def run_anthropic_tool_completion(
        self,
        prompt: LLMPromptContext,
    ):  
        system_content , anthropic_messages = prompt.anthropic_messages
        client = Anthropic(api_key=self.anthropic_api_key)
        model = prompt.llm_config.model or self.anthropic_model

        try:
            assert model is not None, "Model is not set"
            
            completion_kwargs = {
                "model": model,
                "messages": anthropic_messages,
                "max_tokens": prompt.llm_config.max_tokens,
                "temperature": prompt.llm_config.temperature,
                "system": system_content,
            }

            tool = prompt.get_tool()
            if tool and prompt.structured_output is not None:
                completion_kwargs["tools"] = [tool]
                completion_kwargs["tool_choice"] = ToolChoiceToolChoiceTool(name=prompt.structured_output.schema_name, type="tool")
            response = client.beta.prompt_caching.messages.create(**completion_kwargs)
            return LLMOutput(raw_result=response, completion_kwargs=completion_kwargs)
        except Exception as e:
            return LLMOutput(raw_result=str(e), completion_kwargs=completion_kwargs)
