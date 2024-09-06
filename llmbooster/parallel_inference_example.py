import asyncio
from dotenv import load_dotenv
from parallel_inference import ParallelAIUtilities
from models import LLMPromptContext, LLMConfig, StructuredTool
from typing import Literal, List

async def main():
    load_dotenv()
    parallel_ai = ParallelAIUtilities()

    json_schema = {
        "type": "object",
        "properties": {
            "joke": {"type": "string"},
            "explanation": {"type": "string"}
        },
        "required": ["joke", "explanation"],
        "additionalProperties": False
    }

    structured_tool = StructuredTool(
        json_schema=json_schema,
        schema_name="tell_joke",
        schema_description="Generate a programmer joke with explanation"
    )

    # Create prompts for different JSON modes and tool usage
    def create_prompts(client, model, response_formats : List[Literal["json_beg", "text","json_object","structured_output","tool"]]= ["text"], count=5):
        prompts = []
        for response_format in response_formats:
            for i in range(count):
                prompts.append(
                    LLMPromptContext(
                        system_string="You are a helpful assistant that tells programmer jokes.",
                        new_message=f"Tell me a programmer joke about the number {i}.",
                        llm_config=LLMConfig(client=client, model=model, response_format=response_format,max_tokens=250),
                        structured_output=structured_tool,
                        use_schema_instruction=True,
                        
                    )
                )
        return prompts

    # OpenAI prompts
    openai_prompts = create_prompts("openai", "gpt-4o-2024-08-06",["json_beg", "text","json_object","structured_output","tool"])

    # Anthropic prompts
    anthropic_prompts = create_prompts("anthropic", "claude-3-5-sonnet-20240620", ["json_beg", "text","json_object","structured_output","tool"])

    # Run parallel completions
    print("Running parallel completions...")
    all_prompts = openai_prompts + anthropic_prompts
    # all_prompts=anthropic_prompts
    completion_results = await parallel_ai.run_parallel_ai_completion(all_prompts)

    # Print results
    for prompt, result in zip(all_prompts, completion_results):
        print(f"\nClient: {prompt.llm_config.client}, Response Format: {prompt.llm_config.response_format}")
        print(f"Prompt: {prompt.new_message}")
        if result.contains_object:
            print(f"Response (JSON): {result.json_object}")
        else:
            print(f"Response (Text): {result.str_content}")
        print(f"Usage: {result.usage}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())