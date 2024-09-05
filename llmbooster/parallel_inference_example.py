import asyncio
from dotenv import load_dotenv
from parallel_inference import ParallelAIUtilities
from models import LLMPromptContext, LLMConfig, StructuredTool

async def main():
    load_dotenv()
    parallel_ai = ParallelAIUtilities()

    # Create a list of LLMPromptContext objects for regular completions
    regular_prompts = [
        LLMPromptContext(
            system_string="You are a helpful assistant.",
            new_message=f"Tell me a short joke about the number {i}.",
            llm_config=LLMConfig(client="openai", model="gpt-4o-mini", max_tokens=50)
        ) for i in range(100)
    ] 
    regular_prompts = [
        LLMPromptContext(
            system_string="You are a helpful assistant.",
            new_message=f"Tell me a short joke about the letter {chr(65+i)}.",
            llm_config=LLMConfig(client="anthropic", model="claude-3-5-sonnet-20240620", max_tokens=50)
        ) for i in range(3)
    ]

    # Run parallel completions
    print("Running parallel completions...")
    completion_results = await parallel_ai.run_parallel_ai_completion(regular_prompts)

    # Print results
    for prompt, result in zip(regular_prompts, completion_results):
        print(f"Prompt: {prompt.new_message}")
        if result.str_content:
            print(f"Response: {result.str_content}\n")
        else:
            print(f"Error: {result.raw_result}\n")

   

if __name__ == "__main__":
    asyncio.run(main())