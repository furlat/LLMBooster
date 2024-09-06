# Prompt caching through the Anthropic API

Prompt caching allows you to store and reuse context within your prompt. This makes it more practical to include additional information in your prompt—such as detailed instructions and example responses—which help improve every response Claude generates.

In addition, by fully leveraging prompt caching within your prompt, you can reduce latency by >2x and costs up to 90%. This can generate significant savings when building solutions that involve repetitive tasks around detailed book_content.

In this cookbook, we will demonstrate how to use prompt caching in a single turn and across a multi-turn conversation.

## Setup

First, let's set up our environment with the necessary imports and initializations:

```bash
%pip install anthropic bs4
```

```python
import anthropic
import time
import requests
from bs4 import BeautifulSoup

client = anthropic.Anthropic()
MODEL_NAME = "claude-3-5-sonnet-20240620"
```

Now let's fetch some text content to use in our examples. We'll use the text from Pride and Prejudice by Jane Austen which is around ~187,000 tokens long.

```python
def fetch_article_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Get text
    text = soup.get_text()
    
    # Break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # Drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return text

# Fetch the content of the article
book_url = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
book_content = fetch_article_content(book_url)

print(f"Fetched {len(book_content)} characters from the book.")
print("First 500 characters:")
print(book_content[:500])
```

## Example 1: Single turn

Let's demonstrate prompt caching with a large document, comparing the performance and cost between cached and non-cached API calls.

### Part 1: Non-cached API Call

First, let's make a non-cached API call. This will load the prompt into the cache so that our subsequent cached API calls can benefit from the prompt caching.

We will ask for a short output string to keep the output response time low since the benefit of prompt caching applies only to the input processing time.

```python
def make_non_cached_api_call():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "<book>" + book_content + "</book>",
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "type": "text",
                    "text": "What is the title of this book? Only output the title."
                }
            ]
        }
    ]

    start_time = time.time()
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=300,
        messages=messages,
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}

    )
    end_time = time.time()

    return response, end_time - start_time

non_cached_response, non_cached_time = make_non_cached_api_call()

print(f"Non-cached API call time: {non_cached_time:.2f} seconds")
print(f"Non-cached API call input tokens: {non_cached_response.usage.input_tokens}")
print(f"Non-cached API call output tokens: {non_cached_response.usage.output_tokens}")

print("\nSummary (non-cached):")
print(non_cached_response.content)
```

### Part 2: Cached API Call

Now, let's make a cached API call. I'll add in the `"cache_control": {"type": "ephemeral"}` attribute to the content object and add the `"prompt-caching-2024-07-31"` beta header to the request. This will enable prompt caching for this API call.

To keep the output latency constant, we will ask Claude the same question as before. Note that this question is not part of the cached content.

```python
def make_cached_api_call():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "<book>" + book_content + "</book>",
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "type": "text",
                    "text": "What is the title of this book? Only output the title."
                }
            ]
        }
    ]

    start_time = time.time()
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=300,
        messages=messages,
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
    )
    end_time = time.time()

    return response, end_time - start_time

cached_response, cached_time = make_cached_api_call()

print(f"Cached API call time: {cached_time:.2f} seconds")
print(f"Cached API call input tokens: {cached_response.usage.input_tokens}")
print(f"Cached API call output tokens: {cached_response.usage.output_tokens}")

print("\nSummary (cached):")
print(cached_response.content)
```

As you can see, the cached API call only took 3.64 seconds total compared to 21.44 seconds for the non-cached API call. This is a significant improvement in overall latency due to caching.

## Example 2: Multi-turn Conversation with Incremental Caching

Now, let's look at a multi-turn conversation where we add cache breakpoints as the conversation progresses.

```python
class ConversationHistory:
    def __init__(self):
        # Initialize an empty list to store conversation turns
        self.turns = []

    def add_turn_assistant(self, content):
        # Add an assistant's turn to the conversation history
        self.turns.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": content
                }
            ]
        })

    def add_turn_user(self, content):
        # Add a user's turn to the conversation history
        self.turns.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": content
                }
            ]
        })

    def get_turns(self):
        # Retrieve conversation turns with specific formatting
        result = []
        user_turns_processed = 0
        # Iterate through turns in reverse order
        for turn in reversed(self.turns):
            if turn["role"] == "user" and user_turns_processed < 2:
                # Add the last two user turns with ephemeral cache control
                result.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": turn["content"][0]["text"],
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                })
                user_turns_processed += 1
            else:
                # Add other turns as they are
                result.append(turn)
        # Return the turns in the original order
        return list(reversed(result))

# Initialize the conversation history
conversation_history = ConversationHistory()

# System message containing the book content
# Note: 'book_content' should be defined elsewhere in the code
system_message = f"<file_contents> {book_content} </file_contents>"

# Predefined questions for our simulation
questions = [
    "What is the title of this novel?",
    "Who are Mr. and Mrs. Bennet?",
    "What is Netherfield Park?",
    "What is the main theme of this novel?"
]

def simulate_conversation():
    for i, question in enumerate(questions, 1):
        print(f"\nTurn {i}:")
        print(f"User: {question}")
        
        # Add user input to conversation history
        conversation_history.add_turn_user(question)

        # Record the start time for performance measurement
        start_time = time.time()

        # Make an API call to the assistant
        response = client.messages.create(
            model=MODEL_NAME,
            extra_headers={
              "anthropic-beta": "prompt-caching-2024-07-31"
            },
            max_tokens=300,
            system=[
                {"type": "text", "text": system_message, "cache_control": {"type": "ephemeral"}},
            ],
            messages=conversation_history.get_turns(),
        )

        # Record the end time
        end_time = time.time()

        # Extract the assistant's reply
        assistant_reply = response.content[0].text
        print(f"Assistant: {assistant_reply}")

        # Print token usage information
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        input_tokens_cache_read = getattr(response.usage, 'cache_read_input_tokens', '---')
        input_tokens_cache_create = getattr(response.usage, 'cache_creation_input_tokens', '---')
        print(f"User input tokens: {input_tokens}")
        print(f"Output tokens: {output_tokens}")
        print(f"Input tokens (cache read): {input_tokens_cache_read}")
        print(f"Input tokens (cache write): {input_tokens_cache_create}")

        # Calculate and print the elapsed time
        elapsed_time = end_time - start_time

        # Calculate the percentage of input prompt cached
        total_input_tokens = input_tokens + (int(input

_tokens_cache_read) if input_tokens_cache_read != '---' else 0)
        percentage_cached = (int(input_tokens_cache_read) / total_input_tokens * 100 if input_tokens_cache_read != '---' and total_input_tokens > 0 else 0)

        print(f"{percentage_cached:.1f}% of input prompt cached ({total_input_tokens} tokens)")
        print(f"Time taken: {elapsed_time:.2f} seconds")

        # Add assistant's reply to conversation history
        conversation_history.add_turn_assistant(assistant_reply)

# Run the simulated conversation
simulate_conversation()
```
