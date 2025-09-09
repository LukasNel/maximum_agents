Previous Example of Blog Post
Apollo.io

2084

1
1

Deep Reasoning With Tools: Toolcalling in the reasoning trace
A framework I've been working on for the past while.
Lukas Nel
Jun 01, 2025

So, as I was building out deepstock, I realized that the paradigm of providing a fixed set of information like quarterly growth, or stock prices over the last week or so was flawed, as it was essentially static - the AI could never learn the essential operation of seeking out information itself.

At this time too, I was working a lot with smolagents, the huggingface agent framework, which expresses tools as python functions, which are called by the LLM writing a codeblock executed by a local python interpreter.

Thus, I realized: if you could have the reasoning model write codeblocks in its reasoning trace, then you could use the smolagents python interpreter to execute the code inline and append the output. This would allow the model to do search and research in its reasoning trace, which if it could be combined with GRPO, would allow you to train both the "search" and "reasoning" portions simultaneously, with tools allowing it to search through the stock price and financial data.

Thus I built deepreasoningwithtools.

What This System Does: A Simple Example
To understand what deepreasoningwithtools accomplishes, consider this scenario: You ask an AI "What was Apple's stock price yesterday compared to last week's average?"

With traditional AI systems, the model would either:

Give you a generic response about not having real-time data, or

Use pre-loaded static information that's likely outdated

With deepreasoningwithtools, here's what happens:

Step 1: The AI starts reasoning and realizes it needs current stock data. It then writes some python code in

<think>
I need to find Apple's current stock price and calculate last week's average.
Let me fetch this data:

```python
# Get Apple's stock data
apple_stock = get_stock_price('AAPL', days=8)
yesterday_price = apple_stock[-1]
last_week_avg = sum(apple_stock[:-1]) / 7
Step 2: The system detects this code block, pauses the AI's output, and executes the Python code using real stock market APIs

Step 3: The execution results get inserted back into the AI's reasoning:

Output: yesterday_price = $150.25, last_week_avg = $148.73
Step 4: The AI continues its reasoning with the actual data:

Based on the data I retrieved, Apple's stock price yesterday was $150.25, 
which is $1.52 higher than last week's average of $148.73, representing 
a 1.02% increase. I need to tell this to the user to the user.
</think>
<answer>
 $150.25
</answer>
The key insight is that the AI can now seek out information during its thinking process, rather than being limited to pre-existing knowledge. It writes code to solve problems, gets real results, and incorporates those results into its reasoning - all in one continuous thought process. In addition, as this is a standard reasoning trace, you can take it and use techniques like GRPO to train the AI to improve its use of tools over time, allowing applications of RL models to more complex domains like accounting or searching through SQL databases.

Technical Implementation
This framework is meant to be a basic framework that essentially just handles the issue of handing off code in the reasoning trace to the local python interpreter. It uses tools defined in the same way as smolagents.

from deepreasoningwithtools.toolcaller import ToolCaller
from deepreasoningwithtools.samplers import LiteLLMSampler
from deepreasoningwithtools.tools.yfinance_tools import StockPriceTool
from datetime import datetime

# Initialize the tool caller with a LiteLLM model
toolcaller = ToolCaller(
    sampler=LiteLLMSampler(model_name="gpt-3.5-turbo"),
    authorized_imports=["pandas"]
)

# Add tools
tools = [StockPriceTool(cutoff_date=datetime.now().strftime("%Y-%m-%d"))]

# Generate responses
async for output in toolcaller.generate(
    user_prompt="What was Apple's stock price last week?",
    system_prompt="You are a helpful AI assistant...",
    tools=tools
):
    print(output, end="")
How it works is that the toolcaller is setup with a "sampler", which is a connection to a specific LLM that returns a streaming output given a list of messages in openai format - in deepreasoningwithtools, I have two, one using local LLMs with VLLMSampler, and one using an API service with LiteLLMSampler - and a list of authorized imports which can be used in the inline tool blocks. Then you call it with your user and system prompt, and it will automatically handle the offloading to the local python interpreter for you. Specifically, it checks the output stream from the sampler, and if it detects a complete code block, it stops the stream, and hands it off to the python interpreter, appending the output, and then restarting the stream. This is powerful, as it allows the model to do multistep complex search processes.

There are two tricks I use internally in the toolcaller to allow for this to work:

Firstly, there is a feature in python, where you can "send" info to a generator, as in:

def my_generator():
    while True:
        received = yield
        print(f"Got: {received}")

gen = my_generator()
next(gen)  # Prime the generator
gen.send("hello")  # Output: Got: hello
gen.send(42)    
This allows me to interrupt streaming on the detection of a code block.

Secondly, if you send a list of messages with the last message being an "assistant" message, generally most LLM services will continue the last message. This is useful for restarting streams, and in other contexts, for getting LLMs to output proper JSON or other formats. Next time you want to have your LLM generate json, append a message that looks like {"role":"assistant", "content":"Here is the JSON:"} and see how much better that works for structured output.

Testing and Examples
To install it, run:

 pip3 install deepreasoningwithtools[vllm,litellm]
To clone it

git clone https://github.com/LukasNel/TRLLukas/tree/master
Then navigate to the deeptools folder.

To run the test suite for the service, run:

modal setup
modal run run_modal_tests.py
This will scaffold the VLLM sampler and result in output showing the system in action. For example, when asked to compare Tesla and Microsoft's financial performance over the last quarter, the model can fetch real financial data, perform calculations, and provide analysis based on current information.


Extending the Framework
To define a new tool you do the following:

from smolagents import Tool

class CustomTool(Tool):
    name = "custom_tool"
    description = "Description of what the tool does"
    inputs = {
        "param1": {
            "type": "string",
            "description": "Description of param1"
        }
    }
    output_type = "object"

    def forward(self, param1: str):
        # Implement tool logic here
        return result
To define a new sampler, the following:

from deepreasoningwithtools.samplers.abstract import AbstractSampler
from typing import AsyncGenerator

class CustomSampler(AbstractSampler):
    async def sample(self, messages: list[dict[str, str]]) -> AsyncGenerator[str, None]:
        # messages will be a list of dictionaries following the OpenAI format
        # Implement your sampling logic here
        yield "response"
Local Model Support with VLLM
The most interesting component to me is the VLLM sampler, as this allows you to use a local LLM to run this. I have it setup as a VLLM server that handles the actual generation, with the VLLMSampler running a client that calls into the server and does stream generation. I based it on the implementation of the VLLM server and client at trl.

Stream generation was a bit of a hack - I used this colab I found which shows an example of patching the VLLM inference code to allow for output streaming.

Since I wanted to use a GPU for streaming and the other GPU for training, I also had to modify the vllm server to support that by manually changing the world size to be the same as the tensor parallel size you pass in as a parameter - by default its that + 1, which if you only give it access to a single GPU, leads to it spawning two processes, and a lot of bugs from that. If you run into multiple GPU bugs, double check your world size people! It might be a subtle bug there somewhere.

Model Training Integration
In the future, I want to use deepreasoningwithtools to help with training models, and for that I wanted to extend the VLLMSampler to allow for setting model weights as well. The way that this was done in the TRL, and the way I adopted, was to use a PyNcclCommunicator, which allows you to broadcast weights to a set of GPUs from a client. which controls what those weights are. There was a minor complication in that if you have 1 GPU, you have to make sure that you set the client as master properly, in the following way:

For client:

store = TCPStore(
    host_name=self.host,
    port=self.group_port,
    world_size=world_size,
    is_master=False,
    timeout=datetime.timedelta(seconds=300),
)
pg = StatelessProcessGroup(
    rank=self.rank,
    world_size=world_size,
    store=store,
    data_expiration_seconds=3600)
self.pynccl_comm = PyNcclCommunicator(pg, device=0)
For server:

store = TCPStore(
    host_name=self.host,
    port=self.group_port,
    world_size=world_size,
    is_master=True,
    timeout=datetime.timedelta(seconds=300),
)
pg = StatelessProcessGroup(
    rank=self.rank,
    world_size=world_size,
    store=store,
    data_expiration_seconds=3600)
self.pynccl_comm = PyNcclCommunicator(pg, device=0)
Honestly just putting this here because figuring this out took me an entire Saturday.

In addition, you want to make sure that your world size is correctly set.

The upshot is that you can go:

cast(VLLMSampler, self.vllm_toolcaller.sampler).client.update_model_params(self.model)
And it will update the toolcaller with the new model params, allowing for training.

Future Work
Now that I have this framework built, I need to apply it to training - however, the biggest problem I'm running into is that after generation, when I run the gradient calculation across the entire output on my GPU, it dies with a CUDA OOM error. So now that I've figured out how to offload generation, now I need to figure out how to do the calculation of gradients in a way that allows for offloading most of the weights and only calculating a part at a time. I'm still figuring out how to use the deepspeed and similar frameworks to allow me to do this

But even as this is, the system can do complex analysis, like comparing Microsoft vs Tesla stock performance, or more involved financial research that requires multiple data sources and calculations.

The framework represents a step toward AI systems that can actively seek information and use tools during their reasoning process, rather than being constrained by static knowledge cutoffs, in a format that can be used to finetune the models to improve their outputs.

Thanks for reading 2084! Subscribe for free to receive new posts and support my work.

2 Likes
Discussion about this post
Write a comment...
Gail
01 Jun

So interesting, the live data analysis makes for huge applicational uses. Well done!

Liked (2)
Reply
Share
Maxmustermann
17 Jul

Any new results? :) this project sounds amazing :) I am excited.

Like
Reply
Share
1 more comment...

2084: MarcRandbot: Speech Synthesis with Mamba
Using Mamba to do speech synthesis.
Jan 3, 2024 • Lukas Nel
8
2

2084: Deepstock - can you train deepseek to do stock trading?
A short guide to using deepseek to do stocktrading.
Feb 3 • Lukas Nel
5
26

2084: A Dive into DeepSeek
While deepstock is training, an intro on DeepSeek V3
Feb 11 • Lisanthan
6
2

2084: DeepStock V2 - Predicting the market better than 50%
Training AI to Predict the Market with GPRO
Apr 16 • Lukas Nel
5
6

2084: Is ChatGPT self-aware?
Some interesting things I got it to generate
Dec 13, 2022 • Lukas Nel
5
2

2084: The soul of the LLMs
All the LLMs are converging to the same representation
Jul 1 • Lukas Nel
3
2

2084: BitNet overhyped? Training BitNet Mamba on the Tiny Shakespeare dataset
Applying single bits to solve my problems.
Jun 4, 2024 • Lukas Nel
2

2084: Diabetes Seek
Predicting diabetes with DeepSeek from a patient's medical history.
Feb 24 • Lukas Nel
3
1

Bureaucracy is all you need.
A manifesto for paying more attention to how AI agents communicate.
Jan 18 • Lukas Nel
2
1

2084: Can you predict the view count of a video from its thumbnail?
ThumbnailViewMamba or How I learned to stop worrying and love the bot.
Feb 25, 2024 • Lukas Nel
3

© 2025 Lukas Nel
Privacy ∙ Terms ∙ Collection notice
Start writing
Get the app
Substack is the home for great culture

Blog Post

1. An examination of a variety of different ways to call tools from an LLM.

1a. You have the classic JSON tool calling pattern, where you have the LLM generate a JSON object that contains the name of the tool to call and the arguments to call it with,
execute the tool, and then return the result. This is the most straightforward way to call tools from an LLM, and has the advantage of being easy to understand and implement.
However, it has the disadvantage that it can't express more complex logic, like "if the user's question is about the weather, call the weather tool, otherwise call the news tool."
In a lot of cases, you have to embed a lot of your own logic in the LLM's prompt and have it be executed "vibaciously", which can be error prone.

1b. You have the code execution pattern, where you have the LLM generate a code block that contains the logic to call the tool, with the tools being expressed as python functions,
in my experience, this is the most powerful way to call tools from an LLM, and has the advantage of being able to express more complex logic, as well as being able to use the full power of python.

However, a lot of times, these tools are still "functional" python functions, and not objects that have methods and attributes - this has the disadvantage that the LLM has to build up more complex objects by itself, in a single code block,
which means that its hard to do stuff like write word documents, since they require doing longer and more complex operations.

So when I built my maximum agents framework, inspired by smolagents and their code executor that "maintains state between steps",
and looking at various pypi libraries that have nice object-oriented interfaces, like python-pptx and docx,
I found that a "client based" approach, where you have the LLM instantiate an object, and then have it call methods on that object,
over multiple steps to build up more complex objects which you finally return to the user, was the most powerful way to call tools from an LLM.

This allows you to do stuff like write word documents with multiple steps, of many pages, with embedded charts and images, and other complex objects.

This led me to build my maximum agents framework, as well as build a small research agent framework. 

<explanation of how maximum agents framework works, as well as how it's used in the research agent,an explanation of python-pptx and python docx and the specific library I used to show the frontend, as well as an explanation of the idea of the workspaces.>
