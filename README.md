# Project
After seeing Langchain's CUA workflow, I got inspired to try to automate applying to jobs by using this cua agent, scrapybara, and a local llm (so its much more affordable). 

Took me a while to find a LLM that works with my older graphics card, but I eventually found Qwen/Qwen2.5-VL-3B-Instruct --dtype=half. I barely had enough vram to run this. 

Following my own blended implementation of these 2 guides, [local multi-agent systems](https://www.youtube.com/watch?v=4oC1ZKa9-Hs&list=PL3YM9RACYstLK4jq08o94kHKx4HAZI9t8&index=1) and [CUA agents](https://www.youtube.com/watch?v=ndCFqT6xFQ4&list=PL3YM9RACYstLK4jq08o94kHKx4HAZI9t8&index=4), I set aside 5hrs to build a cua agent to help save time in the job application process. 

## What I Learned
The current tooling is quite rigid. Langgraph's cua_agent() only works with open_ai. I tried hard to fit my locally running qwen model into scrapybara's client.act(). Through all the model setup, I finally got it accepted! 
```
INFO 04-07 14:26:28 [model_runner.py:1146] Model loading took 7.1557 GiB and 225.087367 seconds

I only have 7.5GiB available for use

ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization` when initializing the engine.
```
Hahahah! Time to try a different model :D

This has been quite a fun little challenge! Times up. Back to applying regularly. 

### Get started
```
pip install
```
```
vllm serve Qwen/Qwen2.5-VL-3B-Instruct --dtype=half
```
```
python src/main.py
```

vllm serve hfl/Qwen2.5-VL-3B-Instruct-GPTQ-Int4 --dtype=half
