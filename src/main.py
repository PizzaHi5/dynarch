#imports
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langchain.chat_models import init_chat_model
from langgraph.types import interrupt, Command
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.message import add_messages
from langgraph_supervisor import create_supervisor
from langgraph_swarm import create_swarm
from langgraph.pregel import Pregel
from langgraph.graph import StateGraph, START, END, MessagesState

#  from langgraph_codeact import create_codeact
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from open_deep_research.graph import builder # https://github.com/langchain-ai/open_deep_research

from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

# from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from src.models.openai import get_openai_model, ModelTier
from src.tools.search import get_tavily_search

from typing import Annotated, TypedDict, Any, Literal
from pydantic import Field, BaseModel
import uuid, asyncio, builtins, contextlib, io

import os
from dotenv import load_dotenv
load_dotenv()

class AgentNode:
    def __init__(self, name: str, agent: Any):
        self.name = name  # Unique identifier for the agent
        self.agent = agent  # The agent instance
        self.children: list[AgentNode] = []  # List of child agents

    def add_child(self, child: "AgentNode"):
        self.children.append(child)

    def find_agent(self, name: str) -> "AgentNode | None":
        if self.name == name:
            return self
        for child in self.children:
            result = child.find_agent(name)
            if result:
                return result
        return None

class State(BaseModel):
    """Overall data state"""
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    tools: list[Any] = Field(default_factory=list)
    agents: AgentNode = Field(default_factory=lambda: AgentNode(name="root", agent=None))  # Root of the tree

SYS_MSG = """
You are an agent in a company of agents. Your goal is to accomplish what the user requests to the best of your company's ability. Break down the task into actionable broad high-level tasks 
for other agents to handle. 
Create a new agent for each actionable task, and ensure that each agent further breaks down their task into smaller, manageable tasks for their respective agents to handle.

Remember:
- Think about the best way to solve your task at hand

To complete the task, you can make tool calls to hire a team of: 
- supervisor agents to further breakdown a task
- codeact agents to write and execute code for a task
- deep research agents to learn more about how to solve a task and steps to complete the task
"""
#- swarm agents to solve a highly-specific task
memory = MemorySaver()
store = InMemoryStore()
thread_id = str(uuid.uuid4())
#primarly for deep_research
config = {"configurable": {"thread_id": thread_id,
                                "search_api": "tavily",
                                "planner_provider": "openai",
                                "planner_model": ModelTier.ECO_REASONING.value,
                                "writer_provider": "openai",
                                "writer_model": ModelTier.ECONOMY.value,
                                "max_search_depth": 1,
                            }}

# codeact_model = init_chat_model(ModelTier.ECO_REASONING.value, model_provider='openai')    

# https://github.com/langchain-ai/open_deep_research
async def create_deep_research_agent(state: State, config: RunnableConfig, supervisor_name: str, supervisor_prompt: str, topic: str, agent_name: str):
    """
    Creates a Deep Research agent to perform in-depth research on a given topic. Call this tool to gather detailed insights.

    Args:
        state: OverallState object containing tools and messages.
        config: RunnableConfig object for the agent's runtime configuration.
        supervisor_prompt: A string to guide the supervisor's review of the research plan.
        topic: A string specifying the research topic.

    Returns:
        A dictionary containing the research agent's response messages.
    """
    is_proceed = False
    research_plan = None
    output = None
    deep_research = builder.compile(checkpointer=memory, name=agent_name)

    # Add agent to parent node
    add_agent(state,supervisor_name, agent_name, deep_research)

    async for event in deep_research.astream(input={"topic": topic,}, config=config, stream_mode="updates"):
        print("Initial plan: ", event)
        research_plan = event
    while not is_proceed:
        # Report plan is generated, supervisor must review
        review_prompt = f"You are a supervisor tasked with completing:\n{supervisor_prompt}\n\nYour research agent has proposed a plan to learn more about:\nTopic:{topic}\nUsing this plan:\n{research_plan}\n\nDo you accept the proposed research plan?\nAnswer true or return feedback on how to improve the report"
        # Invoke another LLM call for supervisor review
        supervisor_model = get_openai_model(ModelTier.ECO_REASONING.value) 
        accepted_response = supervisor_model.invoke(input=review_prompt, config=config)
        # Digest response
        if accepted_response.content.strip().lower() == "true":
            is_proceed = True
            break  # Exit the inner loop and continue with the accepted plan
        else:
            print("Plan rejected. Generating a new plan...")
            async for event in deep_research.astream(Command(resume=accepted_response), config=config, stream_mode="updates"):
                print("Revised plan: ", event)
                research_plan = event
            continue

    if is_proceed:
        print("Plan accepted. Proceeding with research.")
        async for event in deep_research.astream(Command(resume=True), config=config, stream_mode="updates"):
            output = event
            print("Output research result:\n", event)
    
    return {'messages': output}

# https://github.com/huggingface/smolagents
def create_codeact_agent(state: State, task_prompt: str, supervisor_name: str, agent_name: str):
    """
    Creates a CodeAct agent to write and execute code. Call this tool to automate coding tasks.

    Args:
        state: OverallState object containing tools and configurations.
        task_prompt: A string describing the task the agent should perform (e.g., "Write a Python function to calculate factorial").

    Returns:
        A dictionary containing the agent's response messages.
    """
    # production environment should wrap this in a sandbox
    # codeact_agent = codeact.compile(checkpointer=MemorySaver())
    codeact_agent = CodeAgent(
        tools=state.tools,
        model=get_openai_model()
    )
    # add child agent to parent
    add_agent(state, supervisor_name, agent_name, codeact_agent)

    response = codeact_agent.run(task=task_prompt)
    print("Code Agent Output:\n", response)

    return {'messages': response}

# supervisor agent
def create_supervisor_agent(state: State, config: RunnableConfig, supervisor_name: str, agent_name: str):
    """
    Creates a Supervisor agent to manage and delegate tasks. Call this tool to oversee task execution and coordination.

    Args:
        state: OverallState object containing tools and messages.
        config: RunnableConfig object for the agent's runtime configuration.
        name: Name of the supervisor agent. Make it related to the prompt input
        
    Returns:
        A dictionary containing the supervisor agent's response messages.
    """
    agent_flow = create_supervisor(
        agents=[child.agent for child in state.agents.find_agent(supervisor_name).children],
        model=get_openai_model(ModelTier.ECO_REASONING.value),
        prompt=SYS_MSG,
        tools=state.tools,
        supervisor_name=agent_name
    ) 
    supervisor_agent = agent_flow.compile(name=agent_name)

    # add agent to parent
    if not agent_name == 'root':
        add_agent(state, supervisor_name, agent_name, supervisor_agent)

    result = supervisor_agent.invoke(
        input={"messages": state.messages},
        config=config
    )
    return {'messages': result}

# def create_agent(
#         agent_type: Literal['supervisor', 'codeact', 'deep_research'], 
#         state: State, 
#         config: RunnableConfig, 
#         **kwargs) -> AgentNode:
#     """
#     Factory function to create agents based on type.

#     Args:
#         agent_type: The type of agent to create (e.g., "supervisor", "codeact", "deep_research").
#         state: The current state object.
#         config: Configuration for the agent.
#         kwargs: Dictionary of string arguments for agent creation. Keys can include "name", "task_prompt", "supervisor_prompt", and "topic"

#     Returns:
#         An AgentNode representing the created agent.
#     """
#     if agent_type == "supervisor":
#         agent_instance = create_supervisor_agent(state, config, name=kwargs.get("name", "supervisor"))
#     elif agent_type == "codeact":
#         agent_instance = create_codeact_agent(state, task_prompt=kwargs.get("task_prompt", ""))
#     elif agent_type == "deep_research":
#         agent_instance = create_deep_research_agent(
#             state, config, supervisor_prompt=kwargs.get("supervisor_prompt", ""), topic=kwargs.get("topic", "")
#         )
#     else:
#         raise ValueError(f"Unknown agent type: {agent_type}")

#     return AgentNode(name=kwargs.get("name", agent_type), agent=agent_instance)

# swarm agent
# def create_swarm_agents(state: State, config: RunnableConfig, tools: list[Any]):
#     checkpointer = InMemoryStore()
#     swarm_flow = create_swarm(
#         agents=agents,
#         default_active_agent=agents[0]
#     )
#     swarm = swarm_flow.compile(checkpointer=checkpointer)

# create react agent

# Adds child agent to parent agent in State
def add_agent(state: State, parent_name: str, agent_name: str, agent_instance: Any):
    parent_node = state.agents.find_agent(parent_name)
    if not parent_node:
        raise ValueError(f"Parent agent '{parent_name}' not found.")
    new_agent = AgentNode(name=agent_name, agent=agent_instance)
    parent_node.add_child(new_agent)

def setup(state: State):
    # checkpointer.search(...)
    state.messages = []
    state.tools = [
        create_supervisor_agent,
        create_codeact_agent,
        create_deep_research_agent
    ]
    state.agents = create_supervisor_agent(state=state, config=config, name='root')
    return state

# Graph
builder = StateGraph(State)
builder.add_node("setup", setup)
builder.add_node("ceo", create_supervisor_agent)

builder.add_edge(START, "setup")
builder.add_edge("setup", "ceo")
builder.add_edge("ceo", END)

graph = builder.compile(name="Overall")