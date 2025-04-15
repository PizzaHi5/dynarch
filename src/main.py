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

class AgentNode(BaseModel):
    name: str = ""
    agent: Any = None
    children: list["AgentNode"] = Field(default_factory=list)

    # def __init__(self, name: str, agent: Any):
    #     self.node_name = name  # Unique identifier for the agent
    #     self.agent = agent  # The agent instance
    #     self.children: list[AgentNode] = []  # List of child agents

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

class Task(BaseModel):
    task_prompt: str = Field(examples="Research how to build optimal Python code for interacting with Google Maps API", default_factory=str)
    agent_type_int: Literal['supervisor', 'codeact', 'deep_research'] = Field(description="Must be one of: 'supervisor', 'codeact', 'deep_research'")
    agent_name: str = Field(description="name of the agent assigned to this task", default_factory=str)
    supervisor_prompt: str = Field(description="this is the prompt the supervisor will use to review deep research topics", default_factory=str)
    supervisor_name: str = Field(description="The name of the supervisor agent", default_factory=str)

class SupervisorOutput(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)

class State(BaseModel):
    """Overall data state"""
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    tools: list[Any] = Field(default_factory=list)
    agents: AgentNode = Field(default_factory=lambda: AgentNode(name="root", agent=None))  # Root of the tree

    tasks: list[Task] = Field(default_factory=list)

SYS_MSG = """
You are an agent in a company of agents. Your goal is to accomplish what the user requests to the best of your company's ability. Break down the task into actionable broad high-level tasks 
for other agents to handle. 
Create a new agent for each actionable task, and ensure that each agent further breaks down their task into smaller, manageable tasks for their respective agents to handle.

Remember:
- Think about the best way to solve your task at hand

To complete the task, you can make tool calls to hire a team of: 
- supervisor agents to further breakdown a task
- deep research agents to learn more about how to solve a task and steps to complete a task
- codeact agents to write and execute code to complete a task

Your specific task is listed below:

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
async def create_deep_research_agent(state: State, config: RunnableConfig, supervisor_name: str, supervisor_prompt: str, topic: str, agent_name: str) -> dict:
    """
    Creates a Deep Research agent to perform in-depth research on a given topic. Call this tool to gather detailed insights.

    Args:
        state: OverallState object containing tools and messages.
        config: RunnableConfig object for the agent's runtime configuration.
        supervisor_name: String name of the supervisor agent who made this agent.
        supervisor_prompt: A string to guide the supervisor's review of the research plan.
        topic: A string specifying the research topic.
        agent_name: String name of this new deep research agent

    Returns:
        dict: A dictionary containing the research agent's response messages.
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
def create_codeact_agent(state: State, task_prompt: str, supervisor_name: str, agent_name: str) -> dict:
    """
    Creates a CodeAct agent to write and execute code. Call this tool to complete coding tasks.

    Args:
        state: OverallState object containing tools and configurations.
        task_prompt: A string describing the task the agent should perform (e.g., "Write a Python function to calculate factorial").
        supervisor_name: String name of the supervisor agent who made this agent.
        agent_name: String name of this new codeact agent. 
        
    Returns:
        dict: A dictionary containing the agent's response messages.
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
def create_supervisor_agent(state: State, config: RunnableConfig, supervisor_name: str, agent_name: str, prompt: str) -> SupervisorOutput:
    """
    Creates a Supervisor agent to breakdown high level tasks into many lower level tasks and creates agents for each task. Call this tool to breakdown a high level task into many more detailed tasks.

    Args:
        state: OverallState object containing tools and messages.
        config: RunnableConfig object for the agent's runtime configuration.
        supervisor_name: String name of the supervisor agent who made this agent.
        agent_name: String name of this new supervisor agent. Make it related to the prompt input.
        prompt: String objective this agent will breakdown further to solve with its own agents.
        
    Returns:
        SupervisorOutput: An object containing 'messages' and 'tasks'.
    """
    # Create root supervisor
    if (supervisor_name == '' or supervisor_name is None) and agent_name == 'root':
        agent_flow = create_supervisor(
            model=get_openai_model(ModelTier.ECO_REASONING.value),
            prompt=SYS_MSG,
            response_format=SupervisorOutput,
            tools=state.tools,
            supervisor_name=agent_name
        )
    else:
        agent_flow = create_supervisor(
            agents=[child.agent for child in state.agents.find_agent(supervisor_name).children],
            model=get_openai_model(ModelTier.ECO_REASONING.value),
            prompt=SYS_MSG + prompt,
            response_format=SupervisorOutput,
            tools=state.tools,
            supervisor_name=agent_name
        ) 
    supervisor_agent = agent_flow.compile(checkpointer=memory, name=agent_name)

    # add agent to parent
    if not agent_name == 'root':
        add_agent(state, supervisor_name, agent_name, supervisor_agent)

    # Result is a SupervisorOutput
    result = supervisor_agent.invoke(
        input={"messages": state.messages},
        config=config
    )
    print(result)

    output = SupervisorOutput(messages=result.messages, tasks=result.tasks)
    
    return output

# Looping structure
def execute_tasks(state: State, config: RunnableConfig) -> State:
    """
    Executes the tasks in the state by creating and running the appropriate agents.

    Args:
        state: The overall state object containing tasks and agents.
        config: The runtime configuration for the agents.

    Returns:
        State: The updated state after executing the tasks.
    """
    for task in state.tasks:
        #check/create supervisors
        if task.agent_type_int == 'supervisor':
            create_supervisor_agent(state, config, task.supervisor_name, task.agent_name, task.task_prompt)
        elif task.agent_type_int == 'codeact':
            create_codeact_agent(state, task.task_prompt, task.supervisor_name, task.agent_name)
        elif task.agent_type_int == 'deep_research':
            create_deep_research_agent(state, config, task.supervisor_name, task.supervisor_prompt, task.task_prompt, task.agent_name)
        else:
            raise ValueError("How did you mess this up?")
    
    return state
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
    """
    Adds a child agent to the parent agent in the state.

    Args:
        state: The overall state object containing agents.
        parent_name: The name of the parent agent.
        agent_name: The name of the new agent to add.
        agent_instance: The instance of the new agent.

    Returns:
        None
    """
    parent_node = state.agents.find_agent(parent_name)
    if not parent_node:
        raise ValueError(f"Parent agent '{parent_name}' not found.")
    new_agent = AgentNode(name=agent_name, agent=agent_instance)
    parent_node.add_child(new_agent)

def setup(state: State, config: RunnableConfig) -> dict:
    """
    Sets up the initial state and creates the root supervisor agent.

    Args:
        state: The overall state object containing tools and messages.
        config: The runtime configuration for the agents.

    Returns:
        dict: A dictionary containing 'messages' and 'tasks'.
    """
    # checkpointer.search(...)
    state.messages = []
    state.tools = [
        create_supervisor_agent,
        create_codeact_agent,
        create_deep_research_agent
    ]
    root_node = AgentNode(name="root", agent='None', children=[])
    root_agent_response = create_supervisor_agent(
        state=state, 
        config=config, 
        supervisor_name='', 
        agent_name='root',
        prompt="You are the root agent. Your generated tasks should be broken down further."
        )
    root_node.agent = 'supervisor'
    state.tasks = root_agent_response.tasks

    return {
        'messages': root_agent_response.messages,
        'tasks': root_agent_response.tasks
    }

# Graph
builder = StateGraph(State)
builder.add_node("setup", setup)
builder.add_node("execution", execute_tasks)

builder.add_edge(START, "setup")
builder.add_edge("setup", "execution")
builder.add_edge("execution", END)

graph = builder.compile(name="Overall")