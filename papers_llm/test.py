import warnings
warnings.filterwarnings('ignore')
import torch
import os
from pprint import pprint
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
# Import things that are needed generically
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool

from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent

from typing import Literal, Optional
from langchain.output_parsers import PydanticOutputParser

import tool_pool as tp

dotenv_path = '/root/limlab/lim_helper_v2/.env'
load_dotenv(dotenv_path)

api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
channel_id = os.environ.get("CHANNEL_ID")

# load tool
tools = [tp.loadpaper, tp.recommendpaper, tp.code_matching]
# load Agent prompt
prompt = hub.pull("hwchase17/openai-tools-agent")
# print(prompt)
# Choose the LLM that will drive the agent
# Only certain models support this
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Construct the OpenAI Tools agent
agent = create_openai_tools_agent(llm, tools, prompt=prompt)
# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


output = agent_executor.invoke({"input": "'ToolBench(2307.16789)' 논문에서 소개한 Pass Rate evaluation 방법에 대해서 그림과 함께 설명해줘' 한글로 말해줘"})
print("OUTPUT",output)