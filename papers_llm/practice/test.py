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


# output = agent_executor.invoke({"input": "'Attention is All You Need(아카이브 id 1706.03762)' 의 논문에서 Encoder 파트 구조가 어떻게 구성돼는지 그림과 함께 설명해줘. 한글로 말해줘"})
output = agent_executor.invoke({"input": "'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale'에서 method 파트에 대해 수식과 함께 자세히 설명해줘"})

# output = agent_executor.invoke({"input": "'Toolformer: Language Models Can Teach Themselves to Use Tools' 를 읽었는데 혹시 후속 논문 6편정도 추천해줄 수 있어?"})
# output = agent_executor.invoke({"input": "'Toolformer: Language Models Can Teach Themselves to Use Tools' 를 읽기전에 도움되는 이전 논문 6편정도 추천해줄 수 있어?"})

# output = agent_executor.invoke({"input": " ‘RoBERTa(1907.11692)’ 의 결론에 대해서 간단하게 설명해줘"})
# output = agent_executor.invoke({'input': "'QLoRA'논문에서  'To summarize, QLORA has one storage data type (usually 4-bit NormalFloat)and a computation data type (16-bit BrainFloat). We dequantize the storage data type to the computation data type to perform the forward and backward pass, but we only compute weight gradients for the LoRA parameters which use 16-bit BrainFloat.' 이 부분이 실제로 어떻게 코드로 구현되어있어? github 주소는 다음과 같아."})

print("OUTPUT",output)