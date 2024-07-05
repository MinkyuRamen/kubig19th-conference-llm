import huggingface_hub
huggingface_hub.login(token="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")

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
from typing import Literal, Optional
from langchain.tools import BaseTool, StructuredTool, tool

from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI

from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from dotenv import load_dotenv

from getpaper import GetPaper

from recommendpaper import RecommendPaper

ss_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

# Updated loadpaer tool
# model can use this tool several times, to get the list of the section, and then see the detail content of the paper.
# if the ar5iv_mode is False, 
getpapermodule = GetPaper(ss_api_key, ar5iv_mode = True, page_limit = 5)
recommendpapermodule = RecommendPaper(ss_api_key, threshold = 0.6)
# codeanalysismodule = CodeAnalysis(ss_api_key, openai_key, path_db = '/root/limlab/lim_helper_v2/kubig19th-conference-llm/papers_llm/papers_db', code_db = '/root/limlab/lim_helper_v2/kubig19th-conference-llm/papers_llm/code_db')

class load_paper_input(BaseModel):
    title: str = Field(description="target paper title")
    sections: list = Field(description='list of sections', default = None)
    arxiv_id: Optional[str] = Field(default=None, description="arxiv id of the paper. use it when the prompt contain arxiv id. arxiv IDs are unique identifiers for preprints on the arxiv repository, formatted as `YYMM.NNNNN`. For example, `1706.03762` refers to a paper submitted in June 2017, and `2309.10691` refers to a paper submitted in September 2023.")


# loadpaper를 사용하여 우선 target paper의 section list를 불러온 후, 각 section의 content를 불러오게끔 설정
loadpaper = StructuredTool.from_function(
    func=getpapermodule.load_paper,
    name="loadpaper",
    description="""
        The `loadPaper` tool is designed to facilitate the process of retrieving and reading academic papers based on a given search title. \
        The `title` parameter is a string representing the title of the paper. The 'sections' parameter is a list representing the list of the sections in the paper. \
        The 'arxiv_id' parameter is a string representing the arxiv id. \
        If the sections parameter is none, you can get the section list of the paper. If the sections parameter get the section list, you can load the paper's content. \
        Use this tool several times to get the section first and then get the detail content of each section. \
    """,
    args_schema=load_paper_input
)

## (reference paper/citation paper) recommendation
class recommend_paper_input(BaseModel):
    query: str = Field(description="target paper title")
    rec_type: Literal['reference', 'citation'] = Field(description="reference or citation paper recommendation")
    rec_num: Optional[int] = Field(default=5, description="number of recommended papers default is 5")  # 기본값 5로 설정

recommendpaper = StructuredTool.from_function(
    func=recommendpapermodule.query2recommend_paper,
    name="recommendpaper",
    description="""
        This 'recommendpaper' tool is designed to recommend relevant academic papers based on a given query, \
        focusing on either the papers cited by the target paper (references) or the papers citing the target paper (citations).\
        The `query` parameter is a string representing the title of the paper.\
        The `rec_type` parameter specifies whether the recommendation should be based on references or citations.\
        The `rec_num` parameter specifies the number of recommended papers. if the number is NOT MENTIONED set rec_num=5\
        The recommendation is based on the cosine similarity between the abstracts of the target paper and its references or citations.\
        Users can specify whether they want recommendations from references or citations. \
        The tool returns the top relevant papers sorted by publication date or influential citation count.
    """,
    args_schema=recommend_paper_input
)

