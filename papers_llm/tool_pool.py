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
import api_pool as ap
from dotenv import load_dotenv

from getpaper_v2 import GetPaper_v2 # figure include
from getpaper import GetPaper

from code_analysis import CodeAnalysis
from recommendpaper import RecommendPaper

dotenv_path = '/Users/minkyuramen/Desktop/project/env'
load_dotenv(dotenv_path)

ss_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

# Updated loadpaer tool
# model can use this tool several times, to get the list of the section, and then see the detail content of the paper.
# if the ar5iv_mode is False, 
getpapermodule = GetPaper_v2(ss_api_key, ar5iv_mode = True, path_db = './papers_db', page_limit = 5)
recommendpapermodule = RecommendPaper(ss_api_key, threshold = 0.6)
codeanalysismodule = CodeAnalysis(ss_api_key, openai_key, path_db = './papers_db', code_db = './code_db')

class load_paper_input(BaseModel):
    title: str = Field(description="target paper title")
    sections: list = Field(description='list of sections', default = None)
    show_figure: Optional[bool] = Field(default=False, description="show figure in the paper")
    arxiv_id: Optional[str] = Field(default=None, description="arxiv id of the paper. use it when the prompt contain arxiv id")


# loadpaper를 사용하여 우선 target paper의 section list를 불러온 후, 각 section의 content를 불러오게끔 설정
loadpaper = StructuredTool.from_function(
    func=getpapermodule.load_paper,
    name="loadpaper",
    description="""
        The `loadPaper` tool is designed to facilitate the process of retrieving and reading academic papers based on a given search title. \
        The `title` parameter is a string representing the title of the paper. The 'sections' parameter is a list representing the list of the sections in the paper. \
        The 'show_figure' parameter is a boolean value that determines whether to display the figures in the paper. \
        The 'arxiv_id' parameter is a string representing the arxiv id. \
        If the sections parameter is none, you can get the section list of the paper. If the sections parameter get the section list, you can load the paper's content. \
        Use this tool several times to get the section first and then get the detail content of each section. \
        Do NOT show the figures when 'sections' parameter is None. \
    """,
    args_schema=load_paper_input
)

'''
## target paper context extract
class load_paper_input(BaseModel):
    query: str = Field(description="target paper title")
    start_page: int = Field(description='start page number')
    
loadpaper = StructuredTool.from_function(
    func=ap.query2content,
    name="loadpaper",
    description="""
        The `loadpaper` tool is designed to facilitate the process of retrieving and reading academic papers based on a given search query. \
        The `query` parameter is a string representing the title of the paper. The `loadPaper` tool extracts and returns the text content from the PDF. \
        If the paper does not contain the desired information, you can adjust the start_page argument to specify the starting page for reading. \
        For instance, to extract the title, abstract, or detailed method sections, set start_page to 1. \
        Conversely, for sections like the appendix that are typically found towards the end of the paper, set start_page to a higher number, such as 9.
    """,
    args_schema=load_paper_input
)
'''

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

'''
## (reference paper/citation paper) recommendation
class recommend_input(BaseModel):
    query: str = Field(description="target paper title")
    type: Literal['reference', 'citation'] = Field(description="reference or citation paper recommendation")

recommendpaper = StructuredTool.from_function(
    func=ap.query2recommend_paper,
    name="recommendpaper",
    description="""
        This 'recommendpaper' tool is designed to recommend relevant academic papers based on a given query, \
        focusing on either the papers cited by the target paper (references) or the papers citing the target paper (citations).\
        The `query` parameter is a string representing the title of the paper.\
        The `type` parameter specifies whether the recommendation should be based on references or citations.\
        The recommendation is based on the cosine similarity between the abstracts of the target paper and its references or citations.\
        Users can specify whether they want recommendations from references or citations. \
        The tool returns the top relevant papers sorted by publication date or influential citation count.
    """,
    args_schema=recommend_input
)

## figure explanation & visualization << loadpaper와 통합
class loadfigure(BaseModel):
    query: str = Field(description="target paper title")
    instruction : str = Field(description='user instruction for figure explanation')
    start_page: int = Field(description='start page number')
    
loadfigure = StructuredTool.from_function(
    func=ap.query2figure_and_content,
    name="loadfigure",
    description="""    
        The `loadpaper` tool facilitates the retrieval, reading, and visualization of academic papers based on a given search query. \
        The `instruction` parameter is a string representing the title or relevant keywords of the target paper. \
        The `question` parameter is a string representing the subject being asked to explain.\
        If the paper does not contain the desired information, you can adjust the start_page argument to specify the starting page for reading. \
        For instance, to extract the title, abstract, or detailed method sections, set start_page to 1. \
        This tool not only provides the text content of the paper but also identifies and displays relevant figures as per the user's instructions.
    """,
    args_schema=loadfigure
)
'''

####### Code Analysis와 관련된 tool 정의

class code_analysis_inputs(BaseModel):
    title: str = Field(description="target paper title")
    contents : str = Field(description = "Contents in paper")
    github_link : Optional[str] = Field(description = "Generated code by GPT", default = None)

code_matching = StructuredTool.from_function(
    func=codeanalysismodule.code_analysis,
    name="code_matching",
    description="""
    'code_matching' tool provides references for the most closely matching parts between the content of a research paper and the actual implemented code. \
    'title' is a parameter that takes the title of the research paper. \
    'contents' is a parameter where the user inputs the parts of the paper they are curious about how to implement in code. \
    'response' refers to the code generated by GPT based on 'contents'. \
    The 'code_matching' tool takes the title and content of a research paper as input and compares the GPT-generated code with actual implemented codes to identify the parts that are most similar, 
    thus providing references for the implementation process.""",
    args_schema = code_analysis_inputs
)



####### Code Analysis와 관련된 tool 정의
"""
class cloning_github(BaseModel):
    title: str = Field(description="target paper title")

GitCloning = StructuredTool.from_function(
    func=codeanalysismodule.Git_cloning,
    name="get_github_repository",
    description="""  """,
    args_schema=cloning_github
) ## repo_path를 반환(optional)

load_all_functions = StructuredTool.from_function(
    func=codeanalysismodule.get_all_functions,
    name="get_github_repository_functions_names",
    description=""" """
)

code_matching = StructuredTool.from_function(
    func=codeanalysismodule.code_analysis,
    name="code_analysis",
    description="""  """
)
"""