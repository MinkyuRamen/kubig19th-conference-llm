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
from typing import Literal
from langchain.tools import BaseTool, StructuredTool, tool

from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI

from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
import api_pool as ap
from dotenv import load_dotenv

from getpaper import GetPaper

dotenv_path = '/Users/minkyuramen/Desktop/project/env'
load_dotenv(dotenv_path)

ss_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")


# Updated loadpaer tool
# model can use this tool several times, to get the list of the section, and then see the detail content of the paper.
# if the ar5iv_mode is False, 
getpapermodule = GetPaper(ss_api_key, ar5iv_mode = True, path_db = './papers_db', page_limit = 5)

class load_paper_input(BaseModel):
    title: str = Field(description="target paper title")
    sections: list = Field(description='list of sections', default = None)

loadpaper = StructuredTool.from_function(
    func=getpapermodule.load_paper,
    name="loadpaper",
    description="The `loadPaper` tool is designed to facilitate the process of retrieving and reading academic papers based on a given search title. \
    The `title` parameter is a string representing the title of the paper. The 'sections' parameter is a list representing the list of the sections in the paper. \
    If the sections parameter is none, you can get the section list of the paper. If the sections parameter get the section list, you can load the paper's content. \
    Use this tool several times to get the section first and then get the detail content of each section",
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