import warnings
warnings.filterwarnings('ignore')
import torch

import os
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.mistral.modeling_mistral import MistralForCausalLM
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from transformers import LlamaForCausalLM
from transformers import PreTrainedTokenizerFast
from huggingface_hub import login

from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Optional, List, Mapping, Any
from datetime import datetime

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

import requests
import feedparser
import PyPDF2
import re

from dotenv import load_dotenv
dotenv_path = '.env'
load_dotenv(dotenv_path)

api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
huggingface_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
login(token=huggingface_api_key)

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# sentence-transformers로 embedding 후 abstract간 cosine similarity를 측정하였다.
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')


def query2paperID(query, api_key=api_key):
    # Define the API endpoint URL
    url = 'https://api.semanticscholar.org/graph/v1/paper/search?fields=paperId,title,abstract'

    # paper name 기입
    query_params = {'query': query}
    headers = {'x-api-key': api_key}

    response = requests.get(url, params=query_params, headers=headers).json()
    paper_id = response['data'][0]['paperId']
    title = response['data'][0]['title']
    return paper_id, title

# arxiv api 이용, paper search
def search_arxiv(query, max_results=1):
    base_url = 'http://export.arxiv.org/api/query?'
    query_url = f'search_query=all:{query}&start=0&max_results={max_results}'
    response = requests.get(base_url + query_url)

    if response.status_code != 200:
        raise Exception('Error fetching data from arXiv API')

    feed = feedparser.parse(response.content)
    
    paper = feed.entries[0]
    print(f'### Searched Paper ###')
    print(f'Title: {paper.title}')
    print(f'Authors: {paper.author}')
    print(f'Published: {paper.published}')
    print(f'ID: {paper.id.split("/")[-1]}')

    arxiv_id = paper.id.split('/')[-1]
    return arxiv_id, paper.title


def download_pdf(arxiv_id, path_db='./papers_db'):
    """
    Download the PDF of a paper given its arXiv ID, if it does not already exist.
    """
    if not os.path.exists(path_db):
        os.makedirs(path_db)
    
    file_path = os.path.join(path_db, f'{arxiv_id}.pdf')
    
    if os.path.exists(file_path):
        print(f"File {file_path} already exists. Skipping download.")
        return file_path

    pdf_url = f'https://arxiv.org/pdf/{arxiv_id}.pdf'
    response = requests.get(pdf_url)

    if response.status_code != 200:
        raise Exception('Error downloading PDF from arXiv')

    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

def read_pdf(arxiv_id, path_db='./papers_db', limit_page=3, start_page=1, end_page=None):
    pdf_content = ""
    file_path = f'{path_db}/{arxiv_id}.pdf'
    
    if not os.path.exists(file_path):
        return f"Error: The file {file_path} does not exist."

    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)
            
            if end_page is None or end_page > total_pages:
                end_page = total_pages

            for page_num in range(start_page, end_page):
                page = reader.pages[page_num-1]
                pdf_content += page.extract_text()
                if page_num == limit_page:
                    print('Page limit reached at', limit_page + 1)
                    break

    except FileNotFoundError:
        return f"Error: The file {file_path} does not exist."
    except Exception as e:
        return f"An error occurred while reading the file: {e}"

    pdf_content = re.sub(r'\s+', ' ', pdf_content).strip()
    return pdf_content

def query2content(query:str, start_page:int = 1):
    """
    Perform a query, download the corresponding PDF (if not already downloaded), and extract its content.
    """
    paper_id, title = search_arxiv(query)

    download_path = download_pdf(paper_id)
    
    content = read_pdf(paper_id, start_page = start_page)
    return content


def query2references(query, num=20, api_key=api_key):
    """References"""
    # Define the API endpoint URL
    url = 'https://api.semanticscholar.org/graph/v1/paper/search?fields=paperId,title,abstract'

    # paper name 기입
    query_params = {'query': query}
    headers = {'x-api-key': api_key}

    response = requests.get(url, params=query_params, headers=headers).json()
    paper_id = response['data'][0]['paperId']


    fields = '?fields=title,publicationDate,influentialCitationCount,contexts,intents,abstract'
    """
    context ; snippets of text where the reference is mentioned
    intents ; intents derived from the contexts in which this citation is mentioned.
    """

    url = f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references'+ fields

    # Send the API request
    response2 = requests.get(url=url, headers=headers).json()
    # response

    threshold = 10
    references = []
    for elements in response2['data']:
        try:
            if elements['citedPaper']['influentialCitationCount'] > threshold:
                references.append(elements)
            else: pass
        except: pass

    return response['data'], sorted(references, key=lambda x: x['citedPaper']['influentialCitationCount'], reverse=True)[:num]

def reference_recommend(query:str , num=20, threshold=0.6, recommend=5, api_key=api_key):

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    ### 상위 20개의 reference이 많은 논문들을 select
    target_response, reference = query2references(query=query,num=num,api_key=api_key)
    reference_response = [ref['citedPaper'] for ref in reference]
    reference_context = [ref['contexts'] for ref in reference]
    reference_intent = [ref['intents'] for ref in reference]


    ## target 논문과 num개의 reference 사이의 유사도 계산
    abs_dict = {}
    abs_dict[target_response[0]['title']] = target_response[0]['abstract']

    for keyword in reference_response:
        paper_id, title = keyword['paperId'], keyword['title']
        abstract = str(keyword['abstract'])
        abs_dict[title] = abstract

    sentences = list(abs_dict.values())
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    rec = cosine_similarity(sentence_embeddings)
    
    ## 유사도가 threshold 이상인 논문들을 추출
    indices = np.where(rec[0][1:] > threshold)[0]
    rec_lst = [reference_response[i] for i in indices]
    rec_context = [reference_context[i] for i in indices]
    rec_intent = [reference_intent[i] for i in indices]

    for item in rec_lst:
        if item['publicationDate'] is None:
            item['publicationDate'] = datetime.max

    # intent와 context를 list에 추가
    for i in range(len(rec_lst)):
        rec_lst[i]['intent'] = ' '.join(rec_intent[i])
        rec_lst[i]['context'] = rec_context[i][0]
    
    if len(rec_lst) > recommend:
        rec_lst = rec_lst[:recommend]

    # 날짜순으로 정렬
    return sorted(rec_lst, key=lambda x: datetime.strptime(x['publicationDate'], '%Y-%m-%d') if isinstance(x['publicationDate'], str) else x['publicationDate'])

def query2citations(query, num=20, api_key=api_key):
    # Define the API endpoint URL
    url = 'https://api.semanticscholar.org/graph/v1/paper/search'

    # paper name 기입
    query_params = {'query': query,'fields': 'citations,citations.influentialCitationCount,citations.title,citations.publicationDate,citations.abstract'}

    # semantic scholarship api 넣는다.
    headers = {'x-api-key': api_key}

    citations_response = requests.get(url, params=query_params, headers=headers).json()
    paper_id = citations_response['data'][0]['paperId']

    url = f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=abstract,title'

    target_response = requests.get(url, params=query_params, headers=headers).json()

    return target_response, sorted(citations_response['data'][0]['citations'], key=get_citation_count, reverse=True)[:num]

def get_citation_count(item):
    influential_citation_count = item.get('influentialCitationCount')
    if influential_citation_count is not None:
        return influential_citation_count
    else:
        return 0

def citation_recommend(query, num=20, threshold=0.6, recommend=5, api_key=api_key):

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    ### 상위 20개의 citation이 많은 논문들을 select
    target_response, citation_response = query2citations(query=query, num=num, api_key=api_key)

    ## target 논문과 20개의 citation 사이의 유사도 계산
    abs_dict = {}
    abs_dict[target_response['title']] = target_response['abstract']

    for keyword in citation_response:
        paper_id, title = keyword['paperId'], keyword['title']
        abstract = str(keyword['abstract'])
        abs_dict[title] = abstract

    sentences = list(abs_dict.values())
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    rec = cosine_similarity(sentence_embeddings)
    
    ## 유사도가 threshold 이상인 논문들을 추출
    indices = np.where(rec[0][1:] > threshold)[0]
    rec_lst = [citation_response[i] for i in indices]

    for item in rec_lst:
        if item['publicationDate'] is None:
            item['publicationDate'] = datetime.max

    if len(rec_lst) > recommend:
        rec_lst = rec_lst[:recommend]

    # 날짜순으로 정렬
    return sorted(rec_lst, key=lambda x: datetime.strptime(x['publicationDate'], '%Y-%m-%d') if isinstance(x['publicationDate'], str) else x['publicationDate'])
    