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
login(token="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")

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
import arxiv
import os
import tarfile
import fitz  # PyMuPDF
from PIL import Image
import matplotlib.pyplot as plt

import os
import io
from IPython.display import display, Image as IPImage

from dotenv import load_dotenv
dotenv_path = '/Users/minkyuramen/Desktop/project/env'
load_dotenv(dotenv_path)

api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# sentence-transformers로 embedding 후 abstract간 cosine similarity를 측정하였다.
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

############ query → (reference paper/citation paper) recommendation ############
#################################################################################

def search_arxiv(query, max_results=1):
    '''
    1. semantic scoloar에서 pid와 paper 정보 검색
    2. pid를 통해 arxiv_id 검색
    해당 논문에 대한 id, title을 return
    '''
    # Define the API endpoint URL
    url = 'https://api.semanticscholar.org/graph/v1/paper/search?fields=paperId,title,authors,publicationDate'

    # paper name 기입
    query_params = {'query': query}
    headers = {'x-api-key': api_key}

    try:
        response = requests.get(url, params=query_params, headers=headers).json()
        # print(response)
        paper_id = response['data'][0]['paperId']
        paper_title = response['data'][0]['title']
        paper_authors = [author['name'] for author in response['data'][0]['authors']]
        paper_published = response['data'][0]['publicationDate']
        print(f'### Searched Paper ###')
        print(f'Title: {paper_title}')
        print(f'Authors: {paper_authors}')
        print(f'Published: {paper_published}')
        # print(f'ID: {paper_id}')
        url = f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=externalIds'
        headers = {'x-api-key': api_key}
        response = requests.get(url, params=query_params, headers=headers).json()
        print(response)
        arxiv_id = response['externalIds']['ArXiv']
    except:
        response = requests.get(url, params=query_params, headers=headers).json()
        # print(response)
        paper_id = response['data'][0]['paperId']
        paper_title = response['data'][0]['title']
        paper_authors = [author['name'] for author in response['data'][0]['authors']]
        paper_published = response['data'][0]['publicationDate']
        print(f'### Searched Paper ###')
        print(f'Title: {paper_title}')
        print(f'Authors: {paper_authors}')
        print(f'Published: {paper_published}')
        # print(f'ID: {paper_id}')
        url = f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=externalIds'
        headers = {'x-api-key': api_key}
        response = requests.get(url, params=query_params, headers=headers).json()
        print(response)
        arxiv_id = response['externalIds']['ArXiv']

    print(f'ArXiv_id: {arxiv_id}')

    return arxiv_id, paper_title

def download_pdf(arxiv_id, path_db='./papers_db'):
    """
    1. arXiv ID를 바탕으로 PDF를 path_db에 다운로드
    2. path_db에 이미 다운로드된 파일이 있다면, 다운로드를 skip
    다운로드된 파일의 path return
    """
    if not os.path.exists(path_db):
        os.makedirs(path_db)
    
    file_path = os.path.join(path_db, f'{arxiv_id}.pdf')
    
    if os.path.exists(file_path):
        print(f"File {file_path} already exists. Skipping download.")
        return file_path

    pdf_url = f'https://arxiv.org/pdf/{arxiv_id}.pdf'
    print(f'Downloading PDF from {pdf_url}')
    response = requests.get(pdf_url)

    if response.status_code != 200:
        raise Exception('Error downloading PDF from arXiv')

    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

def read_pdf(arxiv_id, path_db='./papers_db', limit_page=5, start_page=1, end_page=None):
    '''
    arixv_id에 해당하는 pdf 파일을 읽어서 text 형태로 변환
    arxiv_id를 가지는 논문 본문을 return
    '''
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
    <Paper Content Extract>
    query(target paper name)가 input으로 들어오면, target paper의 논문 본문(content)를 반환
    """
    paper_id, title = search_arxiv(query)
    download_path = download_pdf(paper_id)
    content = read_pdf(paper_id, start_page = start_page)

    return content

# print(query2content('Large Language Model Connected with Massive APIs'))

############ query → (reference paper/citation paper) recommendation ############
#################################################################################

def query2references(query, num=20, api_key=api_key):
    """
    <References>
    query(target paper name)가 input으로 들어오면, target paper의 reference(target paper가 인용한 논문)를 가져온다.
        threshold : influentialCitationCount가 limit 이상인 논문만 filtering
        num : 반환할 reference paper 최대 개수
    reference paper를 influentialCitationCount 순으로 정렬
    """    
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

    limit = 10
    references = []
    for elements in response2['data']:
        try:
            if elements['citedPaper']['influentialCitationCount'] > limit:
                references.append(elements)
            else: pass
        except: pass

    return response['data'], sorted(references, key=lambda x: x['citedPaper']['influentialCitationCount'], reverse=True)[:num]

def reference_recommend(query:str , num=20, threshold=0.6, recommend=5, api_key=api_key, sorting=True):
    """
    <References Recommendation>
    query(target paper name)가 input으로 들어오면, target paper의 reference 중 유사한 논문들을 추천
    1. target paper와 reference paper의 abstract간의 cosine similarity를 계산
    2. 유사도가 threshold 이상인 논문들을 추천 + recommend 개수만큼 top K filtering
    3. sorting=True 이면, 날짜순으로 정렬
    """
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    recommend += 1

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
        try:
            rec_lst[i]['intent'] = ' '.join(rec_intent[i])
            rec_lst[i]['context'] = rec_context[i][0]
        except: 
            rec_lst[i]['intent'] = 'None'
            rec_lst[i]['context'] = 'None'
    if recommend:
        if len(rec_lst) > recommend:
            rec_lst = rec_lst[1:recommend]

    # 날짜순으로 정렬
    if sorting==True:
        rec_lst = sorted(rec_lst, key=lambda x: x['influentialCitationCount'], reverse=True)[:num]
        return target_response, sorted(rec_lst, key=lambda x: datetime.strptime(x['publicationDate'], '%Y-%m-%d') if isinstance(x['publicationDate'], str) else x['publicationDate'])
    
    return rec_lst


def query2citations(query, num=20, api_key=api_key):
    '''
    <Citations>
    query(target paper name)가 input으로 들어오면, target paper의 citation(target paper를 인용한 논문)을 가져온다.
        최신 논문들은 citation이 많이 없어 따로 influentialCitationCount filtering X
        num : 반환할 citation paper 최대 개수
    Citation paper를 Citation 순으로 정렬
    '''
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

    def get_citation_count(item):
        influential_citation_count = item.get('influentialCitationCount')
        if influential_citation_count is not None:
            return influential_citation_count
        else:
            return 0
        
    return target_response, sorted(citations_response['data'][0]['citations'], key=get_citation_count, reverse=True)[:num]


def citation_recommend(query, num=20, threshold=0.6, recommend=5, api_key=api_key):
    '''
    <Citations Recommendation>
    query(target paper name)가 input으로 들어오면, target paper의 citation 중 유사한 논문들을 추천
    1. target paper와 citation paper의 abstract간의 cosine similarity를 계산
    2. 유사도가 threshold 이상인 논문들을 추천 + recommend 개수만큼 top K filtering
    3. publicationDate를 기준으로 날짜순으로 정렬
    '''
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    recommend += 1
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
        rec_lst = rec_lst[1:recommend]

    # 날짜순으로 정렬
    return sorted(rec_lst, key=lambda x: datetime.strptime(x['publicationDate'], '%Y-%m-%d') if isinstance(x['publicationDate'], str) else x['publicationDate'])

def query2recommend_paper(query, type='default'):
    '''
    type에 따라 target paper에 대한 citation 혹은 reference를 추천
    '''
    if type == 'citation':
        return citation_recommend(query=query, num=30, api_key=api_key, threshold=0.6, recommend=5)
    elif type == 'reference':
        return reference_recommend(query=query, num=30, api_key=api_key, threshold=0.6, recommend=5)
    else:
        raise Exception('type should be either citation or reference')

###################### query → figure & figure description ######################
#################################################################################

def download_arxiv_source(arxiv_id, output_dir):
    """
    <figure download>
    figure가 포함되어 있는 source를 반환한다.
    """
    # 검색한 arXiv 논문 정보 가져오기
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(search.results())

    # 소스 파일 다운로드 링크
    source_url = paper.pdf_url.replace('pdf', 'e-print')
    
    # 출력 디렉토리가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 파일 다운로드
    response = requests.get(source_url)
    output_path = os.path.join(output_dir, f"{arxiv_id}.tar.gz")
    with open(output_path, 'wb') as f:
        f.write(response.content)
    
    file_path = f'{output_path}'
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=output_dir)
    
    print(f"Source files downloaded to: {output_path}")

def find_pdf_files(root_folder):
    '''
    <figure path find>
    source 폴더 내의 모든 figure를 찾아서 {name:path} 형태의 dictionary 반환
    '''
    pdf_files = {}
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith('.pdf'):
                name = filename.replace('.pdf', '').split('/')[-1]
                pdf_files[name] = os.path.join(dirpath, filename)
    return pdf_files

def query_name_matching(query, pdf_files):
    '''
    <figure name matching>
    query와 pdf_files를 받아 query와 가장 유사한 이름을 가진 pdf 파일의 이름을 반환
    '''
    figure_name = list(pdf_files.keys())
    query = query.lower()
    pdf_names = list(pdf_files.keys())
    pdf_names = [name.lower() for name in pdf_names]
    pdf_names.insert(0,query)
    
    encoded_input = tokenizer(pdf_names, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    rec = cosine_similarity(sentence_embeddings)
    rec = rec[0][1:]
    max_index = rec.argmax()

    return pdf_names[max_index + 1]

def display_figure(pdf_files, name):
    """
    <figure display>
    name과 pdf_files를 받아 figure를 display
    """
    pdf_path = pdf_files[name]
    print(pdf_path)
    # PDF 파일 열기
    pdf_document = fitz.open(pdf_path)
    
    # 첫 페이지 가져오기
    page = pdf_document.load_page(0)
    pix = page.get_pixmap()
    
    # 이미지 변환
    image_bytes = pix.tobytes()
    image = Image.open(io.BytesIO(image_bytes))

    plt.imshow(image)
    plt.axis('off')  # 축을 표시하지 않도록 설정합니다
    plt.show()

def query2figure_and_content(query:str, instruction:str, start_page:int = 1):
    """
    <Paper Content Extract>
    query(target paper name)가 input으로 들어오면, target paper의 논문 본문(content)를 반환
    """
    paper_id, title = search_arxiv(query)
    download_path = download_pdf(paper_id)
    content = read_pdf(paper_id, start_page = start_page)

    ## image download
    output_dir = download_path.replace('.pdf', '')
    arxiv_id = output_dir.split('/')[-1]
    # source file download
    download_arxiv_source(arxiv_id, output_dir)
    # figure file path
    pdf_files = find_pdf_files(output_dir)
    # instruction & figure matching
    name = query_name_matching(instruction, pdf_files)
    # figure display
    display_figure(pdf_files, name)

    return content