from httpx import NetworkError
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
import PyPDF2
import os
import ast
import feedparser
import fitz  # PyMuPDF
import re
import subprocess
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class NetworkError(Exception):
    """Custom exception for network errors."""
    pass

class GitClone:
    """
    
    
    
    """

    def __init__(self, ss_api_key, path_db='./papers_db', code_db='./code_db'):
        self.ss_api_key = ss_api_key
        self.path_db= path_db
        self.code_db = code_db

########## PDF 안의 github repository 링크를 찾아서 cloning
    def get_paper_id_from_title(self, title, api_key):
        search_url = "https://api.semanticscholar.org/graph/v1/paper/search"

        params = {
            "query": title,
            "fields": "paperId",
            "limit": 1
        }
        headers = {
            "x-api-key": api_key
        }
        response = requests.get(search_url, params=params, headers=headers)

        if response.status_code == 403:
            raise Exception("API request failed with status code 403: Forbidden. Please check your API key and permissions.")
        elif response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}")

        data = response.json()

        papers = data.get('data', [])
        if not papers:
            return None

        first_paper = papers[0]
        paper_id = first_paper.get('paperId')

        return paper_id

    def get_arxiv_pdf_url(self, paper_id, api_key):
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=externalIds"

        response = requests.get(url, headers={"x-api-key": api_key})

        if response.status_code == 403:
            raise Exception("API request failed with status code 403: Forbidden. Please check your API key and permissions.")
        elif response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}")

        data = response.json()

        external_ids = data.get('externalIds', {})
        arxiv_id = external_ids.get('ArXiv')

        if not arxiv_id:
            return None

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        return pdf_url

    def download_pdf(self, pdf_url):
        response = requests.get(pdf_url)
        if response.status_code != 200:
            raise Exception(f"Failed to download PDF with status code {response.status_code}")

        return response.content

    def extract_github_links_from_pdf(self, pdf_content):
        # PDF를 메모리에 로드
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")

        # 첫 번째 페이지 추출
        first_page = pdf_document[0]
        text = first_page.get_text()

        # GitHub 링크 추출 (정규 표현식 사용)
        github_links = re.findall(r'(https?://github\.com/[^\s]+)', text)

        return github_links

    def clone_github_repository(self, github_url):
        # GitHub URL에서 리포지토리 이름 추출
        parsed_url = urlparse(github_url)
        repo_name = os.path.basename(parsed_url.path).replace('.git', '')

        # 클론 디렉토리 설정
        clone_dir = os.path.join(self.code_db, repo_name)

        if not os.path.exists(clone_dir):
            os.makedirs(clone_dir)

        # GitHub 리포지토리 클론
        result = subprocess.run(['git', 'clone', github_url, clone_dir], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Failed to clone repository: {result.stderr}")
        self.repo_path = clone_dir
        print(f"Repository cloned into {clone_dir}")

    def Git_cloning(self, title:str): ## git clone하는 과정들의 main 함수
        try:
            paper_id = self.get_paper_id_from_title(title, self.ss_api_key)
            if not paper_id:
                print("Paper ID not found.")
            else:
                # ArXiv PDF URL 얻기
                pdf_url = self.get_arxiv_pdf_url(paper_id, self.ss_api_key)
                if not pdf_url:
                    print("ArXiv PDF URL not found.")
                else:
                    # PDF 다운로드
                    pdf_content = self.download_pdf(pdf_url)

                    # PDF에서 GitHub 링크 추출
                    github_links = self.extract_github_links_from_pdf(pdf_content)
                    print(f"GitHub Links: {github_links}")

                    if len(github_links) > 1 :
                        for github_link in github_links :
                            self.clone_github_repository(github_link)
                    elif len(github_links) == 1 :
                        self.clone_github_repository(github_links[0])
            
            # return self.repo_path

        except Exception as e:
            print(e)

######### Github 내부 python 파일들의 전체 function의 이름을 불러오기

    def get_python_files(self):
        """지정된 디렉토리 내의 모든 Python 파일 목록을 반환"""
        python_files = []
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))
        return python_files

    def get_functions_from_file(self):
        """지정된 Python 파일에서 정의된 모든 함수 이름을 반환"""
        with open(self.repo_path, "r", encoding="utf-8") as file:
            tree = ast.parse(file.read(), filename=self.repo_path)

        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        return functions

    def get_all_functions_from_directory(self): ## main 함수
        """지정된 디렉토리 내의 모든 Python 파일에서 정의된 모든 함수 이름을 반환"""
        python_files = self.get_python_files(self.repo_path)
        all_functions = {}

        for file_path in python_files:
            functions = self.get_functions_from_file(self.repo_path)
            all_functions[file_path] = functions

        return all_functions

    
    def get_all_functions(self, repo_path): ### functions 불러오는 전체 함수
        all_functions = self.get_all_functions_from_directory(self.repo_path)

        for file_path, functions in all_functions.items():
            print(f"File: {file_path}")
            for func in functions:
                print(f"  Function: {func}")

        return all_functions

## Code Generation & Analysis

    def generate_code_from_content(self, content):
        prompt=f"Based on the following content from a research paper, write the corresponding Python code that implements the described concept.\n\nPaper Content: \"{content}\""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response

    # Function to extract code from .py files in a directory
    def extract_code_from_repo(self, repo_path):
        code_files = {}
        for subdir, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(subdir, file)
                    with open(file_path, 'r') as f:
                        code_files[file_path] = f.read()
        return code_files

    def split_code_into_functions(self, code):
        pattern = re.compile(r"def\s+\w+\(.*\):")  # Simplistic pattern to match function definitions
        lines = code.split('\n')
        functions = {}
        current_func_name = None
        current_func_code = []

        for line in lines:
            if pattern.match(line):
                if current_func_name:
                    functions[current_func_name] = '\n'.join(current_func_code)
                current_func_name = line.split('(')[0].replace('def ', '').strip()
                current_func_code = [line]
            elif current_func_name:
                current_func_code.append(line)

        if current_func_name:
            functions[current_func_name] = '\n'.join(current_func_code)

        return functions

    def calculate_cosine_similarity(self, code1, code2):
        vectorizer = TfidfVectorizer().fit_transform([code1, code2])
        vectors = vectorizer.toarray()
        return cosine_similarity(vectors)[0, 1]
    
    def code_analysis(self, contents):
        # Paper content presented to LLaMA3
        contents = """To summarize, QLORA has one storage data type (usually 4-bit NormalFloat)and a computation data type (16-bit BrainFloat).
        We dequantize the storage data type to the computation data type to perform the forward and backward pass,
        but we only compute weight gradients for the LoRA parameters which use 16-bit BrainFloat."""

        # Generate code from paper content
        generated_code = self.generate_code_from_content(contents)

        # Extracted code from the repository
        repo_code_files = self.extract_code_from_repo(self.repo_path)

        # Calculate cosine similarity for each file in the repository
        similarity_scores = {}
        for file_path, code in repo_code_files.items():
            functions = self.split_code_into_functions(code)
            for func_name, func_code in functions.items():
                similarity = self.calculate_cosine_similarity(generated_code, func_code)
                similarity_scores[(file_path, func_name)] = similarity

        # Find the file and function with the highest similarity score
        most_relevant_file, most_relevant_function = max(similarity_scores, key=similarity_scores.get)
        highest_similarity_score = similarity_scores[(most_relevant_file, most_relevant_function)]

        # Present the standardized answer
        print(f"Based on the cosine similarity calculations, the most relevant code in the repository to the part of the paper presented is in the file: {most_relevant_file}, function: {most_relevant_function} with a similarity score of {highest_similarity_score}.\n")
        print(f"Here is the most relevant function code:\n\n{repo_code_files[most_relevant_file].split('def ' + most_relevant_function)[1].split('def ')[0]}")