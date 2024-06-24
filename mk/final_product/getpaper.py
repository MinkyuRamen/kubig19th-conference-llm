import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
import PyPDF2
import re
import os

class GetPaper:
    def __init__(self, ss_api_key, ar5iv_mode = True, path_db='./papers_db', page_limit = 5):
        self.ss_api_key = ss_api_key
        self.ar5iv_mode = ar5iv_mode
        self.path_db= path_db
        self.page_limit = page_limit

    def get_paper_info_by_title(self, title):
        """논문의 제목으로 정보를 가져오는 함수"""
        # Define the API endpoint URL
        url = 'https://api.semanticscholar.org/graph/v1/paper/search?query={}&fields=paperId,title,abstract,authors,citations,fieldsOfStudy,influentialCitationCount,isOpenAccess,openAccessPdf,publicationDate,publicationTypes,references,venue'

        headers = {'x-api-key': self.ss_api_key}
        response = requests.get(url.format(title), headers=headers).json()

        if response.get('data'):
            paper = response['data'][0]
            return paper
        else:
            return None

    def get_ar5iv_url(self, paper):
        "논문의 ar5iv 주소를 받아오는 함수"
        external_ids = paper.get('openAccessPdf', {})
        arxiv_id = external_ids.get('url')
        if 'http' in arxiv_id:
            arxiv_id = arxiv_id.split('/')[-1]
            return f"https://ar5iv.org/abs/{arxiv_id}"
        else:
            return None

    def get_soup_from_url(self, url):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # HTTP 에러가 발생하면 예외를 발생시킴
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup
        except RequestException as e:
            print(f"Error fetching the URL: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
        
    def get_header_from_soup(self, soup):
        # h1부터 h6까지 태그 추출
        headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

        # 태그와 내용을 계층 구조로 저장
        header_list = [(header.name, header.text.strip()) for header in headers]
        title = header_list[0][1]
        header_list = header_list[1:]
        return title, header_list


    def extract_text_under_headers(self, soup, text_list):
        # 결과를 저장할 변수
        results = []

        # 텍스트 리스트를 순회하며 각 텍스트에 해당하는 헤더와 그 아래의 텍스트를 추출
        for text in text_list:
            header_tag = soup.find(lambda tag: tag.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'] and text in tag.get_text())
            if header_tag:
                header_text = header_tag.get_text(strip=False)
                header_level = int(header_tag.name[1])
                current_header = {'tag': header_tag.name, 'text': header_text, 'subsections': []}
                results.append(current_header)

                next_element = header_tag.find_next_sibling()
                while next_element:
                    if next_element.name and next_element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                        next_level = int(next_element.name[1])
                        if next_level <= header_level:
                            break

                        # If it's a tag and within our header range
                        if next_element.name and next_element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                            next_text = next_element.get_text(strip=False)
                            next_subheader = {'tag': next_element.name, 'text': next_text, 'subsections': []}
                            current_header['subsections'].append(next_subheader)
                            current_header = next_subheader
                            header_level = next_level
                    else:
                        if 'subsections' not in current_header:
                            current_header['subsections'] = []
                        current_header['subsections'].append({'tag': 'p', 'text': next_element.get_text(strip=False)})

                    next_element = next_element.find_next_sibling()

        content = ''
        for x in results:
            content += x['text']
            for y in x['subsections']:
                content += y['text']
        content = re.sub(r'\n{3,}', '\n\n', content) # 3번 이상 \n 이 연속되면 2번으로 줄이기
        return content

    def list_section(self, header_list):
        section_list = ''
        for tag, text in header_list:
            level = int(tag[1])  # 태그에서 레벨을 추출 (h1 -> 1, h2 -> 2, ..)
            section_list += '  ' * (level - 1) + text +'\n'
        return section_list


    def download_pdf(self, arxiv_id):
        """
        Download the PDF of a paper given its arXiv ID, if it does not already exist.
        """
        if not os.path.exists(self.path_db):
            os.makedirs(self.path_db)
        
        file_path = os.path.join(self.path_db, f'{arxiv_id}.pdf')
        
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

    def read_pdf(self, arxiv_id, end_page=None):
        pdf_content = ""
        file_path = f'{self.path_db}/{arxiv_id}.pdf'
    
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                
                if end_page is None or end_page > total_pages:
                    end_page = total_pages

                for page_num in range(1, end_page):
                    page = reader.pages[page_num-1]
                    pdf_content += page.extract_text()
                    if page_num == self.page_limit:
                        print('Page limit reached at', self.page_limit + 1)
                        break

        except FileNotFoundError:
            return f"Error: The file {file_path} does not exist."
        except Exception as e:
            return f"An error occurred while reading the file: {e}"

        pdf_content = re.sub(r'\s+', ' ', pdf_content).strip()
        return pdf_content


    def load_paper(self, title:str, sections:list=None):
        '''
        INPUT : title of paper,
                list of sections in paper
        OUTPUT : text of the paper
        '''
        paper = self.get_paper_info_by_title(title)
        url = self.get_ar5iv_url(paper)
        soup = self.get_soup_from_url(url) if self.ar5iv_mode else None
        if (soup):
            title, header_list = self.get_header_from_soup(soup)
            if sections == None:
                sections_list = self.list_section(header_list)
                instruction_for_agent = f'Here is the title and section of the paper\ntitle\n{title}\nsections\n{sections_list}\n\n Use the \'loadpaper\' tool again, specifying the exact sections you want to view in detail.'
                return instruction_for_agent
            else:
                return self.extract_text_under_headers(soup, sections)
        else: # case for ar5iv is not exist or request error
            arxiv_id = url.split('/')[-1]
            download_path = self.download_pdf(arxiv_id)
            pdf_content = self.read_pdf(arxiv_id)
            return pdf_content


