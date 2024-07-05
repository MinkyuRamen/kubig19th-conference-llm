# kubig19th-conference-llm
### MOTIVATION
Al 연구자들은 논문을 이해하고 정리하는데 LLM을 보조 도구로 사용 하고 있다. 그러나 LLM이 논문과 관련한 구체적인 질의응답을 더 잘 수행하기 위해서는 다음과 같은 한계점을 극복할 필요가 있다.

1. 전체 논문을 직접 다운로드한 후 프롬프트로 입력해야 함

2. 전체 논문을 입력 후 질의응답 시 물필요한 정보가 많아 정확성 (Needle in a Haystack)

3. 현재 사용되는 Tool은 Web Search, PDF Reader 등 제한적이기 때문에 유사 논문 추천 등의 고도화된 요청 처리 불가 우리는 이러한 한계를 극복한 새로운 LLM Agent ASKU을 제안한다.
### Function
1. Load Paper

-반복 호출 방식으로 필요한 부분만 불러올 수 있도록  LLM 의 행동 제어

2. Recommend Paper

-Semantic Scholar API / Cosine similarity  기반 자체 알고리즘 이용

3. Code Match

-논문의  Github  링크로부터 코드 다운  / Cosine similarity  기반 알고리즘 이용

### SPEC
1. **getpaper_v2**

getpaper는 (1) user query에게 제공될 section 선정하고, (2) section들의 context 불러와서 질문에 대한 대답을 하며 (3) optionary하게 generated answer에 가장 알맞는 그림을 visualization하는 크게 3가지 phase으로 구성

2. **recommendpaper**

recommend paper는 (1) user query에 따라 citation paper(target paper를 인용한 논문)/reference paper(target paper가 인용한 논문) 중 어느것을 추천할지 결정 후 semantic scolar api로 context를 불러와서 (2) user가 원하는 개수의 paper를 추천해주는 phase로 구성


3. **code_analysis**

codeanalysis는 (1) LLM이 user query를 보고 이에 맞는 code를 생성한 후 (2) 실제 github code와 이를 비교하여 실제로 어떻게 구현되어 있는지 찾는 phase로 구성

### AsKU light


### HOW TO USE
1) `.env` 파일에 다음 정보를 채운다.

```
MODEL_ID=''
BOT_NAME="lim helper2"
SEMANTIC_SCHOLAR_API_KEY=''
OPENAI_API_KEY=''

SLACK_BOT_TOKEN=''
SLACK_APP_TOKEN=''
signing_secret=''
CHANNEL_ID_beta=''
CHANNEL_ID =''
```

2) `papers_llm/main.py` 를 실행


### Contribution
• Slack에서 배포할 수 있는 형태라 활용성 큼

• 논문 호출 시, 전체 논문 중 사용자 유철에 필요한 Section (figure 포함)만 불러오기 때문에 매우 효율적

• 유사 논문 추천 시, 자체 개발한 유사도 기반 필터링 알고리증을 사용하여 실제로 인용되거나 참조된 논문을 중심으로 추천

• 관련 코드 요청 시, 논문에서 제안된 로직을 기반으로 코드른 검 색하여 호출함으로써 더 정확한 구현 가능
