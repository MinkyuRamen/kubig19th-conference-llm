# kubig19th-conference-llm
## 목표 (수정가능)
1. Llama-3-8b로 (아카이브, Semantic Scholoar)를 function call로 넣어서 data generation 후 huggingface에 저장 (original data)
2. 저장된 original data를 stream으로 불러와 정의된 tool (wikipedia search, google search, wolfram alpha, calendar 등)을 이용하여 data augmented (augmented data)
3. [training] augmented data를 이용하여 LoRA fine-tuning (ToolLLaMA)
4. [inference] ToolLLaMA에다가 (아카이브, Semantic Scholoar)를 function call하여 output 생성
## 진행사항
### 16기 박민규
(세부목표) target paper를 공부하기 전 봐야할 premiminaries & target paper 이후에 나온  future works 들에 대한 정보 생성(ft llama) + 시각화 진행
  - semantic scholar api와 sentence transformer를 이용하여 data preprocessing ✅
  - semantic scholar api를 이용하여 preminiaries visualization ✅ >> 조금 더 고급지게 시각화 🏃
  - semantic scholar & archive api를 tool로 사용 >> langchain으로 생성 🏃
  - semantic scholar & archive api 사용하여 llama로 orginal dataset generate 🏃


### 16기 이영노
- ToolFormer 구현

  🏃 `EleutherAI/gpt-j-6B` GPU에 Model Load 이후 ToolFormer `data_generator.py` 실행시, `retrieval_data_{self.num_device}.json` 파일 stack 하는 과정에서 GRAM OOM error 문제 발생 
  --> json 파일 저장 코드 수정

  🏃 차후 `deepspeed` 통한 FT 진행 (`deepspeed` 사용법 공부)

- ToolFormer 개선
  - 배경 : `conceptofmind` huggingface 모델의 Mathematical Reasoning 능력 부족
  - 개선방안 : `from prompts import retrieval_prompt` : prompt 수정
    - e.g. CoT, step-by-step 으로 쪼개는 prompt search 해서 넣어보기 ,tool documentation 넣어주고 zero-shot 으로 시도
    - Sequential Tool Calling 이 가능해야 하는데, 이걸 기존 Toolformer 코드에서 어떻게 수행할지 고민 (LangChain?)
    
