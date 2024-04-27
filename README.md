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
