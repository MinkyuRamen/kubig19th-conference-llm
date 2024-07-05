from dotenv import load_dotenv
from httpx import NetworkError
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.messages import SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_openai_tools_agent
import torch
import os  
from pprint import pprint
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import tool_pool as tp

import warnings
warnings.filterwarnings('ignore')

dotenv_path = '/root/limlab/lim_helper_v2/.env'
load_dotenv(dotenv_path)

api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
channel_id = os.environ.get("CHANNEL_ID")


app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ["signing_secret"]
) 
client = WebClient(os.environ.get("SLACK_BOT_TOKEN"))


# load model
# model = ChatOpenAI(model="gpt-3.5-turbo")
# load tool
tools = [tp.loadpaper, tp.recommendpaper, tp.code_matching]
# load Agent prompt
prompt = hub.pull("hwchase17/openai-tools-agent")


@app.event("app_mention")
def handle_message_events(event, message, say):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    ts = event['ts']
    text = event['text'] + '대답은 한글로 해줘'
    
    agent = create_openai_tools_agent(llm, tools, prompt=prompt)

    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

    try:
        try:
            bot_response = agent_executor.invoke({"input": text, 'chat_history': memory})
        except Exception as e:
            error_message = str(e)
            if 'context_length_exceeded' in error_message:
                raise NetworkError(f'Error 현재 불러오고자 하는 토큰 길이가 LLM이 불러올 수 있는 최대 토큰 길이를 초과했습니다. \n\n세부내용 : {e}')
            elif 'Semantic Scholar API' in error_message:
                raise NetworkError(f'Semantic Scholar API 오류입니다 잠시 후 다시 시도해 주세요. \n\n세부내용 : {e}')

        try:
            if bot_response['output'].split('(')[-1].split(')')[0].split('\n')[0].split('sandbox:')[-1].endswith(('.pdf','.png')) and bot_response['output'].split('(')[-1].split(')')[0].split('\n')[0].split('sandbox:')[-1].startswith('/'):
                img_path = bot_response['output'].split('(')[-1].split(')')[0].split('\n')[0].split('sandbox:')[-1]
                print('there is a figure path FORMAT2!',img_path)

                app.client.files_upload_v2(
                    channels=channel_id,
                    file=img_path,
                    title=img_path.split('/')[0],
                    filetype='pdf',
                    thread_ts=ts  
                )
                app.client.chat_postMessage(
                    channel=channel_id,
                    text=' '.join(bot_response['output'].split('\n')[:-1]),
                    thread_ts=ts  
                )
                
            elif bot_response['output'].split(': ')[-1].split('\n')[0].endswith(('.pdf','.png')) and bot_response['output'].split(': ')[-1].split('\n')[0].startswith('/'):
                img_path = bot_response['output'].split(': ')[-1].split('\n')[0]
                print('there is a figure path FORMAT1!',img_path)
                
                app.client.files_upload_v2(
                    channels=channel_id,
                    file=img_path,
                    title=img_path.split('/')[0],
                    # filetype='pdf',
                    thread_ts=ts  

                )
                app.client.chat_postMessage(
                    channel=channel_id,
                    text=' '.join(bot_response['output'].split('\n')[:-1]),
                    thread_ts=ts  
                )
            else:
                print('there isnot figue 1!')
                app.client.chat_postMessage(
                    channel=channel_id,
                    text=bot_response['output'],
                    thread_ts=ts,
                )
            
        except:
            print('there isnot figue 2!')
            app.client.chat_postMessage(
            channel=channel_id,
            text=bot_response['output'],
            thread_ts=ts,
            )

    except Exception as e:    
        error_message = f"에러 발생: {str(e)}"
        app.client.chat_postMessage(
                channel=channel_id,
                text=error_message,
                thread_ts=ts,
            )


def send_guideline_message(channel_id, user_id):
    app.client.chat_postEphemeral(
        channel=channel_id,
        user=user_id,
        blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "안녕하세요~! 저는 논문 읽기를 도와주는 슬랙 앱 AsKU_paper입니다. 🤖 지금부터 앱 사용법을 차근차근 알려드리겠습니다 😄:"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*1️⃣ 🌷@로 호출하기🌷* AsKU_paper에게 질문하기 위해서는 항상 @AsKU_paper를 통해 호출해주어야 합니다! 도움을 요청하기 위한 최소한의 예의를 지켜주세요 🥲"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*2️⃣ 🌷논문 호출하기🌷* 궁금한 논문을 불러오기 위해 따옴표('') 안에 논문 이름을 적어주세요!('ToolLLM') 가끔 DDPM처럼 여러 논문이 비슷한 약어를 사용하는 경우에는 AsKU가 헷갈릴 수도 있어요 😂 따라서 논문의 full name이나 논문의 arxiv id를 같이 입력하는 것도 권장드립니다."
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*3️⃣ 🌷그림 호출하기🌷* 논문 내의 이미지(🌁)을 함께 참고하고 싶다면, ‘그림과 함께 설명해줘’라고 물어보세요! AsKU가 그림을 함께 출력해줄 거예요!"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*4️⃣ 🌷코드 찾아보기🌷* 논문 내용이 구현된 코드(🧑‍💻)를 찾고 싶다면, `github link`와 함께 '이 내용을 어떻게 구현할 수 있어?’라고 물어보세요! AsKU가 깃허브에서 적절한 함수를 찾아준답니다!"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*5️⃣ 🌷논문 추천받기🌷* 유사한 논문을 추천받고 싶다면, '유사한 논문을 n개 추천해줘!'라고 물어보세요! AsKU가 자체 알고리즘에 따라 적절한 논문을 추천해준답니다!"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*6️⃣ ⚠️질문은 구체적으로!⚠️* AsKU는 토큰 제한이 있기 때문에 전체 논문을 참조하기보다는 일부 section을 통해 답을 찾아요! 구체적으로 질문할수록 gpt4보다 더 좋은 답변을 얻을 수 있답니다!"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*7️⃣ ⚠️한 번에 하나씩 질문하기!⚠️* 여러 개의 질문이 몰리면 AsKU가 힘들어 해요! 🤯 질문 하나 당 30초~1분 정도 요소되니 앞사람의 대답이 끝날 때까지 인내심을 가지고 기다려주세요!"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*[예시 질문 모음]* \n\n • 'DDIM'을 읽기 전에 읽을만한 논문 4편을 추천해줘 \n\n • 'MobileNets'(1704.04861) 의 논문에서 아키텍쳐가 어떻게 구성되어 있는지 그림과 함께 설명해줘 \n\n • 'Attention is all you need'(1706.03762)에서 positional encoding에 대해 그림과 함께 설명해줘 \n\n • 'Attention is All You Need'(1706.03762) 의 논문에서 'To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence aligned RNNs or convolution.' 이 내용을 어떻게 구현할 수 있어? 깃허브 코드는 다음과 같아. https://github.com/nawnoes/pytorch-transformer"
                }
            },
            {
                "type": "divider"
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "👀 설명서를 다시 보고 싶다면 `/view guide`를 입력해주세요. 앱 사용 중 발생하는 문제사항은 @17기_임청수에게 DM 주시면 됩니다!"
                    }
                ]
            }
        ]
    )



@app.event("member_joined_channel")
def handle_member_joined_channel(event, say):
    user_id = event["user"]
    send_guideline_message(channel_id, user_id)

@app.command("/view_guide")
def handle_view_guide(ack, body):
    ack()
    user_id = body["user_id"]
    send_guideline_message(channel_id, user_id)


@app.event("app_home_opened")
def update_home_tab(client, event, logger):
  try:
    # views.publish is the method that your app uses to push a view to the Home tab
    client.views_publish(
      # the user that opened your app's app home
      user_id=event["user"],
      # the view object that appears in the app home
      view={
        "type": "home",
        "callback_id": "home_view",
 
        # body of the view
        "blocks": [
          {
            "type": "section",
            "text": {
              "type": "mrkdwn",
              "text": "AsKU_paper에 오신 것을 환영합니다 :)"
            }
          },
          {
            "type": "divider"
          },
          {
            "type": "section",
            "text": {
              "type": "mrkdwn",
              "text": "지금 바로 검색창에 asku-paper 채널을 검색해서 강력한 논문 질의응답 앱 AsKU_paper를 경험해보세요!"
            }
          }
        ]
      }
    )
  
  except Exception as e:
    logger.error(f"Error publishing home tab: {e}")


if __name__ == "__main__":  
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])  
    handler.start()