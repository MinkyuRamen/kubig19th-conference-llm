from dotenv import load_dotenv
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
model = ChatOpenAI(model="gpt-3.5-turbo")
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
            if bot_response['output'].split(': ')[-1].endswith('.pdf'):
                img_path = bot_response['output'].split(': ')[-1]
                print('there is a figure path FORMAT1!',img_path)
                
                app.client.files_upload_v2(
                    channels=channel_id,
                    file=img_path,
                    title=img_path.split('/')[0],
                    filetype='pdf',
                    thread_ts=ts  

                )
                app.client.chat_postMessage(
                    channel=channel_id,
                    text=bot_response['output'],
                    thread_ts=ts  
                )
                
            elif ['output'].split(']')[-1].split('(')[-1].split(')')[0].endswith('.pdf'):
                img_path = bot_response['output'].split(']')[-1].split('(')[-1].split(')')[0]
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
                    text=bot_response['output'],
                    thread_ts=ts  
                )
            else:         
                app.client.chat_postMessage(
                    channel=channel_id,
                    text=bot_response['output'],
                    thread_ts=ts,
                )
            
        except:
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


if __name__ == "__main__":  
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])  
    handler.start()
