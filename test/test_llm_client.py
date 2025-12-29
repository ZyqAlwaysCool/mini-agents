'''
Description: 
Author: zyq
Date: 2025-12-25 18:11:33
LastEditors: zyq
LastEditTime: 2025-12-26 16:51:10
'''
from mini_agents.core.llm import BaseLLMClient
from mini_agents.core.message import Message
from dotenv import load_dotenv

load_dotenv()

def test_llm_client():
    llm_client = BaseLLMClient()
    test_msg = Message(content="你是谁", role="user")
    answer = llm_client.invoke([test_msg])
    print(answer)

    answer_2 = llm_client.stream_invoke([test_msg])
    for chunk in answer_2:
        print(chunk)
    

if __name__ == "__main__":
    test_llm_client()    