import re
import pandas as pd

from typing import List, Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from constant import model_name

# Initialize LLM
llm = ChatOllama(model=model_name)

class TranscriptProcessor:
    """處理音轉文本和摘要生成的主要類"""
    def __init__(self, llm):
        self.llm = llm
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )

    def preprocess_text(self, text: str) -> str:
        """預處理文本，移除冗餘內容並保留關鍵資訊"""
        # 移除特殊符號和多餘空白
        text = re.sub(r'[\n\t]+', ' ', text)
        text = re.sub(r'\s+', ' ', text) 
        text = text.strip()

        # 移除常見的贅字贅語
        removals = [
            '請不吝點贊 訂閱 轉發 打賞支持明鏡與點點欄目',
            '我們就明天早上8點半早晨財經速解讀再相見',
            '祝各位投資朋友看盤順利操盤愉快'
        ]
        for r in removals:
            text = text.replace(r, '')

        return text

    def split_segments(self, text: str) -> List[str]:
        """將文本分割成較小的片段"""
        return self.text_splitter.split_text(text)

    def generate_summary(self, segments: List[str]) -> str:
        """基於文本片段生成摘要"""
        summaries = []
        for seg in segments:
            prompt = f"請總結以下文字段落的重點:\n{seg}"
            msg = HumanMessage(content=prompt)
            response = self.llm.invoke([msg])
            summaries.append(response.content)

        # 合併並產生最終摘要
        final_prompt = f"請整合以下重點並產生300字以內的摘要:\n{'\n'.join(summaries)}"
        final_msg = HumanMessage(content=final_prompt)
        final_summary = self.llm.invoke([final_msg])

        return final_summary.content
    
class State(TypedDict):
    messages: Annotated[list, add_messages]  # 儲存對話訊息
    transcript: str  # 儲存原始文本
    processed_text: str  # 儲存預處理後的文本
    segments: List[str]  # 儲存分段後的文本
    summary: str  # 儲存最終摘要

processor = TranscriptProcessor(llm)

def preprocess(state: State):
    """預處理文本節點"""
    text = state['transcript']
    processed = processor.preprocess_text(text)
    return {"processed_text": processed}

def split_text(state: State):
    """分割文本節點"""
    text = state['processed_text']
    segments = processor.split_segments(text)
    return {"segments": segments}

def summarize(state: State):
    """生成摘要節點"""
    segments = state['segments']
    summary = processor.generate_summary(segments)
    return {"summary": summary, "messages": [HumanMessage(content=summary)]}

class TranscriptAgent:
    def __init__(self, model_name=model_name):
        self.llm = ChatOllama(model=model_name)
        self.processor = TranscriptProcessor(self.llm)
        self.chain = self._build_chain()
    
    def _build_chain(self):
        """建立並返回工作流程"""
        # 建立工作流程圖
        graph_builder = StateGraph(State)

        # 添加節點
        graph_builder.add_node("preprocess", preprocess)
        graph_builder.add_node("split", split_text) 
        graph_builder.add_node("summarize", summarize)

        # 設定節點關係
        graph_builder.add_edge(START, "preprocess")
        graph_builder.add_edge("preprocess", "split")
        graph_builder.add_edge("split", "summarize")
        graph_builder.add_edge("summarize", END)

        # 編譯工作流程
        return graph_builder.compile()

    def process_transcript(self, transcript: str) -> dict:
        """處理單一逐字稿"""
        return self.chain.invoke({
            "transcript": transcript,
            "messages": [],
            "processed_text": "",
            "segments": [],
            "summary": ""
        })

    def process_batch(self, transcripts: List[str]) -> List[dict]:
        """批次處理多個逐字稿"""
        return [self.process_transcript(t) for t in transcripts]

    def get_chain(self):
        """返回編譯好的工作流程"""
        return self.chain

# 範例使用
if __name__ == "__main__":
    # 初始化 agent
    agent = TranscriptAgent()
    
    # 讀取 CSV 檔案
    df = pd.read_csv('../transcripts_video_v1.1.csv')
    
    # 處理第一個逐字稿作為示例
    sample_transcript = df['transcript'].iloc[0]
    
    # 執行工作流程
    result = agent.process_transcript(sample_transcript)
    
    print("\n最終摘要:")
    print("-" * 50)
    print(result["summary"])