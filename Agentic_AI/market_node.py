"""
基於 ReAct 模式的核心工作流程：

analyze (分析) -> think (思考) -> decide (決策)
允許在 think 階段進行多輪思考-行動循環
智能決策機制：

每個思考步驟都會產生一個 ThoughtProcess 對象，記錄思考內容和行動
agent 可以根據需要自主決定是否調用外部工具獲取更多資訊
支援三種資料獲取行動：價格(price)、成交量(volume)和技術指標(technical)
彈性的工具調用：

使用 yfinance 獲取實時市場數據
使用 ta-lib 計算技術指標如 RSI、MACD
所有工具調用結果都會被加入到 context 中供後續決策參考
持續學習和推理：

agent 會記錄所有的思考過程和觀察結果
每個決策都是基於累積的資訊和多輪思考
"""

from typing import List, Dict, Any, Tuple, Optional, TypedDict
from enum import Enum
from dataclasses import dataclass
import json
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END, START
import yfinance as yf
import pandas as pd
import ta

class MarketSentiment(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

@dataclass
class ThoughtProcess:
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict] = None
    observation: Optional[str] = None

class MarketState(TypedDict):
    summary: str  # 文本摘要
    sentiment: MarketSentiment  # 市場情緒
    thoughts: List[ThoughtProcess]  # 思考過程記錄
    decisions: List[str]  # 決策記錄
    context: Dict[str, Any]  # 額外的市場數據

class ReActMarketAgent:
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.llm = ChatOllama(model=model_name)
        self.chain = self._build_chain()
        
    def _build_chain(self) -> StateGraph:
        """建立 ReAct 工作流程"""
        workflow = StateGraph(MarketState)
        
        # 添加節點
        workflow.add_node("analyze", self._analyze_content)
        workflow.add_node("think", self._think)
        workflow.add_node("decide", self._make_decision)
        
        # 設定工作流程
        workflow.add_edge(START, "analyze")
        workflow.add_edge("analyze", "think")
        workflow.add_edge("think", "decide")
        workflow.add_edge("decide", END)
        
        # 允許思考-行動循環
        workflow.add_edge("think", "think")
        
        return workflow.compile()

    def _analyze_content(self, state: MarketState) -> Dict:
        """分析摘要內容，評估市場情緒"""
        prompt = f"""請分析以下市場摘要，判斷市場情緒（看多/看空/中性）並提供理由：
        
        {state['summary']}
        
        請以 JSON 格式回答，包含：
        1. sentiment: 市場情緒 (BULLISH/BEARISH/NEUTRAL)
        2. reasoning: 理由分析
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        analysis = json.loads(response.content)
        
        return {
            "sentiment": MarketSentiment(analysis["sentiment"].lower()),
            "thoughts": [ThoughtProcess(
                thought=f"Initial analysis: {analysis['reasoning']}"
            )]
        }

    def _think(self, state: MarketState) -> Dict:
        """進行思考並決定是否需要額外資訊"""
        current_context = state["context"]
        last_thought = state["thoughts"][-1]
        
        prompt = f"""基於目前資訊：
        1. 市場情緒: {state['sentiment'].value}
        2. 上一個想法: {last_thought.thought}
        3. 已有的市場數據: {json.dumps(current_context, indent=2)}
        
        請思考是否需要額外資訊？如果需要，請說明需要什麼資訊。
        
        請以 JSON 格式回答：
        1. thought: 思考過程
        2. need_action: 是否需要獲取更多資訊 (true/false)
        3. action: 如果需要行動，指定要獲取的資訊類型 ("price", "volume", "technical")
        4. symbols: 相關的股票代碼列表
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        thinking = json.loads(response.content)
        
        new_thought = ThoughtProcess(
            thought=thinking["thought"],
            action=thinking.get("action"),
            action_input={"symbols": thinking.get("symbols", [])} if thinking["need_action"] else None
        )
        
        # 如果需要行動，執行相應的資料獲取
        if thinking["need_action"]:
            if thinking["action"] == "price":
                data = self._fetch_price_data(thinking["symbols"])
            elif thinking["action"] == "technical":
                data = self._fetch_technical_indicators(thinking["symbols"])
            else:
                data = self._fetch_volume_data(thinking["symbols"])
            
            new_thought.observation = str(data)
            state["context"].update({thinking["action"]: data})
        
        return {
            "thoughts": state["thoughts"] + [new_thought]
        }

    def _make_decision(self, state: MarketState) -> Dict:
        """基於所有資訊作出最終決策"""
        all_thoughts = "\n".join(f"- {t.thought}" for t in state["thoughts"])
        context_data = json.dumps(state["context"], indent=2)
        
        prompt = f"""基於以下資訊，請作出投資決策建議：
        
        市場情緒: {state['sentiment'].value}
        思考過程:
        {all_thoughts}
        
        市場數據:
        {context_data}
        
        請提供具體的投資建議，包含:
        1. 建議操作（買入/賣出/觀望）
        2. 目標標的
        3. 理由說明
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return {
            "decisions": state["decisions"] + [response.content]
        }

    def _fetch_price_data(self, symbols: List[str]) -> Dict:
        """獲取價格數據"""
        data = {}
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period="1mo")
                data[symbol] = {
                    "current": hist["Close"][-1],
                    "change": (hist["Close"][-1] - hist["Close"][0]) / hist["Close"][0]
                }
            except Exception as e:
                data[symbol] = {"error": str(e)}
        return data

    def _fetch_technical_indicators(self, symbols: List[str]) -> Dict:
        """獲取技術指標"""
        data = {}
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period="3mo")
                
                # 計算技術指標
                hist["RSI"] = ta.momentum.RSIIndicator(hist["Close"]).rsi()
                hist["MACD"] = ta.trend.MACD(hist["Close"]).macd()
                
                data[symbol] = {
                    "RSI": hist["RSI"][-1],
                    "MACD": hist["MACD"][-1]
                }
            except Exception as e:
                data[symbol] = {"error": str(e)}
        return data

    def _fetch_volume_data(self, symbols: List[str]) -> Dict:
        """獲取成交量數據"""
        data = {}
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period="1mo")
                data[symbol] = {
                    "avg_volume": hist["Volume"].mean(),
                    "volume_trend": (hist["Volume"][-5:].mean() / hist["Volume"][:-5].mean() - 1)
                }
            except Exception as e:
                data[symbol] = {"error": str(e)}
        return data

    def analyze_market(self, summary: str) -> Dict[str, Any]:
        """執行完整的市場分析流程"""
        initial_state = {
            "summary": summary,
            "sentiment": MarketSentiment.NEUTRAL,
            "thoughts": [],
            "decisions": [],
            "context": {}
        }
        
        return self.chain.invoke(initial_state)

# 使用範例
if __name__ == "__main__":
    agent = ReActMarketAgent()
    summary = """
    今日市場顯示多個看漲信號：
    1. 台積電ADR大漲7%
    2. 外資持續買超
    3. 美股科技股普遍走強
    然而需要注意以下風險：
    1. 美債殖利率持續上升
    2. 通膨數據仍高於預期
    """
    
    result = agent.analyze_market(summary)
    
    print("\n分析結果:")
    print("-" * 50)
    print(f"市場情緒: {result['sentiment'].value}")
    print("\n思考過程:")
    for thought in result["thoughts"]:
        print(f"- {thought.thought}")
        if thought.action:
            print(f"  行動: {thought.action}")
            print(f"  觀察: {thought.observation}")
    
    print("\n決策建議:")
    for decision in result["decisions"]:
        print(f"- {decision}")

# example usage
from Agentic_AI.market_node import ReActMarketAgent

agent = ReActMarketAgent()
result = agent.analyze_market("""
    台積電今日營收報告優於預期，
    外資大舉買超超過50億，
    但需注意美國CPI數據即將公布
""")