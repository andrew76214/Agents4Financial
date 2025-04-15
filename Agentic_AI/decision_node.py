from typing import List, Dict, Any, Optional, TypedDict
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime
import pandas as pd
import numpy as np
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END, START

class DecisionType(Enum):
    BUY = "買入"
    SELL = "賣出"
    HOLD = "觀望"

class RiskLevel(Enum):
    LOW = "低風險"
    MEDIUM = "中等風險"
    HIGH = "高風險"

@dataclass
class MarketContext:
    """市場環境相關資訊"""
    market_sentiment: str
    key_indices: Dict[str, float]
    macro_indicators: Dict[str, Any]
    trading_volume: Dict[str, Any] = None
    market_keywords: List[str] = None
    time_series_data: Dict[str, List[float]] = None
    global_correlations: Dict[str, float] = None

@dataclass
class StockAnalysis:
    """個股分析資訊"""
    symbol: str
    current_price: float
    technical_indicators: Dict[str, float]
    fundamental_metrics: Dict[str, Any]
    risk_factors: List[str]

@dataclass
class PositionStrategy:
    """持倉策略建議"""
    position_size: float  # 建議持倉比例 (0-1)
    stop_loss: float     # 停損價位
    take_profit: float   # 獲利目標
    time_horizon: str    # 建議持有期間

@dataclass
class InvestmentDecision:
    """投資決策完整資訊"""
    timestamp: datetime
    decision_type: DecisionType
    risk_level: RiskLevel
    target_symbol: str
    strategy: PositionStrategy
    reasoning: List[str]
    supporting_data: Dict[str, Any]

@dataclass 
class ThoughtProcess:
    """思考過程記錄"""
    thought: str
    action: Optional[str] = None  
    action_input: Optional[Dict] = None
    observation: Optional[str] = None

class DecisionState(TypedDict):
    """決策狀態"""
    stock_analysis: Any  # StockAnalysis
    market_context: Any  # MarketContext
    sentiment: str  # 市場情緒
    thoughts: List[ThoughtProcess]  # 思考過程記錄
    context: Dict[str, Any]  # 額外的分析數據
    decision: Optional[Any] = None  # 最終決策

class RiskManager:
    """風險管理模組"""
    def __init__(self):
        self.max_position_size = 0.2  # 單一標的最大持倉比例
        self.max_sector_exposure = 0.4  # 單一產業最大持倉比例
        self.min_liquidity = 1000000  # 最小日均成交量
        
    def assess_risk_level(self, analysis: StockAnalysis, market_context: MarketContext) -> RiskLevel:
        risk_score = 0
        
        # Handle potential None values
        technical_indicators = analysis.technical_indicators or {}
        fundamental_metrics = analysis.fundamental_metrics or {}
        
        # 技術指標風險評估
        if technical_indicators.get("RSI", 50) > 70:
            risk_score += 1
        if technical_indicators.get("volatility", 0) > 0.3:
            risk_score += 1
            
        # 基本面風險評估
        if fundamental_metrics.get("debt_ratio", 0) > 0.7:
            risk_score += 1
        if fundamental_metrics.get("current_ratio", 1) < 1:
            risk_score += 1
            
        # 市場環境風險評估
        if getattr(market_context, "market_sentiment", "") == "bearish":
            risk_score += 1
        
        if risk_score >= 3:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
    
    def validate_position_size(self, suggested_size: float, risk_level: RiskLevel) -> float:
        """根據風險等級調整持倉比例"""
        max_size = self.max_position_size
        if risk_level == RiskLevel.HIGH:
            max_size *= 0.5
        elif risk_level == RiskLevel.MEDIUM:
            max_size *= 0.75
        
        return min(suggested_size, max_size)

class DecisionAgent:
    """投資決策生成器"""
    def __init__(self, model_name: str = "llama2:3b"):
        self.llm = ChatOllama(model=model_name)
        self.risk_manager = RiskManager()
        self.chain = self._build_chain()
        
    def _build_chain(self) -> StateGraph:
        """建立 ReAct 工作流程"""
        workflow = StateGraph(DecisionState)
        
        # 添加節點
        workflow.add_node("analyze", self._analyze_stock)
        workflow.add_node("think", self._think)
        workflow.add_node("decide", self._make_final_decision)
        
        # 設定工作流程
        workflow.add_edge(START, "analyze")
        workflow.add_edge("analyze", "think")
        workflow.add_edge("think", "decide")
        workflow.add_edge("decide", END)
        
        # 允許思考-行動循環
        workflow.add_edge("think", "think")
        
        return workflow.compile()

    def _analyze_stock(self, state: DecisionState) -> Dict:
        """分析股票狀態"""
        analysis = state["stock_analysis"]
        market = state["market_context"]
        
        prompt = f"""請分析以下股票資訊，判斷目前狀態：
        
        股票代號: {analysis.symbol}
        技術指標: {json.dumps(analysis.technical_indicators, ensure_ascii=False)}
        基本面指標: {json.dumps(analysis.fundamental_metrics, ensure_ascii=False)}
        風險因素: {json.dumps(analysis.risk_factors, ensure_ascii=False)}
        市場情緒: {market.market_sentiment}
        
        請以下列格式回答：
        {{
            "sentiment": "樂觀/中性/悲觀",
            "reasoning": "分析理由"
        }}
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            analysis = json.loads(content)
            
            return {
                "sentiment": analysis["sentiment"],
                "thoughts": [ThoughtProcess(
                    thought=f"Initial analysis: {analysis['reasoning']}"
                )]
            }
        except Exception as e:
            return {
                "sentiment": "中性",
                "thoughts": [ThoughtProcess(
                    thought=f"Error in analysis: {str(e)}. Defaulting to neutral."
                )]
            }

    def _think(self, state: DecisionState) -> Dict:
        """進行思考並決定是否需要額外資訊"""
        current_context = state["context"]
        last_thought = state["thoughts"][-1]
        
        prompt = f"""基於目前資訊：
        1. 分析結果: {last_thought.thought}
        2. 已有數據: {json.dumps(current_context, indent=2, ensure_ascii=False)}
        
        請思考是否需要額外資訊？需要什麼資訊？
        
        請以下列格式回答：
        {{
            "thought": "思考過程",
            "need_action": true/false,
            "action": "technical/fundamental/market",
            "aspect": ["特定指標或面向"]
        }}
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            thinking = json.loads(content)
            
            new_thought = ThoughtProcess(
                thought=thinking["thought"],
                action=thinking.get("action") if thinking.get("need_action") else None,
                action_input={"aspects": thinking.get("aspect", [])} if thinking.get("need_action") else None
            )
            
            if thinking.get("need_action"):
                # 根據請求獲取額外資訊
                if thinking["action"] == "technical":
                    data = self._analyze_technical_aspects(state["stock_analysis"], thinking.get("aspect", []))
                elif thinking["action"] == "fundamental":
                    data = self._analyze_fundamental_aspects(state["stock_analysis"], thinking.get("aspect", []))
                else:  # market
                    data = self._analyze_market_aspects(state["market_context"], thinking.get("aspect", []))
                
                new_thought.observation = str(data)
                state["context"].update({thinking["action"]: data})
            
            return {
                "thoughts": state["thoughts"] + [new_thought]
            }
            
        except Exception as e:
            new_thought = ThoughtProcess(
                thought=f"Error in thinking process: {str(e)}. Will proceed with available information."
            )
            return {
                "thoughts": state["thoughts"] + [new_thought]
            }

    def _make_final_decision(self, state: DecisionState) -> Dict:
        """基於所有資訊作出最終決策"""
        all_thoughts = "\n".join(f"- {t.thought}" for t in state["thoughts"])
        context_data = json.dumps(state["context"], indent=2, ensure_ascii=False)
        
        prompt = f"""基於以下資訊，請作出投資決策：
        
        市場情緒: {state['sentiment']}
        思考過程:
        {all_thoughts}
        
        分析數據:
        {context_data}
        
        請提供具體的投資建議，包含:
        1. 建議操作（買入/賣出/觀望）
        2. 理由說明
        3. 風險評估
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        # 轉換成正式決策
        if "買入" in response.content:
            decision_type = DecisionType.BUY
        elif "賣出" in response.content:
            decision_type = DecisionType.SELL
        else:
            decision_type = DecisionType.HOLD
            
        risk_level = self.risk_manager.assess_risk_level(
            state["stock_analysis"], 
            state["market_context"]
        )
        
        strategy = self._generate_position_strategy(
            state["stock_analysis"],
            risk_level,
            decision_type
        )
        
        decision = InvestmentDecision(
            timestamp=datetime.now(),
            decision_type=decision_type,
            risk_level=risk_level,
            target_symbol=state["stock_analysis"].symbol,
            strategy=strategy,
            reasoning=[t.thought for t in state["thoughts"]],
            supporting_data=state["context"]
        )
        
        return {
            "decision": decision,
            "thoughts": state["thoughts"]
        }

    def _analyze_technical_aspects(self, analysis: StockAnalysis, aspects: List[str]) -> Dict:
        """分析技術面特定方面"""
        result = {}
        for aspect in aspects:
            if aspect in analysis.technical_indicators:
                result[aspect] = {
                    "value": analysis.technical_indicators[aspect],
                    "interpretation": self._interpret_technical_indicator(aspect, analysis.technical_indicators[aspect])
                }
        return result
    
    def _analyze_fundamental_aspects(self, analysis: StockAnalysis, aspects: List[str]) -> Dict:
        """分析基本面特定方面"""
        result = {}
        for aspect in aspects:
            if aspect in analysis.fundamental_metrics:
                result[aspect] = {
                    "value": analysis.fundamental_metrics[aspect],
                    "interpretation": self._interpret_fundamental_metric(aspect, analysis.fundamental_metrics[aspect])
                }
        return result
    
    def _analyze_market_aspects(self, market_context: MarketContext, aspects: List[str]) -> Dict:
        """分析市場特定方面"""
        result = {}
        for aspect in aspects:
            if aspect == "sentiment":
                result[aspect] = market_context.market_sentiment
            elif aspect in market_context.key_indices:
                result[aspect] = market_context.key_indices[aspect]
        return result
        
    def _interpret_technical_indicator(self, indicator: str, value: float) -> str:
        """解釋技術指標含義"""
        if indicator == "RSI":
            if value > 70:
                return "嚴重超買"
            elif value > 60:
                return "偏多"
            elif value < 30:
                return "嚴重超賣"
            elif value < 40:
                return "偏空"
            return "中性"
        elif indicator == "MACD":
            return "多頭" if value > 0 else "空頭"
        return "需要進一步分析"
    
    def _interpret_fundamental_metric(self, metric: str, value: float) -> str:
        """解釋基本面指標含義"""
        if metric == "PE":
            if value < 10:
                return "低估值"
            elif value > 30:
                return "高估值"
            return "合理估值"
        elif metric == "ROE":
            if value > 0.15:
                return "獲利能力佳"
            elif value < 0.05:
                return "獲利能力差"
            return "獲利能力中等"
        return "需要進一步分析"

    def generate_decision_with_reflection(self, 
                                      stock_analysis: StockAnalysis,
                                      market_context: MarketContext) -> InvestmentDecision:
        """使用反思機制生成投資決策"""
        initial_state = {
            "stock_analysis": stock_analysis,
            "market_context": market_context,
            "sentiment": "neutral",
            "thoughts": [],
            "context": {},
            "decision": None
        }
        
        final_state = self.chain.invoke(initial_state)
        return final_state["decision"]

    def generate_decision(self, 
                         stock_analysis: StockAnalysis,
                         market_context: MarketContext,
                         portfolio_context: Optional[Dict] = None) -> InvestmentDecision:
        """生成投資決策"""
        # 評估風險等級
        risk_level = self.risk_manager.assess_risk_level(stock_analysis, market_context)
        
        # 根據分析生成初步決策
        decision_type = self._determine_decision_type(stock_analysis, market_context)
        
        # 生成持倉策略
        strategy = self._generate_position_strategy(
            stock_analysis, 
            risk_level,
            decision_type
        )
        
        # 整理決策依據
        reasoning = self._generate_reasoning(
            stock_analysis,
            market_context,
            decision_type,
            risk_level
        )
        
        # 建立決策物件
        decision = InvestmentDecision(
            timestamp=datetime.now(),
            decision_type=decision_type,
            risk_level=risk_level,
            target_symbol=stock_analysis.symbol,
            strategy=strategy,
            reasoning=reasoning,
            supporting_data={
                "technical": stock_analysis.technical_indicators,
                "fundamental": stock_analysis.fundamental_metrics,
                "market": market_context.key_indices
            }
        )
        
        return decision
    
    def _determine_decision_type(self, 
                               analysis: StockAnalysis,
                               market_context: MarketContext) -> DecisionType:
        """決定交易方向"""
        # 整合技術面信號
        technical_score = 0
        
        # Safely handle potentially None technical_indicators
        if analysis and analysis.technical_indicators:
            rsi = analysis.technical_indicators.get("RSI", 50)
            macd = analysis.technical_indicators.get("MACD", 0)
            
            if rsi < 30:
                technical_score += 1
            if macd > 0:
                technical_score += 1
            
        # 整合市場情緒
        market_score = 0
        if market_context and market_context.market_sentiment == "bullish":
            market_score += 1
        
        # 綜合判斷
        total_score = technical_score + market_score
        if total_score >= 2:
            return DecisionType.BUY
        elif total_score <= -1:
            return DecisionType.SELL
        return DecisionType.HOLD
    
    def _generate_position_strategy(self,
                                  analysis: StockAnalysis,
                                  risk_level: RiskLevel,
                                  decision_type: DecisionType) -> PositionStrategy:
        """生成持倉策略"""
        # 計算建議持倉比例
        base_position_size = 0.1  # 基礎持倉比例
        if decision_type == DecisionType.BUY:
            # 根據風險等級調整持倉比例
            adjusted_size = self.risk_manager.validate_position_size(
                base_position_size,
                risk_level
            )
            
            # 設定停損和獲利目標
            stop_loss = analysis.current_price * 0.95  # 5%停損
            take_profit = analysis.current_price * 1.15  # 15%獲利目標
            
            return PositionStrategy(
                position_size=adjusted_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                time_horizon="中期(1-3個月)"
            )
        else:
            return PositionStrategy(
                position_size=0,
                stop_loss=0,
                take_profit=0,
                time_horizon="不適用"
            )
    
    def _generate_reasoning(self,
                          analysis: StockAnalysis,
                          market_context: MarketContext,
                          decision_type: DecisionType,
                          risk_level: RiskLevel) -> List[str]:
        """生成決策依據說明"""
        reasoning = []
        
        # 技術面分析
        tech_reasons = []
        if analysis.technical_indicators.get("RSI") < 30:
            tech_reasons.append("RSI指標顯示超賣")
        if analysis.technical_indicators.get("MACD", 0) > 0:
            tech_reasons.append("MACD呈現多頭排列")
        if tech_reasons:
            reasoning.append(f"技術面分析：{'、'.join(tech_reasons)}")
            
        # 基本面分析
        fund_reasons = []
        if analysis.fundamental_metrics.get("PE") < 15:
            fund_reasons.append("本益比低於產業平均")
        if analysis.fundamental_metrics.get("ROE", 0) > 0.15:
            fund_reasons.append("ROE優於同業")
        if fund_reasons:
            reasoning.append(f"基本面分析：{'、'.join(fund_reasons)}")
            
        # 市場環境
        reasoning.append(f"市場情緒：{market_context.market_sentiment}")
        
        # 風險提示
        risk_warning = f"風險等級：{risk_level.value}"
        if risk_level == RiskLevel.HIGH:
            risk_warning += "，建議謹慎操作，嚴格執行停損"
        reasoning.append(risk_warning)
        
        return reasoning
    
    def generate_report(self, decision: InvestmentDecision) -> str:
        """生成決策報告"""
        report = []
        report.append("=== 投資決策報告 ===")
        report.append(f"決策時間: {decision.timestamp}")
        report.append(f"目標商品: {decision.target_symbol}")
        report.append(f"交易方向: {decision.decision_type.value}")
        report.append(f"風險等級: {decision.risk_level.value}")
        
        report.append("\n--- 策略建議 ---")
        report.append(f"建議持倉比例: {decision.strategy.position_size*100:.1f}%")
        if decision.decision_type == DecisionType.BUY:
            report.append(f"停損價位: {decision.strategy.stop_loss:.2f}")
            report.append(f"目標獲利: {decision.strategy.take_profit:.2f}")
        report.append(f"建議持有期間: {decision.strategy.time_horizon}")
        
        report.append("\n--- 決策依據 ---")
        for reason in decision.reasoning:
            report.append(f"• {reason}")
        
        report.append("\n--- 參考數據 ---")
        report.append("技術指標:")
        for k, v in decision.supporting_data["technical"].items():
            report.append(f"  • {k}: {v}")
            
        report.append("基本面指標:")
        for k, v in decision.supporting_data["fundamental"].items():
            report.append(f"  • {k}: {v}")
        
        return "\n".join(report)

# 使用範例
if __name__ == "__main__":
    # 創建決策代理
    agent = DecisionAgent()
    
    # 模擬市場環境
    market_context = MarketContext(
        market_sentiment="bullish",
        key_indices={"大盤指數": 17500, "產業指數": 1200},
        macro_indicators={"GDP成長率": 2.5, "通膨率": 2.1}
    )
    
    # 模擬個股分析
    stock_analysis = StockAnalysis(
        symbol="2330.TW",
        current_price=500.0,
        technical_indicators={
            "RSI": 45,
            "MACD": 2.5,
            "volatility": 0.25
        },
        fundamental_metrics={
            "PE": 13.5,
            "ROE": 0.18,
            "debt_ratio": 0.3,
            "current_ratio": 2.5
        },
        risk_factors=["產業循環風險", "地緣政治風險"]
    )
    
    # 生成決策
    decision = agent.generate_decision(stock_analysis, market_context)
    
    # 輸出決策報告
    print(agent.generate_report(decision))