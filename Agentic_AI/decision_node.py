from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime
import pandas as pd
import numpy as np
from langchain_ollama import ChatOllama

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

class RiskManager:
    """風險管理模組"""
    def __init__(self):
        self.max_position_size = 0.2  # 單一標的最大持倉比例
        self.max_sector_exposure = 0.4  # 單一產業最大持倉比例
        self.min_liquidity = 1000000  # 最小日均成交量
        
    def assess_risk_level(self, analysis: StockAnalysis, market_context: MarketContext) -> RiskLevel:
        risk_score = 0
        
        # 技術指標風險評估
        if analysis.technical_indicators.get("RSI", 50) > 70:
            risk_score += 1
        if analysis.technical_indicators.get("volatility", 0) > 0.3:
            risk_score += 1
            
        # 基本面風險評估
        if analysis.fundamental_metrics.get("debt_ratio", 0) > 0.7:
            risk_score += 1
        if analysis.fundamental_metrics.get("current_ratio", 1) < 1:
            risk_score += 1
            
        # 市場環境風險評估
        if market_context.market_sentiment == "bearish":
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
        if analysis.technical_indicators.get("RSI", 50) < 30:
            technical_score += 1
        if analysis.technical_indicators.get("MACD", 0) > 0:
            technical_score += 1
            
        # 整合市場情緒
        market_score = 0
        if market_context.market_sentiment == "bullish":
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