from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import pandas as pd
from langchain_ollama import ChatOllama

from transcript_node import TranscriptAgent, TranscriptProcessor
from market_node import ReActMarketAgent, MarketContext, StockAnalysis

@dataclass
class AnalysisResult:
    """Combined analysis result"""
    transcript_summary: str
    market_sentiment: str
    trading_signals: List[str]
    stock_recommendations: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    final_decision: Dict[str, Any]

class IntegratedMarketAnalyzer:
    """Integrated system combining transcript analysis and market decisions"""
    
    def __init__(self, model_name: str = "llama2:13b"):
        self.llm = ChatOllama(model=model_name)
        self.transcript_agent = TranscriptAgent(model_name)
        self.market_agent = ReActMarketAgent(model_name)
        
    def extract_market_context(self, summary: str) -> MarketContext:
        """Extract market context from transcript summary"""
        prompt = f"""
        From the following market summary, please extract key market information:
        
        {summary}
        
        Please respond in JSON format with:
        1. market_sentiment: Overall market sentiment (bullish/bearish/neutral)
        2. key_indices: Any mentioned market indices and their values
        3. macro_indicators: Any mentioned macro economic indicators
        """
        
        response = self.llm.invoke([{"role": "user", "content": prompt}])
        context_data = eval(response.content)
        
        return MarketContext(
            market_sentiment=context_data["market_sentiment"],
            key_indices=context_data.get("key_indices", {}),
            macro_indicators=context_data.get("macro_indicators", {})
        )
    
    def extract_stock_analysis(self, summary: str) -> List[StockAnalysis]:
        """Extract stock-specific information from transcript summary"""
        prompt = f"""
        From the following market summary, please extract information about specific stocks:
        
        {summary}
        
        For each mentioned stock, provide:
        1. symbol: Stock symbol
        2. mentioned_price: Any mentioned price
        3. technical_indicators: Any mentioned technical indicators
        4. fundamental_metrics: Any mentioned fundamental metrics
        5. risk_factors: Any mentioned risks
        
        Respond in JSON format as a list of stocks.
        """
        
        response = self.llm.invoke([{"role": "user", "content": prompt}])
        stocks_data = eval(response.content)
        
        return [
            StockAnalysis(
                symbol=stock["symbol"],
                current_price=stock.get("mentioned_price", 0.0),
                technical_indicators=stock.get("technical_indicators", {}),
                fundamental_metrics=stock.get("fundamental_metrics", {}),
                risk_factors=stock.get("risk_factors", [])
            )
            for stock in stocks_data
        ]
    
    def analyze_transcript(self, transcript: str) -> AnalysisResult:
        """Perform complete analysis from transcript to investment decision"""
        
        # Step 1: Process transcript and get summary
        transcript_result = self.transcript_agent.process_transcript(transcript)
        summary = transcript_result["summary"]
        
        # Step 2: Extract market context and stock information
        market_context = self.extract_market_context(summary)
        stock_analyses = self.extract_stock_analysis(summary)
        
        # Step 3: Generate market decisions for each mentioned stock
        decisions = []
        for stock in stock_analyses:
            decision = self.market_agent.generate_decision(
                stock_analysis=stock,
                market_context=market_context
            )
            decisions.append(decision)
            
        # Step 4: Compile final analysis result
        trading_signals = self._extract_trading_signals(summary)
        risk_assessment = self._assess_overall_risk(market_context, stock_analyses)
        
        return AnalysisResult(
            transcript_summary=summary,
            market_sentiment=market_context.market_sentiment,
            trading_signals=trading_signals,
            stock_recommendations=[
                {
                    "symbol": d.target_symbol,
                    "action": d.decision_type.value,
                    "risk_level": d.risk_level.value,
                    "strategy": {
                        "position_size": d.strategy.position_size,
                        "stop_loss": d.strategy.stop_loss,
                        "take_profit": d.strategy.take_profit,
                        "time_horizon": d.strategy.time_horizon
                    },
                    "reasoning": d.reasoning
                }
                for d in decisions
            ],
            risk_assessment=risk_assessment,
            final_decision=self._make_final_decision(decisions, market_context)
        )
    
    def _extract_trading_signals(self, summary: str) -> List[str]:
        """Extract trading signals from summary"""
        prompt = f"""
        From the following market summary, list all trading signals mentioned:
        
        {summary}
        
        Include technical signals, fundamental signals, and market sentiment signals.
        Respond with a Python list of strings.
        """
        
        response = self.llm.invoke([{"role": "user", "content": prompt}])
        return eval(response.content)
    
    def _assess_overall_risk(self, 
                           market_context: MarketContext,
                           stock_analyses: List[StockAnalysis]) -> Dict[str, Any]:
        """Assess overall market and portfolio risk"""
        # Compile risk factors
        market_risks = []
        if market_context.market_sentiment == "bearish":
            market_risks.append("Negative market sentiment")
            
        stock_risks = []
        for stock in stock_analyses:
            stock_risks.extend(stock.risk_factors)
            
        return {
            "market_risks": market_risks,
            "stock_specific_risks": stock_risks,
            "risk_level": "HIGH" if len(market_risks) + len(stock_risks) > 3 else "MEDIUM"
        }
    
    def _make_final_decision(self,
                           decisions: List[Any],
                           market_context: MarketContext) -> Dict[str, Any]:
        """Generate final comprehensive decision"""
        buy_decisions = [d for d in decisions if d.decision_type.value == "買入"]
        sell_decisions = [d for d in decisions if d.decision_type.value == "賣出"]
        
        return {
            "overall_stance": "Bullish" if len(buy_decisions) > len(sell_decisions) else "Bearish",
            "recommended_actions": [
                {
                    "symbol": d.target_symbol,
                    "action": d.decision_type.value,
                    "confidence": "High" if d.strategy.position_size > 0.15 else "Medium"
                }
                for d in decisions if d.decision_type.value != "觀望"
            ],
            "market_context": {
                "sentiment": market_context.market_sentiment,
                "key_indicators": market_context.key_indices
            }
        }
    
    def generate_report(self, result: AnalysisResult) -> str:
        """Generate a comprehensive analysis report"""
        report = []
        report.append("=== Market Analysis Report ===")
        report.append(f"\nTranscript Summary:")
        report.append(result.transcript_summary)
        
        report.append(f"\nMarket Sentiment: {result.market_sentiment}")
        
        report.append("\nTrading Signals:")
        for signal in result.trading_signals:
            report.append(f"• {signal}")
            
        report.append("\nStock Recommendations:")
        for rec in result.stock_recommendations:
            report.append(f"\n{rec['symbol']}:")
            report.append(f"  Action: {rec['action']}")
            report.append(f"  Risk Level: {rec['risk_level']}")
            report.append(f"  Position Size: {rec['strategy']['position_size']*100:.1f}%")
            if rec['strategy']['stop_loss'] > 0:
                report.append(f"  Stop Loss: {rec['strategy']['stop_loss']:.2f}")
                report.append(f"  Take Profit: {rec['strategy']['take_profit']:.2f}")
            report.append("  Reasoning:")
            for reason in rec['reasoning']:
                report.append(f"    • {reason}")
                
        report.append("\nRisk Assessment:")
        report.append(f"Overall Risk Level: {result.risk_assessment['risk_level']}")
        if result.risk_assessment['market_risks']:
            report.append("Market Risks:")
            for risk in result.risk_assessment['market_risks']:
                report.append(f"• {risk}")
        if result.risk_assessment['stock_specific_risks']:
            report.append("Stock-Specific Risks:")
            for risk in result.risk_assessment['stock_specific_risks']:
                report.append(f"• {risk}")
                
        report.append("\nFinal Decision:")
        report.append(f"Overall Stance: {result.final_decision['overall_stance']}")
        report.append("Recommended Actions:")
        for action in result.final_decision['recommended_actions']:
            report.append(f"• {action['symbol']}: {action['action']} (Confidence: {action['confidence']})")
        
        return "\n".join(report)

# Example usage
if __name__ == "__main__":
    # Initialize the integrated analyzer
    analyzer = IntegratedMarketAnalyzer()
    
    # Read sample transcript
    df = pd.read_csv('../transcripts_video_v1.1.csv')
    sample_transcript = df['transcript'].iloc[0]
    
    # Perform analysis
    result = analyzer.analyze_transcript(sample_transcript)
    
    # Generate and print report
    report = analyzer.generate_report(result)
    print(report)