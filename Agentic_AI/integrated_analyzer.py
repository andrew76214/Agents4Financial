from typing import List, Dict, Any
from dataclasses import dataclass
import pandas as pd
import json
import re
from langchain_ollama import ChatOllama
from IPython.display import Image, display

from transcript_node import TranscriptAgent, TranscriptProcessor
from market_node import ReActMarketAgent
from decision_node import MarketContext, StockAnalysis, DecisionAgent

from constant import model_name

from opencc import OpenCC
cc = OpenCC('s2twp')  # 簡體轉繁體並使用台灣用詞

@dataclass
class AnalysisResult:
    """Combined analysis result"""
    transcript_summary: str
    market_sentiment: str
    trading_signals: List[str]
    stock_recommendations: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    final_decision: Dict[str, Any]
    confidence_score: float = 0.0  # Default to 0.0 if not provided

class MarketSentimentAnalyzer:
    """市場情緒分析器"""
    
    def __init__(self):
        self.sentiment_keywords = {
            'positive': ['樂觀', '上漲', '反彈', '利多', '看好', '突破', '創新高'],
            'negative': ['悲觀', '下跌', '重挫', '利空', '看淡', '跌破', '創新低'],
            'neutral': ['震盪', '盤整', '觀望', '持平']
        }
        
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """分析文本中的市場情緒
        
        Returns:
            Dict with sentiment scores (positive, negative, neutral)
        """
        word_count = len(text.split())
        sentiment_scores = {
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 0.0
        }
        
        # 計算每種情緒的出現頻率
        for sentiment, keywords in self.sentiment_keywords.items():
            count = sum(text.count(keyword) for keyword in keywords)
            sentiment_scores[sentiment] = count / word_count
            
        # 標準化分數
        total = sum(sentiment_scores.values())
        if total > 0:
            for key in sentiment_scores:
                sentiment_scores[key] /= total
                
        return sentiment_scores
        
    def get_market_phase(self, sentiment_scores: Dict[str, float]) -> str:
        """根據情緒分數判斷市場階段"""
        if sentiment_scores['positive'] > 0.5:
            return '多頭市場'
        elif sentiment_scores['negative'] > 0.5:
            return '空頭市場'
        elif sentiment_scores['neutral'] > 0.4:
            return '盤整市場'
        else:
            return '轉折階段'
            
    def analyze_vix_impact(self, text: str) -> float:
        """分析VIX指數對市場情緒的影響"""
        vix_pattern = r"VIX.*?(\d+)"
        matches = re.findall(vix_pattern, text)
        
        if matches:
            vix_values = [float(v) for v in matches]
            avg_vix = sum(vix_values) / len(vix_values)
            
            # VIX > 30 通常表示高度恐慌
            if avg_vix > 30:
                return -0.8
            # VIX < 20 通常表示市場平穩
            elif avg_vix < 20:
                return 0.2
                
        return 0.0

class IntegratedMarketAnalyzer:
    """Integrated system combining transcript analysis and market decisions"""
    
    def __init__(self, model_name=model_name, max_iterations=24):
        self.llm = ChatOllama(model=model_name)
        self.transcript_agent = TranscriptAgent(model_name)
        self.market_agent = ReActMarketAgent(model_name)
        self.max_iterations = max_iterations
        self.historical_decisions = {}  # Store historical decisions
        self.history_window = 90  # 分析歷史數據的天數窗口
        self.min_confidence = 0.6  # 最小信心水平
        self.trading_signals = []  # 儲存交易信號
        
        # 載入股票池
        try:
            with open('Agentic_AI/stock_pool.json', 'r', encoding='utf-8') as f:
                self.stock_pool = json.load(f)
                print(f"成功載入 {len(self.stock_pool['stocks'])} 支股票")
        except Exception as e:
            print(f"Warning: Failed to load stock pool: {e}")
            self.stock_pool = {"stocks": [], "metadata": {"version": "1.0"}}
        
    def extract_market_context(self, summary: str) -> MarketContext:
        """從摘要中提取市場背景資訊"""
        prompt = f"""
        From the following market summary, please extract key market information:
        
        {summary}
        
        Please respond in JSON format with:
        1. market_sentiment: Overall market sentiment (bullish/bearish/neutral)
        2. key_indices: Any mentioned market indices and their values
        3. macro_indicators: Any mentioned macro economic indicators
        
        You must use the exact JSON format above, with no additional text.
        """
        
        response = self.llm.invoke([{"role": "user", "content": prompt}])
        # Clean the response content to ensure it only contains the JSON part
        content = response.content.strip()
        # Remove any markdown code block markers if present
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        try:
            context_data = json.loads(content)
        except json.JSONDecodeError:
            # Fallback to neutral sentiment if parsing fails
            context_data = {
                "market_sentiment": "neutral",
                "key_indices": {},
                "macro_indicators": {}
            }
        
        sentiment = context_data["market_sentiment"]
        indices = context_data.get("key_indices", {})
        macro = context_data.get("macro_indicators", {})
        volume = context_data.get("trading_volume", {})
        
        # 新增: 分析市場關鍵字
        keywords = self._extract_market_keywords(summary)
        
        # 新增: 處理時間序列資料
        time_series = self._process_time_series(summary)
        
        # 新增: 分析全球市場關聯性
        global_correlations = self._analyze_global_markets(summary)
        
        return MarketContext(
            market_sentiment=sentiment,
            key_indices=indices,
            macro_indicators=macro,
            trading_volume=volume,
            market_keywords=keywords,
            time_series_data=time_series,
            global_correlations=global_correlations
        )

    def _extract_market_keywords(self, text: str) -> List[str]:
        """提取市場關鍵字和主題"""
        keywords = []
        
        # 搜索常見市場主題
        market_themes = [
            "關稅戰", "貿易衝突", "政策不確定性",
            "供應鏈轉移", "通膨", "升息",
            "經濟衰退", "技術性衰退", "外部衝擊"
        ]
        
        for theme in market_themes:
            if theme in text:
                keywords.append(theme)
        
        return keywords

    def _process_time_series(self, text: str) -> Dict[str, List[float]]:
        """處理時間序列數據"""
        series_data = {}
        
        # 解析數字序列
        number_pattern = r"(\d+\.?\d*%?)"
        matches = re.finditer(number_pattern, text)
        
        current_series = []
        for match in matches:
            value = match.group(1)
            if "%" in value:
                value = float(value.replace("%", "")) / 100
            else:
                value = float(value)
            current_series.append(value)
            
        if current_series:
            series_data["raw_series"] = current_series
            
        return series_data

    def _analyze_global_markets(self, text: str) -> Dict[str, float]:
        """分析全球市場關聯性"""
        correlations = {}
        
        # 主要市場指數
        markets = ["美股", "歐股", "日股", "陸股", "臺股"]
        
        # 計算相關性
        for market in markets:
            if market in text:
                # 簡單的相關性評分 (0-1)
                score = len(re.findall(market, text)) / len(text.split())
                correlations[market] = min(score * 100, 1.0)
                
        return correlations
    
    def extract_stock_analysis(self, summary: str) -> List[StockAnalysis]:
        """Extract stock-specific information and add default stock universe for analysis"""
        try:
            # 從 stock_pool.json 讀取股票池
            with open('Agentic_AI/stock_pool.json', 'r', encoding='utf-8') as f:
                stock_pool = json.load(f)
                default_stocks = stock_pool['stocks']
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Failed to load stock pool from file: {e}")
            return []
        
        # 從摘要中識別市場狀況和產業趨勢
        market_trends = self._identify_market_trends(summary)
        
        # 根據市場趨勢篩選合適的股票
        selected_stocks = self._filter_stocks(default_stocks, market_trends)
        
        # 轉換為StockAnalysis對象
        return [
            StockAnalysis(
                symbol=stock["symbol"],
                current_price=0.0,  # 這裡可以接入實時行情數據
                technical_indicators=self._get_technical_indicators(stock["symbol"]),
                fundamental_metrics=self._get_fundamental_metrics(stock["symbol"]),
                risk_factors=self._assess_stock_risks(stock, market_trends)
            )
            for stock in selected_stocks
        ]

    def _identify_market_trends(self, summary: str) -> Dict[str, Any]:
        """識別市場趨勢和主題"""
        prompt = f"""
        從以下市場總結中識別主要趨勢和主題：
        
        {summary}
        
        請分析並以JSON格式回傳以下資訊:
        {{
            "market_direction": "bullish/bearish/neutral",
            "industry_trends": ["產業1", "產業2"...],
            "risk_factors": ["風險1", "風險2"...],
            "themes": ["主題1", "主題2"...]
        }}
        只需要回傳JSON格式，不要加入額外文字。
        """
        
        response = self.llm.invoke([{"role": "user", "content": prompt}])
        content = response.content.strip()
        
        try:
            # Clean up the response content
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            trends = json.loads(content)
            
            # Ensure all required keys exist with default values
            default_trends = {
                "market_direction": "neutral",
                "industry_trends": ["科技", "AI"],
                "risk_factors": ["市場波動"],
                "themes": ["成長"]
            }
            
            # Update defaults with actual values while preserving structure
            for key in default_trends:
                if key not in trends or not trends[key]:
                    trends[key] = default_trends[key]
                    
            return trends
            
        except json.JSONDecodeError:
            print("Warning: Failed to parse market trends, using defaults")
            return {
                "market_direction": "neutral",
                "industry_trends": ["科技", "AI"],
                "risk_factors": ["市場波動"],
                "themes": ["成長"]
            }

    def _filter_stocks(self, stocks: List[Dict], market_trends: Dict) -> List[Dict]:
        """根據市場趨勢篩選股票"""
        filtered_stocks = []
        
        # Get market direction with fallback to neutral
        market_direction = market_trends.get("market_direction", "neutral").lower()
        
        # 根據市場方向調整選股策略
        if market_direction == "bullish":
            # 偏好成長股和科技股
            filtered_stocks = [
                s for s in stocks 
                if s.get("type") == "科技" or s.get("sector") in ["半導體", "電動車", "軟體服務"]
            ]
        elif market_direction == "bearish":
            # 偏好防禦性股票和價值股
            filtered_stocks = [
                s for s in stocks 
                if s.get("type") in ["金融", "傳產"] or s.get("sector") in ["飲料", "零售", "銀行"]
            ]
        else:
            # 震盪市場採取均衡策略
            # 確保不同類型股票都有代表
            tech_stocks = [s for s in stocks if s.get("type") == "科技"][:2]
            finance_stocks = [s for s in stocks if s.get("type") == "金融"][:1]
            consumer_stocks = [s for s in stocks if s.get("type") == "消費"][:1]
            filtered_stocks = tech_stocks + finance_stocks + consumer_stocks
        
        # 按市場分組，確保台股和美股都有合適配置
        tw_stocks = [s for s in filtered_stocks if s["market"] == "台股"]
        us_stocks = [s for s in filtered_stocks if s["market"] == "美股"]
        
        # 如果某個市場的股票太少，從原始清單中補充
        if len(tw_stocks) < 2:
            tw_candidates = [s for s in stocks if s["market"] == "台股" and s not in tw_stocks]
            tw_stocks.extend(tw_candidates[:2-len(tw_stocks)])
        
        if len(us_stocks) < 2:
            us_candidates = [s for s in stocks if s["market"] == "美股" and s not in us_stocks]
            us_stocks.extend(us_candidates[:2-len(us_stocks)])
        
        filtered_stocks = tw_stocks + us_stocks
        
        # 如果篩選後股票太少，從原始清單中補充
        if len(filtered_stocks) < 4:  # 確保至少有4支股票
            remaining = [s for s in stocks if s not in filtered_stocks]
            filtered_stocks.extend(remaining[:4-len(filtered_stocks)])
        
        return filtered_stocks

    def _get_technical_indicators(self, symbol: str) -> Dict[str, float]:
        """取得技術指標"""
        # TODO: 接入實時技術指標數據
        return {
            "RSI": 50.0,
            "MACD": 0.0,
            "MA20": 0.0,
            "MA60": 0.0
        }

    def _get_fundamental_metrics(self, symbol: str) -> Dict[str, Any]:
        """取得基本面指標"""
        # TODO: 接入實時基本面數據
        return {
            "PE": 15.0,
            "PB": 2.0,
            "ROE": 0.15,
            "Revenue_Growth": 0.10
        }

    def _assess_stock_risks(self, stock: Dict, market_trends: Dict[str, Any]) -> List[str]:
        """評估股票風險"""
        risks = []
        
        # 根據股票類型評估風險
        if stock["type"] == "科技":
            risks.append("科技股波動風險")
        
        # 根據市場添加區域風險
        if stock["market"] == "美股":
            risks.append("匯率風險")
        
        # 添加市場趨勢相關風險
        risks.extend(market_trends.get("risk_factors", []))
        
        return risks
    
    def analyze_transcript(self, transcript: str) -> AnalysisResult:
        """執行完整的分析流程,從講稿到投資決策"""
        iteration_count = 0
        
        print("\n=== 開始分析 ===")
        
        # Step 1: 處理講稿並生成摘要
        print("\n1. 處理文字稿並生成摘要...")
        transcript_result = self.transcript_agent.process_transcript(transcript)
        summary = transcript_result["summary"]
        print(f"摘要完成: {summary}")
        
        # Step 2: 提取市場情緒和股票資訊
        print("\n2. 提取市場背景資訊...")
        market_context = self.extract_market_context(summary)
        # 標準化market_sentiment的值
        market_context.market_sentiment = market_context.market_sentiment.lower()
        if 'bull' in market_context.market_sentiment:
            market_context.market_sentiment = 'bullish'
        elif 'bear' in market_context.market_sentiment:
            market_context.market_sentiment = 'bearish'
        else:
            market_context.market_sentiment = 'neutral'
        print(f"市場情緒: {market_context.market_sentiment}")
        
        print("\n3. 分析個股資訊...")
        stock_analyses = self.extract_stock_analysis(summary)
        print(f"找到 {len(stock_analyses)} 支股票進行分析")
        
        # 計算市場信心指標
        confidence_score = self._calculate_confidence_score(market_context, stock_analyses)
        
        # 生成交易信號
        trading_signals = self._generate_trading_signals(market_context, stock_analyses, confidence_score)
        if not trading_signals:
            trading_signals = [f"市場整體情緒{market_context.market_sentiment}"]
        print(f"\n生成了 {len(trading_signals)} 個交易信號")
        
        # 風險評估
        risk_assessment = self._assess_risk(
            market_context,
            stock_analyses,
            confidence_score
        )
        
        # Step 3: Generate market decisions using the market agent
        print("\n4. 生成市場決策...")
        market_analysis = self.market_agent.analyze_market(summary)
        
        # Integrate market analysis into trading signals and risk assessment
        if market_analysis:
            if 'signals' in market_analysis:
                trading_signals.extend(market_analysis['signals'])
            if 'risks' in market_analysis:
                risk_assessment['market_risks'].extend(market_analysis['risks'])
            if 'sentiment' in market_analysis:
                # Update market context with additional sentiment data
                market_context.market_sentiment = market_analysis['sentiment']
        print(market_analysis)

        # Create decision agent for detailed stock analysis
        print("\n5. 進行詳細股票分析...")
        decision_agent = DecisionAgent()
        decisions = []
        for i, stock in enumerate(stock_analyses, 1):
            print(f"分析股票 {i}/{len(stock_analyses)}: {stock.symbol}")
            # Check iteration limit
            iteration_count += 1
            if iteration_count >= self.max_iterations:
                print(f"警告: 達到最大迭代次數 ({self.max_iterations}). 進入下一階段.")
                break
                
            decision = decision_agent.generate_decision(
                stock_analysis=stock,
                market_context=market_context
            )
            decisions.append(decision)
            
        # Step 4: Compile final analysis result
        print("\n6. 整合分析結果...生成最終決策...")
        final_result = AnalysisResult(
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
            final_decision=self._make_final_decision(decisions, market_context),
            confidence_score=confidence_score
        )
        
        print("\n=== 分析完成 ===\n")
        return final_result
    
    def _calculate_confidence_score(self, market_context: MarketContext,
                                stock_analyses: List[StockAnalysis]) -> float:
        """計算市場信心指標"""
        # 根據市場情緒賦予基礎分數
        base_score = {
            "bullish": 0.8,
            "neutral": 0.5,
            "bearish": 0.2
        }.get(market_context.market_sentiment.lower(), 0.5)
        
        # 考慮技術指標
        if market_context.key_indices:
            if isinstance(market_context.key_indices, dict):
                # Handle dictionary type
                technical_score = sum(
                    1 for v in market_context.key_indices.values() 
                    if isinstance(v, (int, float)) and v > 0
                ) / len(market_context.key_indices)
            else:
                # Handle list type
                technical_score = sum(
                    1 for v in market_context.key_indices
                    if isinstance(v, (int, float)) and v > 0
                ) / len(market_context.key_indices)
        else:
            technical_score = 0.5
            
        # 考慮宏觀指標
        macro_score = len(market_context.macro_indicators) > 0
        
        # 綜合評分
        confidence = (base_score * 0.5 + 
                     technical_score * 0.3 +
                     macro_score * 0.2)
                     
        return min(max(confidence, 0), 1)  # 確保在0-1之間

    def _generate_trading_signals(self, market_context: MarketContext,
                              stock_analyses: List[StockAnalysis],
                              confidence: float) -> List[str]:
        """生成交易信號"""
        signals = []
        
        # 基於市場情緒的信號
        if confidence > self.min_confidence:
            if market_context.market_sentiment.lower() == "bullish":
                signals.append("市場氣氛看漲,可考慮逢低買進")
            elif market_context.market_sentiment.lower() == "bearish":
                signals.append("市場氣氛看空,建議減持部位")
        
        # 基於個股分析的信號
        for stock in stock_analyses:
            if stock.current_price:
                # 技術面信號
                rsi = stock.technical_indicators.get("RSI", 50)
                if rsi < 30:
                    signals.append(f"{stock.symbol}: RSI顯示超賣({rsi:.1f}), 可能存在反彈機會")
                elif rsi > 70:
                    signals.append(f"{stock.symbol}: RSI顯示超買({rsi:.1f}), 注意回檔風險")
                
                # 基本面信號
                pe = stock.fundamental_metrics.get("PE")
                if pe and pe < 15:
                    signals.append(f"{stock.symbol}: 本益比({pe:.1f})低於產業平均, 具有價值投資機會")
                
                # 風險提示
                if stock.risk_factors:
                    risk_str = "、".join(stock.risk_factors[:2])  # 只顯示前兩個風險因素
                    signals.append(f"{stock.symbol}: 需注意風險 - {risk_str}")
        
        return signals

    def _assess_risk(self, market_context: MarketContext,
                   stock_analyses: List[StockAnalysis],
                   confidence: float) -> Dict[str, Any]:
        """評估整體風險"""
        risk_factors = []
        risk_level = "低"
        
        # 評估市場風險
        if market_context.market_sentiment.lower() == "bearish":
            risk_factors.append("市場情緒偏空")
            risk_level = "高"
        
        # 評估總體經濟風險
        if market_context.macro_indicators:
            if isinstance(market_context.macro_indicators, dict):
                # Handle dictionary type
                for indicator, value in market_context.macro_indicators.items():
                    if isinstance(value, str) and "risk" in value.lower():
                        risk_factors.append(f"{indicator}顯示風險提升")
                        risk_level = "中高" if risk_level != "高" else "高"
            else:
                # Handle list type
                for indicator in market_context.macro_indicators:
                    if isinstance(indicator, str) and "risk" in indicator.lower():
                        risk_factors.append(f"{indicator}顯示風險提升")
                        risk_level = "中高" if risk_level != "高" else "高"
        
        # 評估個股風險
        stock_risks = []
        for stock in stock_analyses:
            if hasattr(stock, "risk_factors") and stock.risk_factors:
                stock_risks.extend(stock.risk_factors)
        
        return {
            "risk_level": risk_level,
            "market_risks": risk_factors,
            "stock_specific_risks": stock_risks,
            "confidence_score": confidence
        }
    
    def _extract_trading_signals(self, summary: str) -> List[str]:
        """Extract trading signals from summary"""
        prompt = f"""
        From the following market summary, list all trading signals mentioned:
        
        {summary}
        
        Include technical signals, fundamental signals, and market sentiment signals.
        Return the signals in JSON format as an array of strings, with no additional text.
        """
        
        response = self.llm.invoke([{"role": "user", "content": prompt}])
        # Clean the response content to ensure it only contains the JSON part
        content = response.content.strip()
        # Remove any markdown code block markers if present
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Fallback to empty list if parsing fails
            return []
    
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
        if not decisions:
            # 即使沒有具體決策，也基於市場情緒給出建議
            market_advice = {
                "bullish": "可考慮逢低布局ETF",
                "bearish": "建議降低持股比例",
                "neutral": "維持現有部位，觀察市場動向"
            }
            return {
                "overall_stance": market_context.market_sentiment,
                "recommended_actions": [{
                    "symbol": "整體市場",
                    "action": market_advice.get(market_context.market_sentiment, "觀望"),
                    "confidence": "Medium"
                }],
                "market_context": {
                    "sentiment": market_context.market_sentiment,
                    "key_indicators": market_context.key_indices,
                    "reasoning": ["基於當前市場情緒給出建議"]
                }
            }

        # 分析買賣決策
        buy_decisions = [d for d in decisions if hasattr(d, 'decision_type') and d.decision_type.value == "買入"]
        sell_decisions = [d for d in decisions if hasattr(d, 'decision_type') and d.decision_type.value == "賣出"]
        hold_decisions = [d for d in decisions if hasattr(d, 'decision_type') and d.decision_type.value == "觀望"]
        
        # 根據決策比例判斷整體立場
        total_decisions = len(decisions)
        buy_ratio = len(buy_decisions) / total_decisions if total_decisions > 0 else 0
        sell_ratio = len(sell_decisions) / total_decisions if total_decisions > 0 else 0
        
        # 綜合考慮市場情緒和具體決策
        if market_context.market_sentiment == "bullish" and buy_ratio > 0.3:
            stance = "Bullish"
            confidence = "High"
        elif market_context.market_sentiment == "bearish" and sell_ratio > 0.3:
            stance = "Bearish"
            confidence = "High"
        else:
            stance = "Neutral"
            confidence = "Medium"
        
        # 生成具體建議
        recommendations = []
        for d in decisions:
            if not hasattr(d, 'decision_type'):
                continue
                
            action_confidence = "High" if hasattr(d, 'strategy') and d.strategy.position_size > 0.15 else "Medium"
            
            # 添加更詳細的建議
            recommendation = {
                "symbol": d.target_symbol,
                "action": d.decision_type.value,
                "confidence": action_confidence,
                "position_size": d.strategy.position_size if hasattr(d, 'strategy') else 0.1,
                "reasoning": d.reasoning if hasattr(d, 'reasoning') else []
            }
            
            # 如果有止損止盈建議，添加進去
            if hasattr(d, 'strategy'):
                if d.strategy.stop_loss > 0:
                    recommendation["stop_loss"] = d.strategy.stop_loss
                if d.strategy.take_profit > 0:
                    recommendation["take_profit"] = d.strategy.take_profit
                    
            recommendations.append(recommendation)
        
        return {
            "overall_stance": stance,
            "confidence": confidence,
            "recommended_actions": recommendations,
            "market_context": {
                "sentiment": market_context.market_sentiment,
                "key_indicators": market_context.key_indices
            },
            "summary": {
                "buy_signals": len(buy_decisions),
                "sell_signals": len(sell_decisions),
                "hold_signals": len(hold_decisions)
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
    
    def print_workflow(self):
        """Print the market analysis workflow diagram"""
        print("\nMarket Analysis Workflow:")
        try:
            # Try to get and display the workflow graph from market agent
            display(Image(self.market_agent.chain.get_graph().draw_mermaid_png()))
        except Exception as e:
            print(f"Error displaying market analysis graph: {e}")
    
    def analyze_with_history(self, target_date: str, csv_path: str = './transcripts_video_v1.1.csv') -> AnalysisResult:
        """
        Analyze market data up to a target date using historical context
        
        Args:
            target_date: Target date in format YYYY/MM/DD
            csv_path: Path to CSV file containing transcripts
        """
        # Read and preprocess the data
        df = pd.read_csv(csv_path)
        
        # Extract dates from video names and convert to datetime
        df['date'] = df['video_name'].str.extract(r'(\d{4}/\d{1,2}/\d{1,2})')
        df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
        
        # Filter data up to target date
        target_date = pd.to_datetime(target_date, format='%Y/%m/%d')
        historical_data = df[df['date'] <= target_date].sort_values('date')
        
        if historical_data.empty:
            raise ValueError(f"No data found before {target_date}")
            
        print(f"\n=== 分析截至 {target_date.strftime('%Y/%m/%d')} 的市場數據 ===")
        print(f"找到 {len(historical_data)} 條歷史記錄")
        
        # Combine all historical transcripts with weights
        combined_analysis = self._analyze_historical_data(historical_data)
        
        return combined_analysis
        
    def _analyze_historical_data(self, historical_data: pd.DataFrame) -> AnalysisResult:
        """Analyze historical data with time-based weights"""
        
        all_results = []
        dates = historical_data['date'].tolist()
        
        # Calculate time-based weights (more recent data gets higher weight)
        max_date = max(dates)
        weights = [(pd.Timedelta(max_date - date).days + 1) ** -0.5 for date in dates]
        # Normalize weights
        weights = [w/sum(weights) for w in weights]
        
        print("\n正在分析歷史數據...")
        
        for i, (_, row) in enumerate(historical_data.iterrows()):
            print(f"\n分析 {row['date'].strftime('%Y/%m/%d')} 的數據 ({i+1}/{len(historical_data)})")
            
            # Analyze individual transcript
            result = self.analyze_transcript(row['transcript'])
            
            # Store result with its weight
            all_results.append({
                'date': row['date'],
                'weight': weights[i],
                'result': result
            })
            
            # Store in historical decisions
            self.historical_decisions[row['date']] = result
            
        # Combine all weighted results
        return self._combine_weighted_results(all_results)
        
    def _combine_weighted_results(self, weighted_results: List[Dict]) -> AnalysisResult:
        """Combine multiple weighted analysis results into a single result"""
        
        # Initialize combined values
        combined_summary = []
        combined_signals = []
        combined_recommendations = []
        all_risks = []
        
        # Weight and combine results
        for wr in weighted_results:
            weight = wr['weight']
            result = wr['result']
            
            # Combine summaries with dates
            combined_summary.append(f"[{wr['date'].strftime('%Y/%m/%d')}]: {result.transcript_summary}")
            
            # Combine trading signals
            combined_signals.extend(result.trading_signals)
            
            # Weight and combine stock recommendations
            for rec in result.stock_recommendations:
                rec['weight'] = weight
                combined_recommendations.append(rec)
            
            # Collect all risks
            all_risks.extend(result.risk_assessment.get('market_risks', []))
            all_risks.extend(result.risk_assessment.get('stock_specific_risks', []))
            
        # Create final combined result
        return AnalysisResult(
            transcript_summary="\n\n".join(combined_summary),
            market_sentiment=self._get_weighted_sentiment(weighted_results),
            trading_signals=list(set(combined_signals)),  # Remove duplicates
            stock_recommendations=self._combine_stock_recommendations(combined_recommendations),
            risk_assessment={
                'market_risks': list(set(all_risks)),
                'risk_level': self._calculate_overall_risk_level(weighted_results)
            },
            final_decision=self._make_final_historical_decision(weighted_results)
        )
        
    def _get_weighted_sentiment(self, weighted_results: List[Dict]) -> str:
        """Calculate weighted market sentiment"""
        sentiment_scores = {
            'bullish': 1,
            'neutral': 0,
            'bearish': -1
        }
        
        # 標準化sentiment值
        def normalize_sentiment(sentiment: str) -> str:
            sentiment = str(sentiment).lower()
            if 'bull' in sentiment:
                return 'bullish'
            elif 'bear' in sentiment:
                return 'bearish'
            return 'neutral'
        
        weighted_score = sum(
            sentiment_scores[normalize_sentiment(r['result'].market_sentiment)] * r['weight']
            for r in weighted_results
        )
        
        if weighted_score > 0.2:
            return 'bullish'
        elif weighted_score < -0.2:
            return 'bearish'
        return 'neutral'
        
    def _combine_stock_recommendations(self, weighted_recommendations: List[Dict]) -> List[Dict]:
        """Combine weighted stock recommendations"""
        
        # Group recommendations by symbol
        symbol_groups = {}
        for rec in weighted_recommendations:
            symbol = rec['symbol']
            if symbol not in symbol_groups:
                symbol_groups[symbol] = []
            symbol_groups[symbol].append(rec)
            
        # Combine recommendations for each symbol
        final_recommendations = []
        for symbol, recs in symbol_groups.items():
            total_weight = sum(r['weight'] for r in recs)
            weighted_action = max(
                set(r['action'] for r in recs),
                key=lambda x: sum(r['weight'] for r in recs if r['action'] == x)
            )
            
            # Combine strategies
            avg_position = sum(r['strategy']['position_size'] * r['weight'] for r in recs) / total_weight
            
            final_recommendations.append({
                'symbol': symbol,
                'action': weighted_action,
                'risk_level': max(r['risk_level'] for r in recs),
                'strategy': {
                    'position_size': avg_position,
                    'stop_loss': min(r['strategy']['stop_loss'] for r in recs if r['strategy']['stop_loss'] > 0),
                    'take_profit': max(r['strategy']['take_profit'] for r in recs),
                    'time_horizon': 'long-term'  # Using historical data implies longer-term view
                }
            })
            
        return final_recommendations
        
    def _calculate_overall_risk_level(self, weighted_results: List[Dict]) -> str:
        """Calculate overall risk level from weighted results"""
        risk_scores = {
            'HIGH': 2,
            'MEDIUM': 1,
            'LOW': 0
        }
        
        weighted_score = sum(
            risk_scores[r['result'].risk_assessment['risk_level']] * r['weight']
            for r in weighted_results
        )
        
        if weighted_score > 1.5:
            return 'HIGH'
        elif weighted_score > 0.5:
            return 'MEDIUM'
        return 'LOW'
        
    def _make_final_historical_decision(self, weighted_results: List[Dict]) -> Dict[str, Any]:
        """Generate final decision considering historical context"""
        
        recent_results = sorted(weighted_results, key=lambda x: x['date'])[-3:]  # Look at most recent 3 results
        recent_weight = 0.7  # Give 70% weight to recent results
        
        recent_stance = self._get_weighted_sentiment(recent_results)
        historical_stance = self._get_weighted_sentiment(weighted_results)
        
        final_stance = recent_stance if recent_stance != 'neutral' else historical_stance
        
        return {
            'overall_stance': final_stance,
            'recent_trend': recent_stance,
            'historical_context': historical_stance,
            'recommended_actions': self._combine_stock_recommendations(
                [rec for wr in weighted_results for rec in wr['result'].stock_recommendations]
            ),
            'confidence': 'High' if recent_stance == historical_stance else 'Medium'
        }


if __name__ == "__main__":
    # Initialize the integrated analyzer and opencc converter
    analyzer = IntegratedMarketAnalyzer()
        
    # Read all transcripts
    df = pd.read_csv('./transcripts_video_v1.1.csv')
    
    # Extract dates from video names and convert to datetime
    df['date'] = df['video_name'].str.extract(r'(\d{4}⧸\d{1,2}⧸\d{1,2})')
    df['date'] = pd.to_datetime(df['date'], format='%Y⧸%m⧸%d')
    
    # Filter data starting from 2023/1/30
    start_date = pd.to_datetime('2023⧸1⧸30', format='%Y⧸%m⧸%d')
    df = df[df['date'] >= start_date].sort_values('date')
    
    # Convert transcript to traditional Chinese
    df['transcript'] = df['transcript'].apply(cc.convert)
    
    # Initialize results list
    all_results = []
    
    # Process each transcript
    for idx, row in df.iterrows():
        print(f"\n分析 {row['date'].strftime('%Y⧸%m⧸%d')} 的資料 ({idx+1}/{len(df)})")
        
        # Analyze transcript
        result = analyzer.analyze_transcript(row['transcript'])
        
        # Create result dictionary
        result_dict = {
            'date': row['date'].strftime('%Y⧸%m⧸%d'),
            'market_sentiment': cc.convert(str(result.market_sentiment)),
            'transcript_summary': cc.convert(result.transcript_summary),
            'trading_signals': [cc.convert(signal) for signal in result.trading_signals],
            'risk_level': cc.convert(result.risk_assessment['risk_level']),
            'market_risks': [cc.convert(risk) for risk in result.risk_assessment.get('market_risks', [])],
            'stock_specific_risks': [cc.convert(risk) for risk in result.risk_assessment.get('stock_specific_risks', [])],
            'final_stance': cc.convert(str(result.final_decision['overall_stance'])),
            'stock_recommendations': [
                {
                    'symbol': rec['symbol'],
                    'action': cc.convert(rec['action']),
                    'risk_level': cc.convert(rec['risk_level']),
                    'strategy': rec['strategy']
                }
                for rec in result.stock_recommendations
            ]
        }
        all_results.append(result_dict)
        
        # Save intermediate results to JSON
        with open('analysis_results.json', 'w', encoding='utf-8-sig') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"已儲存結果到 analysis_results.json")
        
    print("\n分析完成！結果已儲存至 analysis_results.json")