# Agents4Financial

A comprehensive financial analysis system that leverages AI agents to process daily financial video transcripts and generate investment insights.

## Project Overview

This project implements an advanced AI-driven financial analysis system that:
- Processes YouTube financial video transcripts
- Analyzes market sentiment and trends
- Generates actionable investment decisions with risk management
- Integrates multiple data sources for comprehensive market analysis

## System Architecture

### Workflow Diagram

```mermaid
graph TB
    subgraph TranscriptAgent
        A[Raw Transcript] --> B[Preprocess]
        B --> C[Split Text]
        C --> D[Summarize]
        D --> E[Summary Output]
    end

    subgraph ReActMarketAgent
        E --> F[Analyze Market]
        F --> G[Think]
        G --> H{Need More Info?}
        H -->|Yes| I[Fetch Data]
        I --> G
        H -->|No| J[Make Decision]
    end

    subgraph DecisionAgent
        J --> K[Initial Analysis]
        K --> L[Think]
        L --> M{Need Details?}
        M -->|Yes| N[Analyze Aspects]
        N --> L
        M -->|No| O[Risk Assessment]
        O --> P[Generate Strategy]
        P --> Q[Final Decision]
    end

    subgraph Data Sources
        R[(Market Data)] --> I
        R --> N
        S[(Technical Indicators)] --> I
        S --> N
        T[(Fundamentals)] --> I
        T --> N
    end

    subgraph Risk Management
        U[Risk Level] --> O
        V[Position Limits] --> P
        W[Stop Loss] --> P
    end

    classDef primary fill:#e1f5fe,stroke:#01579b
    classDef secondary fill:#f3e5f5,stroke:#4a148c
    classDef data fill:#efebe9,stroke:#3e2723
    classDef decision fill:#ffebee,stroke:#b71c1c

    class A,B,C,D,E primary
    class F,G,H,I,J secondary
    class K,L,M,N,O,P,Q decision
    class R,S,T data
    class U,V,W data
```

### 1. Transcript Processing (transcript_node.py)
- Processes raw transcripts from financial videos
- Performs intelligent text segmentation and summarization
- Extracts key market insights and sentiment indicators
- Utilizes LangChain and Ollama for natural language processing

### 2. Market Analysis (market_node.py)
- Implements ReAct (Reasoning + Action) architecture for market analysis
- Processes market data through a multi-stage pipeline:
  - Analysis → Thinking → Decision
- Integrates with external data sources (yfinance, technical indicators)
- Provides dynamic market sentiment assessment

### 3. Decision Making (decision_node.py)
- Implements ReAct (Reasoning + Action) architecture for decision making
- Features advanced reflection and self-improvement capabilities:
  - Analyze → Think → Decide workflow
  - Multi-stage thinking process with data gathering
  - Detailed reasoning and observation tracking
- Provides two decision generation modes:
  - Standard mode: Direct analysis and decision
  - Reflection mode: Deep thinking with multiple analysis cycles
- Implements comprehensive risk management
- Produces detailed investment reports with:
  - Technical analysis
  - Fundamental metrics
  - Risk assessments
  - Position sizing recommendations
  - Full reasoning trace

### 4. Integrated Analysis (integrated_analyzer.py)
- Combines all components into a unified analysis pipeline
- Provides historical data analysis capabilities
- Generates comprehensive market reports
- Implements sentiment tracking and trend analysis

## Key Features

- 🤖 AI-Powered Analysis: Utilizes advanced LLM models for market analysis
- 📈 Technical Analysis: Integrates multiple technical indicators
- 📊 Fundamental Analysis: Processes company fundamentals and macro indicators
- 🎯 Risk Management: Built-in risk assessment and position sizing
- 📝 Detailed Reporting: Generates comprehensive investment reports
- 🔄 Historical Analysis: Supports historical data processing and backtesting

## Prerequisites

- Python 3.10+
- Required packages (install via pip):
  - langchain
  - langchain-ollama
  - opencc
  - pandas
  - yfinance
  - ta-lib
  - numpy

## Setup

1. Clone the repository:
\`\`\`bash
git clone https://github.com/yourusername/Agents4Financial.git
cd Agents4Financial
\`\`\`

2. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

3. Configure the Ollama model in constant.py:
\`\`\`python
model_name = "gemma3:27b"  # or your preferred model
\`\`\`

## Usage

### Basic Usage
\`\`\`python
from Agentic_AI.integrated_analyzer import IntegratedMarketAnalyzer
from Agentic_AI.decision_node import DecisionAgent

# Initialize analyzer and decision agent
analyzer = IntegratedMarketAnalyzer()
decision_agent = DecisionAgent()

# Standard decision making
decision = decision_agent.generate_decision(stock_analysis, market_context)

# Advanced decision making with reflection
reflective_decision = decision_agent.generate_decision_with_reflection(stock_analysis, market_context)

# Generate detailed report
report = decision_agent.generate_report(reflective_decision)
print(report)
\`\`\`

### Historical Analysis
\`\`\`python
# Analyze historical data up to a specific date
historical_result = analyzer.analyze_with_history("2024/04/14")
\`\`\`

## Data Structure

The system uses a modular architecture with several key components:

- **TranscriptAgent**: Processes and summarizes financial transcripts
- **ReActMarketAgent**: Analyzes market conditions using ReAct architecture
- **DecisionAgent**: Generates investment decisions with risk management
- **IntegratedMarketAnalyzer**: Combines all components for comprehensive analysis

## Output Format

The system generates structured analysis results including:

- Market sentiment analysis
- Trading signals and recommendations
- Risk assessments
- Position sizing recommendations
- Technical and fundamental indicators
- Historical trend analysis

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
