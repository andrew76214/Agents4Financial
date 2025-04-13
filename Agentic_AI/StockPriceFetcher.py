import yfinance as yf
from datetime import datetime, timedelta

class StockPriceFetcher:
    """
    此類別用於取得指定股票在特定日期的開盤與收盤價。
    """
    def __init__(self, symbol):
        """
        初始化方法
        :param symbol: 股票代碼 (例如 "AAPL", "2330.TW" 等)
        """
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)

    def get_open_close(self, date_str):
        """
        取得指定日期的開盤和收盤價格。
        :param date_str: 指定日期的字串，格式為 "YYYY-MM-DD"
        :return: 一個 tuple (開盤價, 收盤價)，如果當日無交易資料則回傳 (None, None)
        """
        try:
            # 將日期字串轉換成日期物件
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            # 因為 yfinance 的 end 參數是排除在外的，所以將結束日期設定為指定日的隔天
            next_day = date_obj + timedelta(days=1)
            # 使用 history 方法取得交易資料
            data = self.ticker.history(start=date_str, end=next_day.strftime("%Y-%m-%d"))

            if data.empty:
                print("指定日期無交易資料")
                return None, None

            # 從資料中取出第一筆數據的開盤與收盤價
            open_price = data.iloc[0]['Open']
            close_price = data.iloc[0]['Close']
            return open_price, close_price
        except Exception as e:
            print("發生錯誤:", e)
            return None, None

# 使用範例
if __name__ == "__main__":
    # 建立物件並指定股票代碼 (這裡以蘋果公司股票代碼 "AAPL" 為例)
    fetcher = StockPriceFetcher("AAPL")
    # 指定查詢的日期，格式為 "YYYY-MM-DD"
    open_price, close_price = fetcher.get_open_close("2023-04-10")
    print("開盤價:", open_price, "收盤價:", close_price)
