import os
import csv
import sys
import datetime
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

class VideoDownloader:
    def __init__(self, channel_url, output_file="video/yutinghao_finance_videos.csv", max_videos=10000):
        """
        channel_url: 財經網紅頻道或直播清單 URL，例如：https://www.youtube.com/@yutinghaofinance/streams
        output_file: 儲存結果的 CSV 檔案路徑，預設存在 video 資料夾中
        max_videos: 要處理的最大影片數（預設 10000，可依需求調整）
        """
        self.channel_url = channel_url
        self.output_file = output_file
        self.max_videos = max_videos
        self.today_str = datetime.datetime.now().strftime("%Y%m%d")
        self.initialize_csv()

    def initialize_csv(self):
        """
        確保 CSV 檔案所在資料夾存在，若 CSV 檔案不存在則建立檔案並寫入表頭。
        """
        folder = os.path.dirname(self.output_file)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
            print(f"建立資料夾：{folder}")
        if not os.path.exists(self.output_file):
            with open(self.output_file, "w", newline='', encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["video_id", "title", "url", "transcript", "date"])
                writer.writeheader()

    def fetch_video_list(self):
        """
        使用 yt-dlp 從頻道 URL 擷取影片列表（僅取得影片基本資訊）。
        """
        ydl_opts = {
            'ignoreerrors': True,
            'quiet': True,
            'skip_download': True,
            'extract_flat': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(self.channel_url, download=False)
            videos = info.get('entries', [])
            return videos

    def download_transcript(self, video_id):
        """
        利用 youtube-transcript-api 嘗試取得影片逐字稿，
        若無法取得則回傳 None。
        """
        try:
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'zh-TW'])
            transcript_text = " ".join([seg['text'] for seg in transcript_data])
            print(f"取得影片 {video_id} 的逐字稿成功")
            return transcript_text
        except Exception as e:
            print(f"無法取得影片 {video_id} 的逐字稿: {e}")
            return None

    def append_to_csv(self, video_info):
        """
        將單一影片資訊附加到 CSV 檔案中
        """
        with open(self.output_file, "a", newline='', encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["video_id", "title", "url", "transcript", "date"])
            writer.writerow(video_info)
        print(f"已將影片 {video_info['video_id']} 資訊儲存至 {self.output_file}")

    def process_videos(self):
        """
        依序處理擷取到的影片列表，
        若遇到影片沒有逐字稿則結束程式（退出）。
        """
        videos = self.fetch_video_list()
        count = 0
        for video in videos:
            if count >= self.max_videos:
                break
            if video is None:
                continue
            video_id = video.get('id')
            if not video_id:
                continue
            title = video.get('title', 'No Title')
            url = f"https://www.youtube.com/watch?v={video_id}"
            transcript = self.download_transcript(video_id)
            if not transcript:
                print(f"影片 {video_id} 沒有提供逐字稿，結束程式")
                sys.exit(1)  # 遇到沒有逐字稿的影片就離開程式
            video_info = {
                "video_id": video_id,
                "title": title,
                "url": url,
                "transcript": transcript,
                "date": self.today_str
            }
            self.append_to_csv(video_info)
            count += 1

    def run(self):
        """
        主流程：擷取影片資訊及逐字稿後持續存入 CSV。
        """
        self.process_videos()

if __name__ == "__main__":
    channel_url = "https://www.youtube.com/@yutinghaofinance/streams"
    downloader = VideoDownloader(channel_url)
    downloader.run()
