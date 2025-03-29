import os, csv
import datetime
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
import openai
from dotenv import load_dotenv

# 從 .env 檔案中讀取環境變數
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class Video_Downloader:
    def __init__(self, channel_url, output_file="video/yutinghao_finance_videos.csv", max_videos=10000):
        """
        channel_url: 財經網紅頻道或直播清單 URL，例如：https://www.youtube.com/@yutinghaofinance/streams
        output_file: 儲存結果的 CSV 檔案路徑，預設存在 video 資料夾中
        max_videos: 要處理的最大影片數（預設 10000，可依需求調整）
        """
        self.channel_url = channel_url
        self.output_file = output_file
        self.max_videos = max_videos
        self.video_info_list = []
        self.today_str = datetime.datetime.now().strftime("%Y%m%d")

    def fetch_video_list(self):
        """
        使用 yt-dlp 從頻道 URL 擷取影片列表（採用 extract_flat 加速處理）。
        """
        ydl_opts = {
            'ignoreerrors': True,
            'quiet': True,
            'skip_download': True,
            'extract_flat': True  # 僅擷取影片列表，不取得完整資訊
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(self.channel_url, download=False)
            videos = info.get('entries', [])
            return videos

    def download_transcript(self, video_id):
        """
        利用 youtube-transcript-api 嘗試取得影片逐字稿，
        若無法取得則回傳空字串。
        """
        try:
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'zh-TW'])
            transcript_text = " ".join([seg['text'] for seg in transcript_data])
            return transcript_text
        except Exception as e:
            print(f"無法取得影片 {video_id} 的逐字稿: {e}")
            return ""

    def process_videos(self):
        """
        依序處理影片列表，取得每支影片的基本資訊及逐字稿，並存入 video_info_list。
        """
        videos = self.fetch_video_list()
        count = 0
        for video in videos:
            if count >= self.max_videos:
                break
            if video is None:
                continue

            video_id = video.get('id')
            title = video.get('title', 'No Title')
            url = f"https://www.youtube.com/watch?v={video_id}"
            transcript = self.download_transcript(video_id)

            self.video_info_list.append({
                "video_id": video_id,
                "title": title,
                "url": url,
                "transcript": transcript,
                "date": self.today_str
            })
            count += 1

    def save_to_csv(self):
        """
        將收集到的影片資訊儲存成 CSV 檔案，並確保儲存資料夾存在。
        """
        folder = os.path.dirname(self.output_file)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
            print(f"建立資料夾：{folder}")

        keys = ["video_id", "title", "url", "transcript", "date"]
        with open(self.output_file, "w", newline='', encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            for info in self.video_info_list:
                writer.writerow(info)
        print(f"已儲存影片資訊至 {self.output_file}")

    def run(self):
        """
        主流程：處理影片資訊後儲存至 CSV。
        """
        self.process_videos()
        self.save_to_csv()


class Video_Downloader_With_Whisper(Video_Downloader):
    def __init__(self, channel_url, output_file="video/yutinghao_finance_videos.csv", max_videos=10):
        super().__init__(channel_url, output_file, max_videos)
        # 使用 OpenAI 的 Whisper API，不再需要本地模型載入

    def download_video_and_transcribe(self, video_id):
        """
        使用 yt-dlp 下載影片的最高畫質（結合最佳影片與最佳音訊），
        然後利用 OpenAI Whisper API 將影片中的音訊轉成文字。
        """
        # 確保 video 資料夾存在
        video_dir = "video"
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
            print(f"建立資料夾：{video_dir}")

        video_url = f"https://www.youtube.com/watch?v={video_id}"
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'merge_output_format': 'mp4',
            'outtmpl': f"{video_dir}/{video_id}.%(ext)s",
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            # 取得下載後的完整檔案路徑
            filename = ydl.prepare_filename(info)
            print(f"下載完成，檔案儲存於: {filename}")

        # 呼叫 OpenAI Whisper API 進行語音轉文字
        print("使用 OpenAI Whisper API 進行語音轉文字...")
        with open(filename, "rb") as audio_file:
            result = openai.Audio.transcribe("whisper-1", audio_file)
        transcript = result.get("text", "")
        return transcript

    def process_videos(self):
        """
        依序處理影片列表，若原始逐字稿取得失敗則使用 OpenAI Whisper API 產生逐字稿。
        """
        videos = self.fetch_video_list()
        count = 0
        for video in videos:
            if count >= self.max_videos:
                break
            if video is None:
                continue

            video_id = video.get('id')
            title = video.get('title', 'No Title')
            url = f"https://www.youtube.com/watch?v={video_id}"
            transcript = self.download_transcript(video_id)

            # 當逐字稿為空時，使用 OpenAI Whisper API 產生逐字稿
            if not transcript:
                print(f"影片 {video_id} 沒有提供逐字稿，改用 OpenAI Whisper API 處理...")
                transcript = self.download_video_and_transcribe(video_id)

            self.video_info_list.append({
                "video_id": video_id,
                "title": title,
                "url": url,
                "transcript": transcript,
                "date": self.today_str
            })
            count += 1

# 使用範例：
# downloader = Video_Downloader_With_Whisper("https://www.youtube.com/@yutinghaofinance/streams", max_videos=5)
# downloader.run()
