import os
import csv
import datetime
import yt_dlp
import concurrent.futures
import whisper
from youtube_transcript_api import YouTubeTranscriptApi

class Video_Downloader:
    def __init__(self, channel_url, output_file="video/yutinghao_finance_videos.csv", max_videos=10):
        """
        channel_url: 財經網紅頻道或直播清單 URL，例如：https://www.youtube.com/@yutinghaofinance/streams
        output_file: 儲存結果的 CSV 檔案路徑，預設存在 video 資料夾中
        max_videos: 要處理的最大影片數（預設 10 支，可依需求調整）
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

    def process_single_video(self, video):
        """
        處理單一影片的資訊取得，回傳包含影片基本資料的字典。
        """
        if video is None:
            return None
        video_id = video.get('id')
        title = video.get('title', 'No Title')
        url = f"https://www.youtube.com/watch?v={video_id}"
        transcript = self.download_transcript(video_id)
        return {
            "video_id": video_id,
            "title": title,
            "url": url,
            "transcript": transcript,
            "date": self.today_str
        }

    def process_videos(self):
        """
        使用 ThreadPoolExecutor 同時處理兩部影片，
        並在每部影片完成後，立即將資訊寫入 CSV 檔案中。
        """
        videos = self.fetch_video_list()
        videos_to_process = [video for video in videos if video is not None][:self.max_videos]

        # 確保輸出資料夾存在
        folder = os.path.dirname(self.output_file)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
            print(f"建立資料夾：{folder}")

        keys = ["video_id", "title", "url", "transcript", "date"]

        # 開啟 CSV 檔案並先寫入表頭
        with open(self.output_file, "w", newline='', encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_to_video = {
                    executor.submit(self.process_single_video, video): video for video in videos_to_process
                }
                for future in concurrent.futures.as_completed(future_to_video):
                    res = future.result()
                    if res:
                        writer.writerow(res)
                        self.video_info_list.append(res)
                        print(f"已儲存影片 {res['video_id']} 資訊到 CSV")
        print(f"所有影片資訊已儲存至 {self.output_file}")

    def run(self):
        """
        主流程：處理影片資訊後立即儲存到 CSV。
        """
        self.process_videos()


class Video_Downloader_With_Whisper(Video_Downloader):
    def __init__(self, channel_url, output_file="video/yutinghao_finance_videos.csv", max_videos=10):
        super().__init__(channel_url, output_file, max_videos)
        self.whisper_model = None  # 模型會在首次需要時載入

    def download_video_and_transcribe(self, video_id):
        """
        使用 yt-dlp 下載影片的最高畫質（結合最佳影片與最佳音訊），
        然後利用 Whisper 模型將影片中的音訊轉成文字。
        """
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
            filename = ydl.prepare_filename(info)
            print(f"下載完成，檔案儲存於: {filename}")

        if self.whisper_model is None:
            try:
                print("載入 Whisper 模型...")
                self.whisper_model = whisper.load_model("large-v3")
            except ImportError as e:
                print("請先安裝 openai-whisper: pip install openai-whisper")
                raise e

        print("使用 Whisper 進行語音轉文字...")
        result = self.whisper_model.transcribe(filename)
        transcript = result.get("text", "")
        return transcript

    def process_single_video(self, video):
        """
        覆寫 process_single_video，若無法取得逐字稿則使用 Whisper 模型處理。
        """
        if video is None:
            return None
        video_id = video.get('id')
        title = video.get('title', 'No Title')
        url = f"https://www.youtube.com/watch?v={video_id}"
        transcript = self.download_transcript(video_id)
        if not transcript:
            print(f"影片 {video_id} 沒有提供逐字稿，改用 Whisper 處理...")
            transcript = self.download_video_and_transcribe(video_id)
        return {
            "video_id": video_id,
            "title": title,
            "url": url,
            "transcript": transcript,
            "date": self.today_str
        }

    def process_videos(self):
        """
        同樣使用 ThreadPoolExecutor 同時處理兩部影片，
        並在每部影片完成後立即寫入 CSV 檔案中。
        """
        videos = self.fetch_video_list()
        videos_to_process = [video for video in videos if video is not None][:self.max_videos]

        folder = os.path.dirname(self.output_file)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
            print(f"建立資料夾：{folder}")

        keys = ["video_id", "title", "url", "transcript", "date"]

        with open(self.output_file, "w", newline='', encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_to_video = {
                    executor.submit(self.process_single_video, video): video for video in videos_to_process
                }
                for future in concurrent.futures.as_completed(future_to_video):
                    res = future.result()
                    if res:
                        writer.writerow(res)
                        self.video_info_list.append(res)
                        print(f"已儲存影片 {res['video_id']} 資訊到 CSV")
        print(f"所有影片資訊已儲存至 {self.output_file}")
