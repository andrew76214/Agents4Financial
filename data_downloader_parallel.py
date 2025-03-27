import os
import csv
import datetime
import yt_dlp
import concurrent.futures
import threading
import whisper
from youtube_transcript_api import YouTubeTranscriptApi

class Video_Downloader:
    def __init__(self, channel_url, output_file="video/yutinghao_finance_videos.csv", max_videos=10):
        self.channel_url = channel_url
        self.output_file = output_file
        self.max_videos = max_videos
        self.video_info_list = []
        self.today_str = datetime.datetime.now().strftime("%Y%m%d")

    def fetch_video_list(self):
        ydl_opts = {
            'ignoreerrors': True,
            'quiet': True,
            'skip_download': True,
            'extract_flat': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(self.channel_url, download=False)
            videos = info.get('entries', [])
            return videos

    def download_transcript(self, video_id):
        try:
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'zh-TW'])
            transcript_text = " ".join([seg['text'] for seg in transcript_data])
            return transcript_text
        except Exception as e:
            print(f"無法取得影片 {video_id} 的逐字稿: {e}")
            return ""

    def process_single_video(self, video):
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

    def run(self):
        self.process_videos()


class Video_Downloader_With_Whisper(Video_Downloader):
    def __init__(self, channel_url, output_file="video/yutinghao_finance_videos.csv", max_videos=10):
        super().__init__(channel_url, output_file, max_videos)
        # 使用 thread-local storage 為每個執行緒建立自己的 Whisper 模型實例
        self.local = threading.local()

    def get_whisper_model(self):
        # 如果目前執行緒還未載入模型，就載入並儲存在 thread-local storage 中
        if not hasattr(self.local, "whisper_model"):
            try:
                print("載入 Whisper 模型...")
                self.local.whisper_model = whisper.load_model("large-v3")
            except ImportError as e:
                print("請先安裝 openai-whisper: pip install openai-whisper")
                raise e
        return self.local.whisper_model

    def download_video_and_transcribe(self, video_id):
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

        model = self.get_whisper_model()
        print("使用 Whisper 進行語音轉文字...")
        result = model.transcribe(filename)
        transcript = result.get("text", "")
        return transcript

    def process_single_video(self, video):
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
