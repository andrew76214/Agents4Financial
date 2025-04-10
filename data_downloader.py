import os
import csv
import datetime
import yt_dlp
import subprocess
from youtube_transcript_api import YouTubeTranscriptApi
import whisper

def convert_to_mono(input_file, output_file):
    """
    利用 ffmpeg 將輸入檔案轉換成單聲道音訊。
    """
    command = ["ffmpeg", "-y", "-i", input_file, "-ac", "1", output_file]
    subprocess.run(command, check=True)
    print(f"已將 {input_file} 轉換為單聲道並儲存至 {output_file}")

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
        self.video_info_list = []
        self.today_str = datetime.datetime.now().strftime("%Y%m%d")
        # 使用 Whisper large-v3 模型，無需 API key
        self.asr_model = whisper.load_model("large-v3")

    def fetch_video_list(self):
        """
        使用 yt-dlp 從頻道 URL 擷取影片列表（採用 extract_flat 加速處理）。
        """
        ydl_opts = {
            'ignoreerrors': True,
            'quiet': True,
            'skip_download': True,
            'extract_flat': True,  # 僅擷取影片列表，不取得完整資訊
            'cookies': 'all_cookies_new.txt'
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
            print(f"取得影片 {video_id} 的逐字稿成功")
            return transcript_text
        except Exception as e:
            print(f"無法取得影片 {video_id} 的逐字稿: {e}")
            return ""

    def download_video_and_transcribe(self, video_id):
        """
        下載影片並使用 Whisper 模型進行語音轉文字，
        回傳轉換後的文字結果。
        """
        # 確保 video 資料夾存在
        video_dir = "video"
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
            print(f"建立資料夾：{video_dir}")

        video_url = f"https://www.youtube.com/watch?v={video_id}"
        ydl_opts = {
            'format': 'bestaudio/best',
            'merge_output_format': 'mp4',
            'outtmpl': f"{video_dir}/{video_id}.%(ext)s",
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            downloaded_file = ydl.prepare_filename(info)
            print(f"下載完成，檔案儲存於: {downloaded_file}")

        # 轉換下載的影片音訊為單聲道 wav 檔案
        mono_audio_file = os.path.join(video_dir, f"{video_id}_mono.wav")
        convert_to_mono(downloaded_file, mono_audio_file)

        print("使用 Whisper large-v3 模型進行語音轉文字...")
        # 使用 Whisper 模型轉錄，並傳回轉換後的文字
        result = self.asr_model.transcribe(mono_audio_file)
        transcript = result.get("text", "")
        return transcript

    def process_videos(self):
        """
        依序處理影片列表，若原始逐字稿取得失敗則使用 Whisper 模型產生逐字稿，
        並將結果存入 video_info_list。
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

            # 當逐字稿為空時，改用 Whisper 模型進行轉錄
            if not transcript:
                print(f"影片 {video_id} 沒有提供逐字稿，改用 Whisper 進行轉錄...")
                transcript = self.download_video_and_transcribe(video_id)

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

def main():
    """
    主程式進入點，設定頻道 URL、最大影片數等參數，並執行影片處理流程。
    """
    channel_url = "https://www.youtube.com/@yutinghaofinance/streams"
    max_videos = 10000
    output_file = "video/yutinghao_finance_videos.csv"
    downloader = VideoDownloader(channel_url, output_file, max_videos)
    downloader.run()

if __name__ == '__main__':
    main()
