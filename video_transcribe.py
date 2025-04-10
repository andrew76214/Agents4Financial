import os
import csv
import datetime
import yt_dlp
import subprocess
import whisper
import re  # <-- added import for regular expressions

def convert_to_mono(input_file, output_file):
    """
    利用 ffmpeg 將輸入檔案轉換成單聲道音訊。
    """
    command = ["ffmpeg", "-y", "-i", input_file, "-ac", "1", output_file]
    subprocess.run(command, check=True)
    print(f"已將 {input_file} 轉換為單聲道並儲存至 {output_file}")

class VideoDownloader:
    def __init__(self, output_file="video/yutinghao_finance_videos.csv"):
        """
        output_file: 儲存結果的 CSV 檔案路徑，預設存在 video 資料夾中
        cookies_file: 若使用 cookies 檔案驗證，請填入檔案路徑，例如 "./all_cookies.txt"
        cookies_browser: 若使用瀏覽器自動導入 cookies，請填入瀏覽器名稱（例如 "chrome"）
        """
        self.output_file = output_file
        self.video_info_list = []
        self.today_str = datetime.datetime.now().strftime("%Y%m%d")
        # 載入 OpenAI Whisper 模型
        self.asr_model = whisper.load_model("large-v3")

    def transcribe_local_video(self, video_path):
        """
        處理本機影片轉成單聲道並使用 Whisper 進行轉錄。
        """
        mono_audio_file = os.path.splitext(video_path)[0] + "_mono.wav"
        convert_to_mono(video_path, mono_audio_file)
        print("使用 OpenAI Whisper 模型進行語音轉文字...")
        result = self.asr_model.transcribe(mono_audio_file)
        transcript = result.get("text", "")
        return transcript

    def process_videos(self):
        """
        改成處理 stream_video/ 資料夾中的所有影片，
        使用 OpenAI Whisper 模型產生逐字稿，
        並將結果存入 video_info_list。每處理完一部影片就存檔。
        """
        video_dir = "stream_videos_2"
        if not os.path.exists(video_dir):
            print(f"資料夾 {video_dir} 不存在")
            return
        files = os.listdir(video_dir)
        # Sort files by date (assumed formats: YYYY/M/D, YYYY/MM/DD, YYYY/M/DD, or YYYY/MM/D in filename)
        def extract_date(filename):
            m = re.search(r'(\d{4})[\/-](\d{1,2})[\/-](\d{1,2})', filename)
            if m:
                return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            else:
                return datetime.date.min
        files.sort(key=extract_date)
        count = 0
        for file in files:
            if not file.lower().endswith((".mp4", ".mkv", ".webm")):
                continue
            video_path = os.path.join(video_dir, file)
            video_id = os.path.splitext(file)[0]
            title = file
            url = video_path  # 本機檔案路徑作為 URL
            transcript = self.transcribe_local_video(video_path)
            self.video_info_list.append({
                "video_id": video_id,
                "title": title,
                "url": url,
                "transcript": transcript,
                "date": self.today_str
            })
            self.save_to_csv()   # 每處理完一部影片即存檔
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
        file_exists = os.path.exists(self.output_file)
        # Open file in append mode if exists, else write mode to create new file with header
        mode = "a" if file_exists else "w"
        with open(self.output_file, mode, newline='', encoding="utf-8-sig") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            if not file_exists:
                writer.writeheader()
            # Write only the latest video info added
            writer.writerow(self.video_info_list[-1])
        print(f"已儲存影片資訊至 {self.output_file}")

    def run(self):
        """
        主流程：處理影片資訊（每部影片處理後即存檔）。
        """
        self.process_videos()

def main():
    output_file = "video/yutinghao_finance_videos_v1.1.csv"
    downloader = VideoDownloader(output_file)
    downloader.run()

if __name__ == '__main__':
    main()
