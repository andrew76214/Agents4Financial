import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import csv
import datetime
import subprocess
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, EncoderDecoderCache
import torch
import librosa

def convert_to_mono(input_file, output_file):
    """
    利用 ffmpeg 將輸入檔案轉換成單聲道音訊。
    """
    command = ["ffmpeg", "-y", "-i", input_file, "-ac", "1", output_file]
    subprocess.run(command, check=True)
    print(f"已將 {input_file} 轉換為單聲道並儲存至 {output_file}")

class LocalVideoTranscriber:
    def __init__(self, video_folder="stream_video", output_csv="./data/stream_video_transcripts.csv"):
        self.video_folder = video_folder
        self.output_csv = output_csv
        self.video_info_list = []
        self.today_str = datetime.datetime.now().strftime("%Y%m%d")
        # Load model directly
        self.processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3")

    def transcribe_videos(self):
        """
        逐一處理 stream_video 資料夾中的影片，
        將影片音訊轉為單聲道後使用 CrisperWhisper 模型進行語音轉文字，
        並將結果儲存至 video_info_list。
        """
        for filename in os.listdir(self.video_folder):
            file_path = os.path.join(self.video_folder, filename)
            # 只處理影片檔案 (根據副檔名過濾)
            if os.path.isfile(file_path) and filename.lower().endswith(('.mp4', '.mkv', '.mov', '.avi')):
                print(f"處理影片: {filename}")
                # 產生單聲道音訊檔案
                mono_audio_file = os.path.join(self.video_folder, f"{os.path.splitext(filename)[0]}_mono.wav")
                convert_to_mono(file_path, mono_audio_file)

                print("使用 CrisperWhisper 模型進行語音轉文字...")
                # 使用 processor 和 model 進行語音轉文字
                audio_data, _ = librosa.load(mono_audio_file, sr=16000)
                inputs = self.processor(audio_data, return_tensors="pt", sampling_rate=16000)
                inputs["attention_mask"] = torch.ones(inputs["input_features"].shape[:-1], dtype=torch.long)
                outputs = self.model.generate(
                    **inputs,
                    language="chinese",  # Ensure translation to chinese
                    forced_decoder_ids=None,  # Remove conflicting forced_decoder_ids
                    past_key_values=EncoderDecoderCache.from_legacy_cache(None)
                )
                transcript = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
                self.video_info_list.append({
                    "檔名": filename,
                    "逐字稿": transcript,
                    "日期": self.today_str
                })
                self.save_to_csv()  # 每處理完一份影片，就儲存進 CSV

    def save_to_csv(self):
        """
        將收集到的影片逐字稿資訊儲存成 CSV 檔案。
        """
        folder = os.path.dirname(self.output_csv)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
            print(f"建立資料夾：{folder}")

        keys = ["檔名", "逐字稿", "日期"]
        with open(self.output_csv, "w", newline='', encoding="utf-8-sig") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            for info in self.video_info_list:
                writer.writerow(info)
        print(f"已儲存逐字稿資訊至 {self.output_csv}")

    def run(self):
        """
        主流程：處理所有影片並儲存逐字稿資訊。
        """
        self.transcribe_videos()
        self.save_to_csv()

def main():
    transcriber = LocalVideoTranscriber(video_folder="stream_video", output_csv="stream_video_transcripts.csv")
    transcriber.run()

if __name__ == '__main__':
    main()
