import os
import csv
import whisper

def transcribe_video(model, video_path):
    """
    使用指定的 Whisper 模型對影片進行語音辨識，
    並回傳產生的逐字稿文字。
    """
    result = model.transcribe(video_path)
    return result.get("text", "")

def main():
    video_dir = "stream_videos_2"         # 存放影片的資料夾路徑
    output_csv = "transcripts.csv"         # 輸出的 CSV 檔案名稱

    # 載入 Whisper large-v3 模型
    # 若安裝的模型名稱不同，請依需求調整參數
    print("正在載入 Whisper 模型，請稍候...")
    model = whisper.load_model("large-v3")
    print("模型載入完成。")

    # 建立 CSV 檔案並寫入表頭
    with open(output_csv, "w", newline='', encoding="utf-8-sig") as csvfile:
        fieldnames = ["video_name", "transcript"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 逐一處理資料夾內的影片檔案
        for filename in os.listdir(video_dir):
            if filename.lower().endswith(('.mp4', '.mkv', '.avi')):
                video_path = os.path.join(video_dir, filename)
                print(f"開始處理影片：{filename}")
                try:
                    transcript = transcribe_video(model, video_path)
                except Exception as e:
                    print(f"處理 {filename} 時發生錯誤: {e}")
                    transcript = ""
                # 將結果寫入 CSV
                writer.writerow({
                    "video_name": filename,
                    "transcript": transcript
                })
                print(f"已儲存 {filename} 的逐字稿至 CSV。")

if __name__ == "__main__":
    main()
