import os
import csv
import whisper
import opencc

def transcribe_video(model, video_path, converter):
    """
    使用 Whisper 對影片進行語音辨識，並將結果轉換為繁體中文。
    """
    result = model.transcribe(video_path, language="zh")
    simplified_text = result.get("text", "")
    traditional_text = converter.convert(simplified_text)
    return traditional_text

def main():
    video_dir = "stream_video"            # 存放影片的資料夾
    output_csv = "transcripts_video_v1.1.csv"   # 輸出 CSV 檔案

    # 載入 Whisper large-v3 模型
    print("正在載入 Whisper 模型，請稍候...")
    model = whisper.load_model("large-v3")
    print("模型載入完成。")
    
    # 建立簡體轉繁體的 OpenCC 轉換器
    converter = opencc.OpenCC('s2t')

    # 建立 CSV 並寫入表頭
    with open(output_csv, "w", newline='', encoding="utf-8-sig") as csvfile:
        fieldnames = ["video_name", "transcript"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 遍歷資料夾中的影片
        for filename in os.listdir(video_dir):
            if filename.lower().endswith(('.mkv', 'webm')):
                video_path = os.path.join(video_dir, filename)
                print(f"開始處理影片：{filename}")
                try:
                    transcript = transcribe_video(model, video_path, converter)
                except Exception as e:
                    print(f"處理 {filename} 時發生錯誤: {e}")
                    transcript = ""
                writer.writerow({
                    "video_name": filename,
                    "transcript": transcript
                })
                print(f"已儲存 {filename} 的逐字稿至 CSV。")

if __name__ == "__main__":
    main()
