import os
import csv
import whisper
import opencc
import re

def get_date_from_filename(fn):
    """
    從檔名中提取日期。
    匹配形如 2024⧸12⧸2 的日期模式。
    """
    m = re.search(r'(\d{4})[⧸/-](\d{1,2})[⧸/-](\d{1,2})', fn)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return (0, 0, 0)

def transcribe_video(model, video_path, converter):
    """
    使用 Whisper 對影片進行語音辨識，並將結果轉換為繁體中文。
    """
    result = model.transcribe(video_path, language="zh")
    simplified_text = result.get("text", "")
    traditional_text = converter.convert(simplified_text)
    return traditional_text

def main():
    video_dir = "gooaye"            # 存放影片的資料夾
    output_csv = "transcripts_video_gooaye_v1.0.csv"   # 輸出 CSV 檔案

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

        # 使用日期排序資料夾中的影片
        files = os.listdir(video_dir)
        sorted_files = sorted(files, key=get_date_from_filename, reverse=True)
        for filename in sorted_files:
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
