# Web Voyager

Web Voyager 是 Agents4Financial 專案中的一個模組，旨在協助使用者自動辨識與視覺化網頁上具互動性的元素，進而支持金融數據抓取與分析。此專案包含兩個主要部分：

- **Jupyter Notebook (web_voyager.ipynb)**：展示如何整合網頁數據提取與互動操作的流程，並作為工具使用的範例。
- **JavaScript 程式 (mark_page.js)**：實作在網頁上標記所有可點擊及互動性元素的功能，以便開發者或分析師快速掌握頁面結構與元素資訊。

## 目錄
- [簡介](#簡介)
- [功能特性](#功能特性)
- [技術架構](#技術架構)
- [安裝與運行](#安裝與運行)
- [使用方法](#使用方法)
- [貢獻](#貢獻)
- [授權條款](#授權條款)
- [聯絡資訊](#聯絡資訊)

## 簡介
Web Voyager 為金融數據分析量身打造的工具，能夠自動掃描網頁中所有具有互動功能的元素（如按鈕、連結、輸入框等），並利用彩色框線與浮動標籤進行視覺化展示。透過與 Agents4Financial 系統的整合，使用者能更直觀地識別頁面結構與可操作項目，進一步提升數據抓取與決策效率。

## 功能特性
- **自動標記互動元素**：利用 mark_page.js，搜尋並標記所有可點擊或具互動性的 HTML 元素。
- **動態視覺化展示**：以隨機色彩建立邊框與標籤，直觀顯示每個元素的位置、類型及基本文字內容。
- **清晰的元素資訊回傳**：返回各元素中心點座標、類型、文字內容及 aria-label 以供進一步處理。
- **整合展示與分析**：透過 Jupyter Notebook 提供完整的示範流程，方便用戶理解工具運作與數據提取過程。

## 技術架構
- **前端技術**：
  - JavaScript (ES6)：主要用於頁面元素標記與視覺化。
  - HTML / CSS：自訂樣式與視覺效果（如自訂滾動條樣式）。
- **後端/應用層**：
  - Python：透過 Jupyter Notebook (web_voyager.ipynb) 示範網頁互動與數據抓取流程。
- **工具與依賴**：
  - Jupyter Notebook 或 Jupyter Lab（用於運行 web_voyager.ipynb）
  - 現代瀏覽器（支援 ES6 與 CSS3）

## 安裝與運行

### 先決條件
- Git
- Python 3.7 或以上版本
- Jupyter Notebook 或 Jupyter Lab
- 現代網頁瀏覽器

### 安裝步驟
1. **克隆專案**
    ```bash
    git clone https://github.com/andrew76214/Agents4Financial.git
    cd Agents4Financial/web_voyager
    ```
2. **啟動 Jupyter Notebook**
    ```bash
    jupyter notebook web_voyager.ipynb
    ```
    或者使用 Jupyter Lab：
    ```bash
    jupyter lab web_voyager.ipynb
    ```

## 使用方法

### Jupyter Notebook (web_voyager.ipynb)
- 此 Notebook 提供了一個完整的示範，說明如何運用 Web Voyager 進行網頁互動與數據提取。請依據 Notebook 中的說明逐步執行各個 Cell，以了解工具的工作流程及數據處理方式。

### JavaScript 程式 (mark_page.js)
- **主要功能**：
  - **markPage() 函式**：自動掃描網頁上所有符合條件（例如：可點擊、具互動性）的元素，並於畫面上以隨機色彩繪製虛線邊框，且在標記框角落顯示元素序號。
  - **unmarkPage() 函式**：清除先前所有的標記，恢復頁面原貌。
- **應用方式**：
  - 可將 mark_page.js 載入至任意網頁中，作為除錯工具或網頁結構分析工具使用。
  - 亦可封裝為書籤程式（Bookmarklet），以便在瀏覽任何網頁時快速啟動標記功能。

## 貢獻
我們歡迎所有對專案改進與功能擴展的貢獻！如果您有建議、發現錯誤或想提交新功能，請依照以下流程進行：
1. Fork 此專案
2. 建立新分支並進行修改
3. 提交 Pull Request
4. 如有疑問，請在 GitHub [Issues](https://github.com/andrew76214/Agents4Financial/issues) 中留言

## 授權條款
本專案採用 [MIT 授權條款](LICENSE) 授權。詳細內容請參閱專案根目錄中的 LICENSE 文件。

## 聯絡資訊
若有任何問題、回饋或合作意向，請在 GitHub 上提交 Issue 或直接聯絡專案負責人 [andrew76214](https://github.com/andrew76214)。
