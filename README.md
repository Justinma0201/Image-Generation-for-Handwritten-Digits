# Image-Generation-for-Handwritten-Digits

## 安裝與準備（以下指令皆假設當前資料夾為 `HW3_114064558/`）

### 必要條件

環境：

  * **torch==2.3.1+cu121**
  * **torchvision==0.18.1+cu121**
  * **tqdm==4.67.1**
  * **scikit-learn==1.7.2**
  * **Numpy 1.26.4**
  * **pandas==2.3.3**
  * **pillow==11.3.0**
  * **opencv-python==4.11.0.86**
  * **scipy==1.16.3**
  * **wandb==0.23.0**

**安裝依賴套件：**

```bash
pip install -r ./code_114064558/requirements.txt
```
-----
##  預期檔案結構

請將訓練圖片放置至名為 **data** 的資料夾並存放於 **hw3_114064558** 資料夾底下

```bash
hw3_114064558
     |-------- report_114064558.pdf
     |-------- code_114064558  
          |-------- src
                    |-------- main.py
          |-------- readme.md
          |-------- requirements.txt
     |-------- data
     |-------- image_114064558
```
-----
##  執行流程（以下指令皆假設當前資料夾為 `HW3_114064558/`）
**若路徑出現錯誤請再換成絕對路徑執行**

### 1\. 模型訓練 (`main.py`)
在大約第388行處將 mode 改為 **train** 後執行
```bash
python ./code_114064558/src/main.py
```

### 2\. 生成圖片 (`main.py`)

在大約第388行處將 mode 改為 **sample** 後執行
> 預測完成，得 **image_114064558**

```bash
python ./code_114064558/src/main.py
```

-----

### 3\. 生成Diffusion Process (`main.py`)

在大約第388行處將mode改為 **visualize** 後執行
> 預測完成，得 **diffusion_process_8x8.png**

```bash
python ./code_114064558/src/main.py
```

## 預期輸出與結果

生成的影像會儲存在 `./image_114064558`

Diffusion Process則會存至 `./diffusion_process_8x8.png`
