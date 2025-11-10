# 步驟 1：使用支援 CUDA 11.8 並原生搭載 Python 3.10 的 Ubuntu 22.04 映像檔
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# 步驟 2：設定時區，避免 apt-get 卡住
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei

# 步驟 3：安裝 pip 和 *所有* 必要的系統工具 (包含完整的 FFmpeg 編解碼器)
RUN apt-get update && apt-get install -y \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    libmpg123-dev \
    libavcodec-extra \
    && rm -rf /var/lib/apt/lists/*

# 步驟 4：建立軟連結，讓 "python" 指令指向 "python3"
RUN ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# 步驟 5：設定工作目錄
WORKDIR /app

# 步驟 6：複製 requirements.txt
COPY requirements.txt .

# 步驟 7：安裝 Python 依賴
RUN pip install --no-cache-dir \
    -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118

# 步驟 8：複製剩餘的程式碼
COPY . .

# 步驟 9：建立目錄
RUN mkdir -p uploads outputs model_cache

# 步驟 10：開放埠號
EXPOSE 8000

# 步驟 11：執行
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
