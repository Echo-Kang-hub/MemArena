FROM python:3.12-slim

WORKDIR /app

# 安装 uv，用于更快的依赖解析与安装
RUN pip install --no-cache-dir uv

# 先复制依赖定义，利用 Docker layer cache 提升构建速度
COPY pyproject.toml /app/pyproject.toml

# 使用 uv pip 安装项目运行依赖到系统环境
RUN uv pip install --system .

# 复制后端源码
COPY app /app/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
