FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY inference.py /app/inference.py
COPY weights/ /app/weights/

CMD ["python", "-m", "inference", "--help"]