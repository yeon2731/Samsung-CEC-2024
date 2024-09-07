FROM python:3.9-slim

RUN pip install torch numpy datasets transformers accelerate

WORKDIR /

COPY test_script.py .

CMD ["python", "test_script.py"]

