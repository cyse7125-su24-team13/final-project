FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --timeout=1000 -r app/requirements.txt


EXPOSE 8080

CMD ["python", "app.py"]
