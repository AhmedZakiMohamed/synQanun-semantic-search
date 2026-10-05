FROM python:3.10-slim

WORKDIR /app


COPY requirements.txt .

# Install and Update pip, then install dependencies from requirements.txt  
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
