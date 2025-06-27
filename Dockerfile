FROM python:3.11-slim

WORKDIR /app

# system libraries needed by implicit
RUN apt-get update && apt-get install -y libgomp1


COPY models ./models
COPY api/app.py .


RUN pip install --no-cache-dir fastapi uvicorn numpy scikit-learn pandas scipy implicit


EXPOSE 8080


CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
