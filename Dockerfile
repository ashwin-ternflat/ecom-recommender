FROM python:3.11-slim

WORKDIR /app

# Install system libraries needed by implicit
RUN apt-get update && apt-get install -y libgomp1

# Copy code and model
COPY models ./models
COPY api/app.py .

# Install Python dependencies
RUN pip install --no-cache-dir fastapi uvicorn numpy scikit-learn pandas scipy implicit

# ✅ Cloud Run requires port 8080
EXPOSE 8080

# ✅ Run on port 8080 to match Cloud Run's PORT env
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
