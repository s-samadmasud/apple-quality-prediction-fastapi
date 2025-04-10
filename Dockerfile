FROM python:3.10-slim

WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

EXPOSE 8000

CMD ["app:app", "--host", "0.0.0.0", "--port", "8000"]