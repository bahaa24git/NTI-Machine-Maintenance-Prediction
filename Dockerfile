FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and training scripts plus data, then train models during image build
COPY app.py model_training.py train_models.py factory_sensor_simulator_2040.csv ./

# Train models so the artifact exists inside the container even when it's not committed
# (this avoids relying on a pre-built joblib file being present in the repo)
RUN python train_models.py

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
