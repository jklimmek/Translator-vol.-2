FROM bitnami/pytorch:1.13.1

WORKDIR /src/

COPY scripts/ scripts/
COPY settings/ settings/
COPY inference_models/ inference_models/
COPY templates/ templates/
COPY tokenizers/ tokenizers/
COPY notebooks/ notebooks/
COPY app.py .
COPY requirements.txt .

RUN pip install -r requirements.txt

CMD ["python", "app.py"]