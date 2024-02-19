FROM python:3.7

WORKDIR /Data-Preprocessing-and-Model-Building

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY Dataset /Data-Preprocessing-and-Model-Building/Dataset
COPY PassportClassification.py .

CMD ["python", "PassportClassification.py"]
