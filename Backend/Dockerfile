FROM python:3.7

WORKDIR /Backend

COPY requirements.txt /Backend
COPY uploads /Backend/uploads
RUN pip install -r requirements.txt

ENV MODEL /Backend/myModel.ckpt

COPY Backend.py /Backend
COPY myModel.ckpt /Backend

EXPOSE 5000

CMD ["python", "Backend.py"]