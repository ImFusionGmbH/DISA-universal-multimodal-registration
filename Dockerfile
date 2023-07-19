FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install --no-install-recommends -y g++ && rm -rf /var/lib/apt/lists/*
COPY requirements.txt  requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir /training

COPY grid.py /training/grid.py
COPY dataset.py /training/dataset.py
COPY lc2.py /training/lc2.py
COPY neural.py /training/neural.py
COPY train.py /training/train.py

WORKDIR /training
CMD ["python", "train.py"]