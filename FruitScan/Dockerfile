FROM python:3.11-slim-bookworm

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install --upgrade pip  && \
    pip3 install --no-cache-dir -r requirements.txt

# Install the necessary library
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get install -y libglib2.0-0 && \
    apt-get install -y graphviz

COPY . /app/

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]