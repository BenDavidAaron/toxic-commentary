FROM heroku/miniconda

ADD ./app/requirements.txt /tmp/requirements.txt

RUN pip install -qr /tmp/requirements.txt 

ADD ./app /opt/app/
WORKDIR /opt/app

CMD gunicorn --bind 0.0.0.0:$PORT wsgi