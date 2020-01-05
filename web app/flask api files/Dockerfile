FROM python:3.7-onbuild

EXPOSE 500

CMD gunicorn deploy:app --log-file - --bind 0.0.0.0:500
