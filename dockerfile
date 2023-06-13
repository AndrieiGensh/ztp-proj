FROM python:3.10.3
COPY . .
RUN pip install scikit-learn
RUN pip install pandas
RUN pip install numpy
RUN pip install dask
RUN pip install flask
RUN python -m pip install "dask[distributed]" --upgrade
EXPOSE 5000
CMD [ "flask", "--app", "server/server", "run","--host","0.0.0.0","--port","5000"]