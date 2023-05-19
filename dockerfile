FROM python:3.10.3
COPY . .
RUN pip install scikit-learn
RUN pip install pandas
RUN pip install numpy
ENTRYPOINT [ "python3" ]
CMD [ "./model/model.py" ]