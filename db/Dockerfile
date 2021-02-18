# For more information, please refer to https://aka.ms/vscode-docker-python
FROM mysql:8.0.19

EXPOSE 3306

ENV MYSQL_ROOT_PASSWORD root 

# COPY setup.sh /mysql/setup.sh

#RUN /mysql/setup.sh

#ADD datay.sh /
#COPY datay.sql /
#RUN /datay.sh
#ADD ./app/Points_to_Database.csv /home

CMD ["--secure-file-priv=/home"]
COPY datay.sql datay.sql
COPY setup.sql /docker-entrypoint-initdb.d/setup.sql