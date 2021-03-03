FROM openjdk:8-jdk-stretch
RUN apt-get update
RUN apt-get install apt-transport-https
RUN echo "deb https://dl.bintray.com/sbt/debian /" | tee -a /etc/apt/sources.list.d/sbt.list
RUN curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | apt-key add
RUN apt-get update
RUN apt-get install sbt
COPY . /opt/trustvis/enforcer
WORKDIR /opt/trustvis/enforcer
RUN /usr/bin/sbt compile
EXPOSE 3100
ENTRYPOINT ["/usr/bin/sbt", "run"]
#ENTRYPOINT ["/bin/bash"]
