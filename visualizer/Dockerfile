FROM node:10.14-alpine

RUN apk add --update pwgen netcat-openbsd python make gcc git g++ bash mysql-client

# First install dependencies
COPY server/package.json /opt/trustvis/visualizer/server/package.json
#COPY server/package-lock.json /opt/trustvis/visualizer/server/package-lock.json
COPY client/package.json /opt/trustvis/visualizer/client/package.json
#COPY client/package-lock.json /opt/trustvis/visualizer/client/package-lock.json
COPY ivis-core/server/package.json /opt/trustvis/visualizer/ivis-core/server/package.json
#COPY ivis-core/server/package-lock.json /opt/trustvis/visualizer/ivis-core/server/package-lock.json
COPY ivis-core/client/package.json /opt/trustvis/visualizer/ivis-core/client/package.json
#COPY ivis-core/client/package-lock.json /opt/trustvis/visualizer/ivis-core/client/package-lock.json
COPY ivis-core/shared/package.json /opt/trustvis/visualizer/ivis-core/shared/package.json
#COPY ivis-core/shared/package-lock.json /opt/trustvis/visualizer/ivis-core/shared/package-lock.json
COPY ivis-core/embedding/package.json /opt/trustvis/visualizer/ivis-core/embedding/package.json
#COPY ivis-core/embedding/package-lock.json /opt/trustvis/visualizer/ivis-core/embedding/package-lock.json

WORKDIR /opt/trustvis/visualizer/

RUN for idx in client server ivis-core/shared ivis-core/client ivis-core/server ivis-core/embedding; do (cd $idx && npm install); done

# Later, copy the app files. That improves development speed as buiding the Docker image will not have
# to download and install all the NPM dependencies every time there's a change in the source code
COPY . /opt/trustvis/visualizer

RUN cd client && npm run build

EXPOSE 8080 8081 8082
ENTRYPOINT ["bash", "/opt/trustvis/visualizer/docker-entrypoint.sh"]
