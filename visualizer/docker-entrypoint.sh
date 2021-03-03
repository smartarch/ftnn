#!/bin/bash
set -e

function printHelp {
    cat <<EOF

Optional parameters:
  --trustedUrlBase XXX        - sets the trusted url of the TrustVis Visualizer instance (default: http://localhost:8080)
  --sandboxUrlBase XXX        - sets the sandbox url of the TrustVis Visualizer instance (default: http://localhost:8081)
  --withProxy                 - use if TrustVis Visualizer instance is behind an http reverse proxy
  --trustvisEnforcerUrl XXX   - sets TrustVis Enforcer host (default: http://trustvis-enforcer:3100/)
  --elasticsearchHost XXX               - sets elasticsearch host (default: elasticsearch)
  --mongoHost XXX             - sets mongo host (default: mongo)
  --redisHost XXX             - sets redis host (default: redis)
  --mySqlHost XXX             - sets mysql host (default: mysql)
EOF

    exit 1
}


urlBaseTrusted=http://localhost:8080
urlBaseSandbox=http://localhost:8081
withProxy=false
trustvisEnforcerUrl=http://trustvis-enforcer:3100/
mongoHost=mongo
redisHost=redis
mySqlHost=mysql
elasticsearchHost=elasticsearch

while [ $# -gt 0 ]; do
    case "$1" in
        --help)
            printHelp
            ;;
        --trustedUrlBase)
            urlBaseTrusted="$2"
            shift 2
            ;;
        --sandboxUrlBase)
            urlBaseSandbox="$2"
            shift 2
            ;;
        --withProxy)
            withProxy="$2"
            shift 2
            ;;
        --mongoHost)
            mongoHost="$2"
            shift 2
            ;;
        --redisHost)
            redisHost="$2"
            shift 2
            ;;
        --elasticsearchHost)
            elasticsearchHost="$2"
            shift 2
            ;;
        --mySqlHost)
            mySqlHost="$2"
            shift 2
            ;;
        --trustvisEnforcerUrl)
            trustvisEnforcerUrl="$2"
            shift 2
            ;;
        *)
            echo "Error: unrecognized option $1."
            printHelp
    esac
done

cat > server/config/production.yaml <<EOT
www:
  host: 0.0.0.0
  proxy: $wwwProxy
  secret: "`pwgen -1`"

  trustedPort: 8080
  trustedPortIsHttps: false
  sandboxPort: 8081
  sandboxPortIsHttps: false
  apiPort: 8082
  apiPortIsHttps: false

  trustedUrlBase: $urlBaseTrusted
  sandboxUrlBase: $urlBaseSandbox

mysql:
  host: $mySqlHost

redis:
  enabled: true
  host: $redisHost

elasticsearch:
  host: $elasticsearchHost

log:
  level: info
  
enforcer:
  url: $trustvisEnforcerUrl
EOT



# Wait for the other services to start
while ! nc -z $mySqlHost 3306; do sleep 1; done
while ! nc -z $redisHost 6379; do sleep 1; done
#while ! nc -z $mongoHost 27017; do sleep 1; done
while ! nc -z $elasticsearchHost 9200; do sleep 1; done
while ! mysql -h mysql -u trustvis --password=trustvis -e 'show databases'; do sleep 1; done

cd server
NODE_ENV=production node index.js
