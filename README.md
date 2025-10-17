### Setting up Spark History Server
export SPARK_HOME=/path/to/spark
mkdir -p /tmp/spark-events
chmod 777 /tmp/spark-events
uv run $SPARK_HOME/sbin/start-history-server.sh

#### and make sure to include following lines in the code
from pyspark import SparkConf, SparkContext
conf = SparkConf().setAppName("TestApp")
conf.set("spark.eventLog.enabled", "true")
conf.set("spark.eventLog.dir", "file:///tmp/spark-events")
sc = SparkContext(conf=conf)

The UI is accessible at http://localhost:18080/

#### to stop the history server,
uv run $SPARK_HOME/sbin/stop-history-server.sh
