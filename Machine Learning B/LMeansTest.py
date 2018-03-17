from pyspark import  SparkContext
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.linalg import Vectors

sc = SparkContext( 'local', 'pyspark')
data = sc.textFile('D:/kmeans_data.txt')
parsedData = data.map(lambda s : Vectors.dense(map(lambda x : float(x), s.split(' ')))).cache()
#设置簇的个数为3
numClusters = 3
#迭代20次
numIterations = 20
#运行10次,选出最优解
runs = 10
#设置初始K选取方式为k-means++
initMode = "k-means||"
clusters = KMeans.train(
    parsedData, 3, maxIterations=20, runs=10, initializationMode=initMode, seed=50, initializationSteps=5, epsilon=1e-4)
#clusters = KMeans().setInitializationMode(initMode).setK(numClusters).setMaxIterations(numIterations).run(parsedData)

#打印出测试数据属于哪个簇
print('\n'.join(parsedData.map(lambda v : str(v) + " belong to cluster :" + str(clusters.predict(v))).collect()))
#Evaluateclustering by computing Within Set Sum of Squared Errors
WSSSE = clusters.computeCost(parsedData)
print("WithinSet Sum of Squared Errors = " + str(WSSSE))

a21 =clusters.predict(Vectors.dense(1.2,1.3))
a22 =clusters.predict(Vectors.dense(4.1,4.2))

#打印出中心点
print("Clustercenters:")
for center in clusters.clusterCenters :
    print(center)

print("Prediction of (1.2,1.3)-->" + str(a21))
print("Prediction of (4.1,4.2)-->" + str(a22))


sc.stop()
