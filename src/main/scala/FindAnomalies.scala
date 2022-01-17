import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object FindAnomalies {
  def main(args: Array[String]): Unit = {
    var now = System.nanoTime()

    val spark = SparkSession.builder().appName("kMeansProject").master("local[2]").getOrCreate()

    var df = spark.read.options(Map("delimiter"->","))
      .csv("data-example2122-pps.txt")

    df = df.na.drop()

    var cdf = df.select(df.columns.map{
      case column@"_c0" =>
      col(column).cast("Float").as(column)
      case column@"_c1" =>
        col(column).cast("Float").as(column)
    }: _*)


    val minMaxC0 = cdf.agg(min("_c0"), max("_c0")).head()
    val minC0 = minMaxC0.getFloat(0)
    val maxC0 = minMaxC0.getFloat(1)

    val minMaxC1 = cdf.agg(min("_c1"), max("_c1")).head()
    val minC1 = minMaxC1.getFloat(0)
    val maxC1 = minMaxC1.getFloat(1)

    cdf = cdf.withColumn("_c0", (col("_c0")- minC0) / (maxC0 - minC0))
    cdf = cdf.withColumn("_c1", (col("_c1")- minC1) / (maxC1 - minC1))

    val cols = Array("_c0", "_c1")
    val assembler = new VectorAssembler().setInputCols(cols).setOutputCol("features")
    val featureDf = assembler.transform(cdf)

    val kmeans = new KMeans().setK(5).setSeed(1L)
    val model = kmeans.fit(featureDf)

    // Make predictions
    var predictions = model.transform(featureDf)

    // Evaluate clustering by computing Silhouette score
    val evaluator = new ClusteringEvaluator()

    val silhouette = evaluator.evaluate(predictions)

    predictions = predictions.sort("prediction")

    predictions = predictions.withColumn("distance",
      when(col("prediction")===0, sqrt(pow(col("_c0") - model.clusterCenters(0).toArray(0), 2) + pow(col("_c1") - model.clusterCenters(0).toArray(1), 2)))
     .when(col("prediction")===1, sqrt(pow(col("_c0") - model.clusterCenters(1).toArray(0), 2) + pow(col("_c1") - model.clusterCenters(1).toArray(1), 2)))
     .when(col("prediction")===2, sqrt(pow(col("_c0") - model.clusterCenters(2).toArray(0), 2) + pow(col("_c1") - model.clusterCenters(2).toArray(1), 2)))
     .when(col("prediction")===3, sqrt(pow(col("_c0") - model.clusterCenters(3).toArray(0), 2) + pow(col("_c1") - model.clusterCenters(3).toArray(1), 2)))
     .when(col("prediction")===4, sqrt(pow(col("_c0") - model.clusterCenters(4).toArray(0), 2) + pow(col("_c1") - model.clusterCenters(4).toArray(1), 2)))
    )
    val avgDistance = predictions.groupBy("prediction").agg(avg(col("distance")))

    val divDistance = predictions.withColumn("divDistance",
      when(col("prediction")===0, col("distance") / avgDistance.take(1)(0).getAs[Float]("avg(distance)"))
     .when(col("prediction")===1, col("distance") / avgDistance.take(2)(0).getAs[Float]("avg(distance)"))
     .when(col("prediction")===2, col("distance") / avgDistance.take(3)(0).getAs[Float]("avg(distance)"))
     .when(col("prediction")===3, col("distance") / avgDistance.take(4)(0).getAs[Float]("avg(distance)"))
     .when(col("prediction")===4, col("distance") / avgDistance.take(5)(0).getAs[Float]("avg(distance)"))
    )
    println("Outliers:")
    divDistance.filter(col("divDistance") > 2.4).show()

    spark.stop()
    var timeElapsed = System.nanoTime() - now
    println("Time elapsed: " + (timeElapsed/1000000000) + " sec.")
  }
}
