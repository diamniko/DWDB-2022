import javafx.scene.chart.ScatterChart
import org.apache.parquet.format.Util
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.clustering.KMeansModel
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.{MinMaxScaler, Normalizer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.functions.{col, column, lit, max, min, pow, shuffle, sqrt}
import org.apache.spark.sql.types.{FloatType, IntegerType}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.recommendation.ALS.Rating
import org.apache.spark.sql.Row

import scala.io.Source

object FindAnomalies {
  def main(args: Array[String]): Unit = {
    println("***********************************************************************************************")
    println("***********************************************************************************************")
    println("Hello, Spark!")

    val spark = SparkSession.builder().appName("kMeansProject").master("local[2]").getOrCreate()

    var df = spark.read.options(Map("delimiter"->","))
      .csv("data-example2122-pps.txt")
    df.show()

    df = df.na.drop()
    df.show()

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
    cdf.show()

    val cols = Array("_c0", "_c1")
    val assembler = new VectorAssembler().setInputCols(cols).setOutputCol("features")
    val featureDf = assembler.transform(cdf)

    val kmeans = new KMeans().setK(5).setSeed(1L)
    val model = kmeans.fit(featureDf)

    // Make predictions
    var predictions = model.transform(featureDf)
    predictions.show()

    // Evaluate clustering by computing Silhouette score
    val evaluator = new ClusteringEvaluator()

    val silhouette = evaluator.evaluate(predictions)
    println(s"Silhouette with squared euclidean distance = $silhouette")
    println("***************************************************")
    // Shows the result.
    println("Cluster Centers: ")
    model.clusterCenters.foreach(println)

    predictions = predictions.sort("prediction")
    predictions.show()

    predictions = predictions.select(df.columns.map{
      case column@"prediction" =>
        col(column).cast("Int").as(column)
    }: _*)

    predictions = predictions.withColumn("distance", lit(sqrt(pow(col("_c0") - model.clusterCenters(predictions.map(r => Rating(r.getAs[Int]("prediction"))).toArray(0), 2) + pow(col("_c1") - model.clusterCenters().toArray(1), 2))))
    val avgDistance = predictions.groupBy("prediction").agg(avg(col("distance")))
    val divDistance = distance.withColumn("divDistance", (col("distance")/col("avgDistance")))

    divDistance.filter(col("divDistance") > 1.5).show(false)

    println("***********************************************************************************************")
    println("***********************************************************************************************")
    println("Data parsed!")

    println("***********************************************************************************************")
    println("***********************************************************************************************")



    spark.stop()
    println("Spark stopped!")
    println("***********************************************************************************************")
    println("***********************************************************************************************")
  }
}
