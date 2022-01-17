import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object FindAnomalies {
  def main(args: Array[String]): Unit = {
    var now = System.nanoTime() // Αρχικοποίηση χρονομέτρου

    // Έναρξη του Spark
    val spark = SparkSession.builder().appName("kMeansProject").master("local[2]").getOrCreate()

    var df = spark.read.options(Map("delimiter"->",")).csv(args(0)) // Διαβάζουμε το datafile αγνοώντας τα ','
    df = df.na.drop() // Αφαιρούμε τις εγγραφές που έχουν κενά πεδία

    // Μετασχηματίζουμε τα δεδομένα σε float
    var cdf = df.select(df.columns.map{
      case column@"_c0" =>
      col(column).cast("Float").as(column)
      case column@"_c1" =>
        col(column).cast("Float").as(column)
    }: _*)

    // Εύρεση ελαχίστης και μέγιστης τιμής των στηλών "_c0" και "_c1" για να χρησιμοποιηθούν στον μετασχηματισμό
    val minMaxC0 = cdf.agg(min("_c0"), max("_c0")).head()
    val minC0 = minMaxC0.getFloat(0)
    val maxC0 = minMaxC0.getFloat(1)
    val minMaxC1 = cdf.agg(min("_c1"), max("_c1")).head()
    val minC1 = minMaxC1.getFloat(0)
    val maxC1 = minMaxC1.getFloat(1)

    // Εφαρμόζουμε τον μετασχηματισμό 0-1 σε κάθε στήλη
    cdf = cdf.withColumn("_c0", (col("_c0")- minC0) / (maxC0 - minC0))
    cdf = cdf.withColumn("_c1", (col("_c1")- minC1) / (maxC1 - minC1))

    // Τροποποίηση του data frame για να εφαρμοστεί ο k-means
    val cols = Array("_c0", "_c1")
    val assembler = new VectorAssembler().setInputCols(cols).setOutputCol("features")
    val featureDf = assembler.transform(cdf)
    val kMeans = new KMeans().setK(5).setSeed(1L)
    val model = kMeans.fit(featureDf)

    // Δημιουργία συστάδων
    var predictions = model.transform(featureDf)

    predictions = predictions.sort("prediction")  // Ταξινόμηση των εγγραφών με βάση τις συστάδες που ανήκουν

    // Υπολογίζουμε για κάθε εγγραφή την ευκλείδεια απόσταση του σημείου από το κέντρο της συστάδας που ανήκει
    predictions = predictions.withColumn("distance",
      when(col("prediction")===0, sqrt(pow(col("_c0") - model.clusterCenters(0).toArray(0), 2) + pow(col("_c1") - model.clusterCenters(0).toArray(1), 2)))
     .when(col("prediction")===1, sqrt(pow(col("_c0") - model.clusterCenters(1).toArray(0), 2) + pow(col("_c1") - model.clusterCenters(1).toArray(1), 2)))
     .when(col("prediction")===2, sqrt(pow(col("_c0") - model.clusterCenters(2).toArray(0), 2) + pow(col("_c1") - model.clusterCenters(2).toArray(1), 2)))
     .when(col("prediction")===3, sqrt(pow(col("_c0") - model.clusterCenters(3).toArray(0), 2) + pow(col("_c1") - model.clusterCenters(3).toArray(1), 2)))
     .when(col("prediction")===4, sqrt(pow(col("_c0") - model.clusterCenters(4).toArray(0), 2) + pow(col("_c1") - model.clusterCenters(4).toArray(1), 2)))
    )

    // Υπολογισμός της μέσης απόστασης των σημείων κάθε συστάδας
    val avgDistance = predictions.groupBy("prediction").agg(avg(col("distance")))

    // Διαιρούμε την απόσταση του κάθε σημείου με τη μέση απόσταση και θα χρησιμοποιήσουμε τον λόγο ώστε να εντοπίσουμε
    // τις ανωμαλίες
    // Όσο μεγαλύτερος ο λόγος, τόσο πιο απομακρυσμένο είναι το σημείο από το κέντρο της συστάδας του
    val divDistance = predictions.withColumn("divDistance",
      when(col("prediction")===0, col("distance") / avgDistance.take(1)(0).getAs[Float]("avg(distance)"))
     .when(col("prediction")===1, col("distance") / avgDistance.take(2)(0).getAs[Float]("avg(distance)"))
     .when(col("prediction")===2, col("distance") / avgDistance.take(3)(0).getAs[Float]("avg(distance)"))
     .when(col("prediction")===3, col("distance") / avgDistance.take(4)(0).getAs[Float]("avg(distance)"))
     .when(col("prediction")===4, col("distance") / avgDistance.take(5)(0).getAs[Float]("avg(distance)"))
    )

    println("Outliers:")  // Εκτύπωση των ανωμαλιών
    // Επιλέγουμε ως ανωμαλίες τις εγγραφές με λόγο άνω του 2.4
    divDistance.filter(col("divDistance") > 2.4).show()

    spark.stop()  // Τερματισμός Spark
    var timeElapsed = System.nanoTime() - now // Τερματισμός χρονομέτρου
    println("Time elapsed: " + (timeElapsed/1000000000) + " sec.")  // Εμφάνιση χρόνου εκτέλεσης
  }
}