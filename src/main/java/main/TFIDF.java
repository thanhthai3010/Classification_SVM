package main;
import java.util.Arrays;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;

import scala.Tuple2;

public class TFIDF {
	public static void main(String[] args) {
		SparkConf sparkConf = new SparkConf().setAppName("JavaBookExample").setMaster("local");
	    JavaSparkContext sc = new JavaSparkContext(sparkConf);

	    // Load 2 types of emails from text files: spam and ham (non-spam).
	    // Each line has text from one email.
	    JavaRDD<String> positive = sc.textFile("TFIDF_DATA/pos.txt");
	    JavaRDD<String> negative = sc.textFile("TFIDF_DATA/neg.txt");
	    
	    JavaPairRDD<String, Long>  termCounts = positive.union(negative).flatMap(new FlatMapFunction<String, String>() {

			/**
			 * 
			 */
			private static final long serialVersionUID = 1L;

			public Iterable<String> call(String content) throws Exception {
				return Arrays.asList(content.split(" "));
			}
		}).mapToPair(new PairFunction<String, String, Long>() {

			/**
			 * 
			 */
			private static final long serialVersionUID = 1L;

			public Tuple2<String, Long> call(String content) throws Exception {
				return new Tuple2<String, Long>(content, 1L);
			}
		}).reduceByKey(new Function2<Long, Long, Long>() {
			
			/**
			 * 
			 */
			private static final long serialVersionUID = 1L;

			public Long call(Long count1, Long count2) throws Exception {
				return count1 + count2;
			}
		});
		
		JavaPairRDD<String, Long> afterFilter = termCounts.filter(new Function<Tuple2<String,Long>, Boolean>() {
			
			/**
			 * 
			 */
			private static final long serialVersionUID = 1L;

			public Boolean call(Tuple2<String, Long> itemWordCount) throws Exception {
				if (itemWordCount._2 >= 10 && !Stopwords.isStopword(itemWordCount._1)) {
					return true;
				} else {
					return false;
				}
			}
		});
	    
//		termCounts.saveAsTextFile("termCounts");
//		afterFilter.saveAsTextFile("afterFilter");
		
		int sizeOfVocabulary = afterFilter.collect().size();
	    
	    // Create a HashingTF instance to map email text to vectors of 100 features.
	    final HashingTF tf = new HashingTF(sizeOfVocabulary);

	    // Each email is split into words, and each word is mapped to one feature.
	    // Create LabeledPoint datasets for positive (spam) and negative (ham) examples.
	    JavaRDD<LabeledPoint> positiveExamples = positive.map(new Function<String, LabeledPoint>() {
	      /**
	       * 
	       */
			private static final long serialVersionUID = 1L;

		public LabeledPoint call(String content) {
	        return new LabeledPoint(1, tf.transform(Arrays.asList(content.split(" "))));
	      }
	    });
	    JavaRDD<LabeledPoint> negativeExamples = negative.map(new Function<String, LabeledPoint>() {
	      /**
			 * 
			 */
			private static final long serialVersionUID = 1L;

		public LabeledPoint call(String content) {
	        return new LabeledPoint(0, tf.transform(Arrays.asList(content.split(" "))));
	      }
	    });
	    
	    JavaRDD<LabeledPoint> data = positiveExamples.union(negativeExamples);
	    JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.6, 0.4}, 11L);
	    JavaRDD<LabeledPoint> training = splits[0].cache();
	    JavaRDD<LabeledPoint> test = splits[1].cache();

	    // Create a Logistic Regression learner which uses the LBFGS optimizer.
	    LogisticRegressionWithSGD lrLearner = new LogisticRegressionWithSGD();
	    // Run the actual learning algorithm on the training data.
	    final LogisticRegressionModel model = lrLearner.run(training.rdd());

	    // Test on a positive example (spam) and a negative one (ham).
	    // First apply the same HashingTF feature transformation used on the training data.
	    Vector posTestExample =
	        tf.transform(Arrays.asList("kiếm được bộ gấu teddy dễ_thương đã rồi tính".split(" ")));
	    Vector negTestExample =
	        tf.transform(Arrays.asList("cảm thấy bị xúc_phạm".split(" ")));
	    // Now use the learned model to predict spam/ham for new emails.
	    System.out.println("Prediction for positive test example: " + model.predict(posTestExample));
	    System.out.println("Prediction for negative test example: " + model.predict(negTestExample));

	    
	    
	    // Compute raw scores on the test set.
	    JavaRDD<Tuple2<Object, Object>> predictionAndLabels = test.map(
	      new Function<LabeledPoint, Tuple2<Object, Object>>() {
	        /**
			 * 
			 */
			private static final long serialVersionUID = 7362070184439588772L;

			public Tuple2<Object, Object> call(LabeledPoint p) {
	          Double prediction = model.predict(p.features());
	          return new Tuple2<Object, Object>(prediction, p.label());
	        }
	      }
	    );

	    // Get evaluation metrics.
	    MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());

	    // Confusion matrix
	    Matrix confusion = metrics.confusionMatrix();
	    System.out.println("Confusion matrix: \n" + confusion);

	    // Overall statistics
	    System.out.println("Precision = " + metrics.precision());
	    System.out.println("Recall = " + metrics.recall());
	    System.out.println("F1 Score = " + metrics.fMeasure());

	    // Stats by labels
	    for (int i = 0; i < metrics.labels().length; i++) {
	      System.out.format("Class %f precision = %f\n", metrics.labels()[i],metrics.precision(
	        metrics.labels()[i]));
	      System.out.format("Class %f recall = %f\n", metrics.labels()[i], metrics.recall(
	        metrics.labels()[i]));
	      System.out.format("Class %f F1 score = %f\n", metrics.labels()[i], metrics.fMeasure(
	        metrics.labels()[i]));
	    }
	    
	    sc.stop();
	    sc.close();
	}
}
