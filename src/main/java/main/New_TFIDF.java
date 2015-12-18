package main;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.feature.IDF;
import org.apache.spark.mllib.feature.IDFModel;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;

import scala.Tuple2;

public class New_TFIDF {

	@SuppressWarnings("resource")
	public static void main(String[] args) {
		SparkConf sparkConf = new SparkConf().setAppName("TF-IDF").setMaster("local");
	    JavaSparkContext sc = new JavaSparkContext(sparkConf);	    

        // 1.) Load the documents
        JavaRDD<String> dataFull = sc.textFile("TFIDF_DATA/dataFull.txt"); 
	    
	    JavaPairRDD<String, Long>  termCounts = dataFull.flatMap(new FlatMapFunction<String, String>() {

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
		
		int sizeOfVocabulary = afterFilter.collect().size();
		
		/**
		 * Union positive and negative to get full data
		 */
        // 2.) Hash all documents
        HashingTF tf = new HashingTF(sizeOfVocabulary);
        JavaRDD<LabeledPoint> tupleData = dataFull.map(content -> {
                String[] datas = content.split("\t");
                List<String> myList = Arrays.asList(datas[1].split(" "));
                return new LabeledPoint(Double.parseDouble(datas[0]), tf.transform(myList));
        }); 
        // 3.) Create a flat RDD with all vectors
        JavaRDD<Vector> hashedData = tupleData.map(label -> label.features());
        // 4.) Create a IDFModel out of our flat vector RDD
        IDFModel idfModel = new IDF().fit(hashedData);
        // 5.) Create tfidf RDD
        JavaRDD<Vector> idf = idfModel.transform(hashedData);
        
        // 6.) Create Labledpoint RDD
        JavaRDD<LabeledPoint> dataAfterTFIDF = idf.zip(tupleData).map(t -> {
            return new LabeledPoint(t._2.label(), t._1);
        });

	    // random splits data for training and testing
	    JavaRDD<LabeledPoint>[] splits = dataAfterTFIDF.randomSplit(new double[]{0.6, 0.4}, 11L);
	    JavaRDD<LabeledPoint> training = splits[0].cache();
	    JavaRDD<LabeledPoint> test = splits[1].cache();
	    
	    // Create a Logistic Regression learner which uses the LBFGS optimizer.
	    LogisticRegressionWithLBFGS lrLearner = new LogisticRegressionWithLBFGS();
	    // Run the actual learning algorithm on the training data.

	    lrLearner.optimizer().setNumIterations(100);
	    final LogisticRegressionModel model = lrLearner.setNumClasses(2).run(training.rdd());
	    
	    /**
	     * need to get vector
	     */
	    // Test on a positive example and a negative one.
	    // First apply the same HashingTF feature transformation used on the training data.
	    
		Vector posTestExample = idfModel.transform(tf.transform(Arrays
				.asList("người miền trung không liên_quan ngại_ngùng"
						.split(" "))));
		Vector negTestExample = idfModel.transform(tf.transform(Arrays
				.asList("vui_vẻ".split(" "))));
		// Now use the learned model to predict positive/negative for new
		// comments.
		System.out.println("Prediction for positive test example: "
				+ model.predict(posTestExample));
		System.out.println("Prediction for negative test example: "
				+ model.predict(negTestExample));
	    
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
	
	/**
	 * Create LabeledPoint data for positive and negative data
	 * @param tf
	 * @param data
	 * @param labelPoint
	 * @return a LabeledPoint
	 */
	public static JavaRDD<LabeledPoint> getLabeldPointData(HashingTF tf,
			JavaRDD<String> data, double labelPoint) {
		
		// 1.)Create LabeledPoint from input data
        JavaRDD<LabeledPoint> tupleData = data.map(content -> {
                List<String> myList = Arrays.asList(content.split(" "));
                return new LabeledPoint(labelPoint, tf.transform(myList));
        }); 
        // 2.) Create a flat RDD with all vectors
        JavaRDD<Vector> hashedData = tupleData.map(label -> label.features());
        // 3.) Create a IDFModel out of our flat vector RDD
        IDFModel idfModel = new IDF().fit(hashedData);
        // 4.) Create tfidf RDD
        JavaRDD<Vector> idf = idfModel.transform(hashedData);
        // 5.) Create Labledpoint RDD
        JavaRDD<LabeledPoint> idfTransformed = idf.zip(tupleData).map(t -> {
            return new LabeledPoint(t._2.label(), t._1);
        });
        
		return idfTransformed;
	}
	
//	public Vector getVector(HashingTF tf){
//		Vector hash = tf.transform(Arrays.asList("thích_thú".split(" ")));
//		JavaRDD<Vector> hashedData = hash.rd
//		return null;
//	}
}
