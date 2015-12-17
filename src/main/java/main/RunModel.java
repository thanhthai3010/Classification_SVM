package main;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SortedSet;
import java.util.TreeSet;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;

import scala.Tuple2;

public class RunModel {

	public static void main(String[] args) {
		SparkConf sparkConf = new SparkConf().setAppName("JavaBookExample")
				.setMaster("local");
		SparkContext sc = new SparkContext(sparkConf);
		String path = "Data/featuresData.txt";
	    JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc, path).toJavaRDD().cache();
	    
	     LogisticRegressionModel modelT = null;

	    // Split initial RDD into two... [60% training data, 40% testing data].
	    JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.6, 0.4}, 11L);
	    JavaRDD<LabeledPoint> training = splits[0].cache();
	    JavaRDD<LabeledPoint> test = splits[1];

		// Create a Logistic Regression learner which uses the LBFGS optimizer.
		LogisticRegressionWithLBFGS lrLearner = new LogisticRegressionWithLBFGS();
		// Run the actual learning algorithm on the training data.

		lrLearner.optimizer().setNumIterations(100);
		modelT = lrLearner.setNumClasses(3).run(training.rdd());
		
		final LogisticRegressionModel model = modelT;
		//TODO
		model.setThreshold(0.11);
		
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

	    //Weighted stats
	    System.out.format("Weighted precision = %f\n", metrics.weightedPrecision());
	    System.out.format("Weighted recall = %f\n", metrics.weightedRecall());
	    System.out.format("Weighted F1 score = %f\n", metrics.weightedFMeasure());
	    System.out.format("Weighted false positive rate = %f\n", metrics.weightedFalsePositiveRate());

	    /**********************************************************************************/
	    //predict
	    // 705:1 2521:1 4309:1 4391:1 
	    
        // 2.) Load corpus data
	    JavaSparkContext jSc = new JavaSparkContext(sc);
        JavaRDD<String> corpusSentiment = jSc.textFile("Data/corpus.txt");
	    
		/**
		 * Create a list of Vocabulary and set ID increment from 0 for each word.
		 */
		final Map<String, Long> wordAndIndexOfWord = new LinkedHashMap<String, Long>();
		for (Tuple2<String, Long> item : corpusSentiment.zipWithIndex().collect()) {
			wordAndIndexOfWord.put(item._1, (item._2 + 1));
		}
        
		String needToCheck = "buồn thiệt chớ, không duyên mà cứ gặp";
		
		// stored features of input text
		Map<Long, Integer> wordIDAndWordCount = new HashMap<Long, Integer>(0);
		for (String item : needToCheck.split(" ")) {
			// For each word in listWord of Document input
			if (wordAndIndexOfWord.containsKey(item.toLowerCase())) {
				// Check if vocabAndCount has contain this word
				
				// Get Id of Word.
				Long idxOfWord = wordAndIndexOfWord.get(item.toLowerCase());
				if (!wordIDAndWordCount.containsKey(idxOfWord)) {
					// if this is the first time. Add to HashMap value <idOfWord, 0>
					wordIDAndWordCount.put(idxOfWord, 0);
				}
				// Increase the number of appear
				wordIDAndWordCount.put(idxOfWord, wordIDAndWordCount.get(idxOfWord) + 1);
			}
		}
		
		// SortValue key ascesding
		SortedSet<Long> keys = new TreeSet<Long>(wordIDAndWordCount.keySet());
		
	    int[] key = new int[4244];	 
	    double[] value = new double[4244];
	    int i = 0;
	    for(Long wordID : keys) {
		    key[i] = Integer.parseInt(wordID.toString());
		    value[i] = wordIDAndWordCount.get(wordID);
		    i++;
		}
	    
	    Vector predictResult = Vectors.sparse(4244, key, value);
	    
	    double rs = model.predict(predictResult);
	    System.out.println(rs);
		
	    jSc.stop();
	    jSc.close();
		sc.stop();
	}
}
