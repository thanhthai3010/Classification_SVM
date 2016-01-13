package main;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
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
        JavaRDD<String> dataFull = sc.textFile("TFIDF_DATA/dataClassifyFull_New.txt").cache();
        
		/**
		 * Union positive and negative to get full data
		 */
        // 2.) Hash all documents
        HashingTF hashingTF = new HashingTF(40000);
        JavaRDD<LabeledPoint> tupleData = dataFull.map(content -> {
                String[] datas = content.split("\t");
                
                String filter = datas[1].replaceAll("[0-9]", " ");
                
                List<String> myList = Arrays.asList(Stopwords.removeStopWords(filter.toLowerCase()).split(" "));
                return new LabeledPoint(Double.parseDouble(datas[0]), hashingTF.transform(myList));
        });
        // 3.) Create a flat RDD with all vectors
        JavaRDD<Vector> hashedData = tupleData.map(label -> label.features());
        hashedData.saveAsTextFile("hashedData");
        // 4.) Create a IDFModel out of our flat vector RDD
        IDFModel idfModel = new IDF(5).fit(hashedData);
        // 5.) Create tfidf RDD
        JavaRDD<Vector> idf = idfModel.transform(hashedData);
//        idf.saveAsTextFile("idfModel");
        
        // 6.) Create Labledpoint RDD
        JavaRDD<LabeledPoint> dataAfterTFIDF = idf.zip(tupleData).map(t -> {
            return new LabeledPoint(t._2.label(), t._1);
        });
//        dataAfterTFIDF.saveAsTextFile("dataAfterTFIDF");

	    // random splits data for training and testing
	    JavaRDD<LabeledPoint>[] splits = dataAfterTFIDF.randomSplit(new double[]{0.6, 0.4}, 11L);
	    JavaRDD<LabeledPoint> training = splits[0].cache();
	    JavaRDD<LabeledPoint> test = splits[1].cache();
	    
	    // Create a Logistic Regression learner which uses the LBFGS optimizer.
	    LogisticRegressionWithLBFGS lrLearner = new LogisticRegressionWithLBFGS();
	    // Run the actual learning algorithm on the training data.

	    lrLearner.optimizer().setNumIterations(100);
	    final LogisticRegressionModel model = lrLearner.setNumClasses(2).run(training.rdd());
	    
	    Set<String> listSet = new HashSet<String>(sc.textFile("TFIDF_DATA/listInput.txt").collect());
	    
	    System.out.println(listSet.size());
	    
	    for (String item : listSet) {
		    if (item.length() >= 2 ) {
				Vector posTestExample = idfModel.transform(hashingTF.transform(Arrays
						.asList(item)));
				System.out.println(item + posTestExample.toString());
				System.out.println("result: " + model.predict(posTestExample));
			}
		}
	    
		String in = "có_lẽ_nào thích bạn_thân suốt ba đặc_biệt trở thành_thân thi đh phượt nộp hồ_sơ rút hồ_sơ bla bla hắn chăm troll ghét kinh pha_trò đh khối đh qg hắn chở mặc_dù học mặt ... tự_nhiên kể mẹ đh chả nhắc hắn mẹ thích hả chối lay đẹp khùng vui_tươi trả_lời mẹ câu mẹ hắn it chính_hiệu tốt nỗi bạn_bè toan_tính khá chắn đặc_biệt giúp_đỡ con_gái điên_điên hắn chỡ mệt cạy họng cảm_ơn lỗi mắt hắn tốt trừ troll nhát ma vấn_đề hắn xem hắn bạn_gái tự suy_diễn lung_tung hi_vọng khùng tạm_thời cảm_giác tên tốt tốt chết điên mối tình vắt vai bắt đơn_phương phá_hoại gia can đau khó_chịu hự bực_mình nhăn_nhó tên bạn_thân đáng ghét ";
		in = Stopwords.removeStopWords(in);

		List<String> myList = Arrays.asList(in.split(" "));
		
		Vector posTestExample = idfModel.transform(hashingTF.transform(myList));
		System.out.println(in + posTestExample.toString());
		System.out.println("result: " + model.predict(posTestExample));

	    
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
	    //metrics.

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
