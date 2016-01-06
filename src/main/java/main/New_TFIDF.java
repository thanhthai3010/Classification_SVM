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
        JavaRDD<String> dataFull = sc.textFile("TFIDF_DATA/dataClassifyFull.txt");
        
	    JavaPairRDD<String, Long>  termCounts = dataFull.flatMap(new FlatMapFunction<String, String>() {

			/**
			 * 
			 */
			private static final long serialVersionUID = 1L;

			public Iterable<String> call(String contents) throws Exception {
				String[] values = contents.split("\t");
				String filter = values[1].replaceAll("[0-9]", "");
				return Arrays.asList(filter.split(" "));
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
				if (!Stopwords.isStopword(itemWordCount._1)) {
					return true;
				} else {
					return false;
				}
			}
		});
		
		int sizeOfVocabulary = afterFilter.collect().size();
		
		System.out.println("sizeOfVocabulary " + sizeOfVocabulary);
		
		/**
		 * Union positive and negative to get full data
		 */
        // 2.) Hash all documents
        HashingTF hashingTF = new HashingTF(sizeOfVocabulary);
        JavaRDD<LabeledPoint> tupleData = dataFull.map(content -> {
                String[] datas = content.split("\t");
                
                String filter = datas[1].replaceAll("[0-9]", " ");
                
                List<String> myList = Arrays.asList(Stopwords.removeStopWords(filter).split(" "));
                return new LabeledPoint(Double.parseDouble(datas[0]), hashingTF.transform(myList));
        });
        // 3.) Create a flat RDD with all vectors
        JavaRDD<Vector> hashedData = tupleData.map(label -> label.features());
        // 4.) Create a IDFModel out of our flat vector RDD
        IDFModel idfModel = new IDF(2).fit(hashedData);
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
	    
//	    model.clearThreshold();
	    /**
	     * need to get vector
	     */
	    // Test on a positive example and a negative one.
	    // First apply the same HashingTF feature transformation used on the training data.
	    
	    String pos = Stopwords.removeStopWords("bật_cười ủa . Hình_như cũng do qđiểm thôi . Mình học TE ở trường từ TE 1 đến TE5 rồi bạn . Sinh_viên rồi , qtrọng là ở tự học thôi . Bản_thân có muốn học hay ko . Chứ còn với những người làm_biếng thì_có giáo_trình sao đi_nữa cũng vẫn vậy . Rồi có chắc giao bài về nha là làm ko bật_cười hay lại than trời ông nọ bà kia khó . Bản_thân thấy học TE ở trường giúp làm_quen rất nhiều vs các từ chuyên_ngành . Ko có bh là đủ để học av . Ko có thể_nào mà đi hết trung_tâm này trung_tâm khác là bạn có_thể giỏi đc . Cái cốt_lõi vẫn là bản_thân thôi . Lớn rồi tữ vận_động đi , đừng trách_móc nữa . Thêm nữa , toeic bằng đầu_ra điểm nv là quá dễ_dàng rồi . Lấy cái bằng đó chưa chắc là đủ cạnh_tranh khi ra trường đâu . Thân .".toLowerCase());
	    
		Vector posTestExample = idfModel.transform(hashingTF.transform(Arrays
				.asList(pos.split(" "))));
		
		String neg = Stopwords.removeStopWords("Mọi người cho em hỏi cái vụ điểm liệt môn toán cao_cấp là có thật ko ? Thi dưới 2đ có bị liệt và rớt môn ko ? sợ quá"
				.toLowerCase());
		
		System.out.println(neg);
		
		Vector negTestExample = idfModel.transform(hashingTF.transform(Arrays
				.asList(neg.split(" "))));
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
