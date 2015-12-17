package main;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SortedSet;
import java.util.TreeSet;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

public class CreateSVMFile {

	private static JavaSparkContext sc;
	
	public static void main(String[] args) {
		SparkConf sparkConf = new SparkConf().setAppName("JavaBookExample")
				.setMaster("local");
		sc = new JavaSparkContext(sparkConf);
		
		 // 1.) Load the documents
        JavaRDD<String> data = sc.textFile("Data/DATA.txt"); 
        
        // 2.) Load corpus data
        JavaRDD<String> corpusSentiment = sc.textFile("Data/corpus.txt");
        
		/**
		 * Transform data input into a List of VietNamese Words
		 */
		JavaRDD<InputType> corpus = transformInputData(data);

		corpus.cache();

		List<InputType> afterFilterStopword = new ArrayList<InputType>();
		
		/**
		 * In this step, we will remove all of StopWords
		 */
		afterFilterStopword = filterOutStopWord(corpus);
		
		List<String> allContents = new ArrayList<String>();
		for (InputType item : afterFilterStopword) {
			for (String comment : item.getContent().split(" ")) {
				allContents.add(comment);
			}
		}
		
		JavaRDD<String> vocaRDD = sc.parallelize(allContents);
		
		List<Tuple2<String, Long>>  termCounts = vocaRDD.mapToPair(new PairFunction<String, String, Long>() {

			/**
			 * 
			 */
			private static final long serialVersionUID = 1L;

			public Tuple2<String, Long> call(String commentContent) {
				// Check if all of s is UPPER CASE
				String result = "";
				if (commentContent.equals(commentContent.toUpperCase())) {
					result = commentContent;
				} else {
					result = commentContent.toLowerCase();
				}
				return new Tuple2<String, Long>(result, 1L);
			}
			
		}).reduceByKey(new Function2<Long, Long, Long>() {
			private static final long serialVersionUID = 1L;

			public Long call(Long i1, Long i2) {
				return i1 + i2;
			}
		}).collect();
		
		/**
		 * Create a list of Vocabulary
		 */
		int sizeOfVocabulary = termCounts.size();
		List<String> vocabularys = new ArrayList<String>();
		for (int i = 0; i < sizeOfVocabulary; i++) {
			vocabularys.add(termCounts.get(i)._1);
		}

		// TODO
		// get corpus from vietSentiWordNet
		/**
		 * Create a list of Vocabulary and set ID increment from 0 for each word.
		 */
		
		final Map<String, Long> wordAndIndexOfWord = new LinkedHashMap<String, Long>();
		for (Tuple2<String, Long> item : corpusSentiment.zipWithIndex().collect()) {
			wordAndIndexOfWord.put(item._1, (item._2 + 1));
		}

		/**
		 * 
		 */
		Writer writerDATA = null;
		try {
			writerDATA = new BufferedWriter(new OutputStreamWriter(
					new FileOutputStream("Data/featuresData.txt"), "utf-8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		// transform input data to data for SVM
		for (InputType inputType : afterFilterStopword) {

			// stored features of input text
			Map<Long, Integer> wordIDAndWordCount = new HashMap<Long, Integer>(0);
			for (String item : inputType.getContent().split(" ")) {
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
			
			// write data
			String features = "";
			if (keys.size() > 0) {
			    for(Long key : keys) {
				    	features = features + " " + key + ":" + wordIDAndWordCount.get(key);
			    }
			    
				try {
					writerDATA.write(inputType.getLabel() + features + " ");
					writerDATA.write("\n");
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		try {
			writerDATA.flush();
			writerDATA.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
		sc.stop();
		sc.close();
	}
	
	/**
	 * Split each sentences input into a List of VietNamese Words
	 * @param data JavaRDD<String>
	 * @return List of each words
	 */
	private static JavaRDD<InputType> transformInputData(JavaRDD<String> data) {
		/**
		 * Split each sentences input into a List of VietNamese Words
		 */
		JavaRDD<InputType> corpus = data.map(new Function<String,  InputType>() {
			private static final long serialVersionUID = 1L;

			public InputType call(String content) throws Exception {
				String data[] = content.split("\t");
				int id = 0;
				try {
					id = Integer.parseInt(data[0]);
				} catch (Exception e) {
					System.out.println("error");
				}
				return new InputType(id, data[1]);
			}
		});
		System.out.println("Done transform data");
		return corpus;
	}
	
	/**
	 * Remove StopWord
	 * @param inputSentenes
	 * @return
	 */
	private static List<InputType> filterOutStopWord(JavaRDD<InputType> inputSentences){
		
		List<InputType> result = new ArrayList<InputType>();
		
		for (InputType item : inputSentences.collect()) {
			String affterRemoveStopWords = item.getContent();
			InputType tmp = new InputType(item.getLabel(), affterRemoveStopWords);
			result.add(tmp);
		}
		System.out.println("Done filter stopWord");
		return result;
		
	}
}
