package mahout.classifier;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Reader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.StopFilter;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.standard.StandardFilter;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;
import org.apache.mahout.classifier.sgd.CrossFoldLearner;
import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.encoders.Dictionary;
import org.apache.mahout.vectorizer.encoders.FeatureVectorEncoder;
import org.apache.mahout.vectorizer.encoders.StaticWordValueEncoder;

import com.google.common.base.Charsets;
import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Lists;
import com.google.common.collect.Multiset;
import com.google.common.io.Closeables;
import com.google.common.io.Files;

public class InsultDataCF {
	private static final Analyzer analyzer = new StandardAnalyzer(Version.LUCENE_46);
	private static final FeatureVectorEncoder encoder = new StaticWordValueEncoder("insult");

	public static void main(String[] args) throws IOException {

		int FEATURE = 10000;
		File trainFile = new File("insultdata/train.csv");
		BufferedReader reader = Files.newReader(trainFile, Charsets.UTF_8);
		reader.readLine();
		String line = null;

		// Vector v = new RandomAccessSparseVector(FEATURE);
		List<Vector> vectorList = new ArrayList<Vector>();
		List<Integer> valueList = new ArrayList<Integer>();
		List order = Lists.newArrayList();
		 Dictionary dict = new Dictionary();

		while ((line = reader.readLine()) != null) {

			String[] lines = line.split(",",3);
			if (lines.length != 3) {
				continue;
			}
			order.add(order.size());
			String insult = lines[2];
			Reader in = new StringReader(insult);
			Multiset<String> words = ConcurrentHashMultiset.create();
			getWords(analyzer, in, words);
			Vector v = new RandomAccessSparseVector(FEATURE);
			for (String word : words) {
			//	System.out.print(word + "  ");
				encoder.addToVector(word, Math.log1p(words.count(word)), v);
			}
			//System.out.println();
			String insultValue = lines[0];
			//System.out.println(insultValue);
			vectorList.add(v);
			valueList.add(dict.intern(insultValue));
		}
		Random random = RandomUtils.getRandom();
		// Collections.shuffle(order, random);
		List<Integer> train = order.subList(0, (int) (order.size()));
		 List<Integer> test = order.subList((int) (order.size() * .75),order.size());

		System.out.println(vectorList.size());
		System.out.println(order.size());
		System.out.println(train.size());
		// System.out.println(test.size());

		// int[] correct = new int[test.size() + 1];

		// for (int run = 0; run < 1; run++) {
		OnlineLogisticRegression learningAlgorithm = new OnlineLogisticRegression(2, FEATURE, new L1());
		//learningAlgorithm.setInterval(800);
		//learningAlgorithm.setAveragingWindow(500);
		CrossFoldLearner crossFL = new CrossFoldLearner(4, 2, FEATURE, new L1());
		crossFL.addModel(learningAlgorithm);
		for (int pass = 0; pass < 30; pass++) {
			Collections.shuffle(train, random);
			for (int i : train) {
				crossFL.train(valueList.get(i), vectorList.get(i));
			}
		}
		//System.out.println("AUC=" + learningAlgorithm.getBest().getPayload().getLearner().auc() + ", %-correct=" + learningAlgorithm.getBest().getPayload().getLearner().percentCorrect());
		crossFL.close();
		
		int x = 0;
		String modelFile = "ModelFileSGD-ALR";
		//ModelSerializer.writeBinary(modelFile, learningAlgorithm.getBest().getPayload().getLearner().getModels().get(0));
		//OnlineLogisticRegression model = ModelSerializer.readBinary(new FileInputStream(modelFile), OnlineLogisticRegression.class);

//		 for (Integer k : test) {
//			 Vector r = model.classifyFull(vectorList.get(k));
////			if(r>0.5){
////				r=1;
////			}else{
////				r = 0;
////			}
////			
////			 x += r == valueList.get(k) ? 1 : 0;
//			 
//			 System.out.println(r.size()+"  "+r.get(0)+"  "+r.get(1)+"   "+model.classifyScalar(vectorList.get(k)));
//		 }
//		 float y = Float.valueOf(x)/401;
//		 System.out.println(x+"   "+y);
//			
//		 System.out.println();
		reader.close();

		// String modelFile = "ModelFileSGD-ALR";
		// ModelSerializer.writeBinary(modelFile,
		// learningAlgorithm.getBest().getPayload().getLearner().getModels().get(0));
		// OnlineLogisticRegression model = ModelSerializer.readBinary(new
		// FileInputStream(modelFile), OnlineLogisticRegression.class);

		File testFile = new File("insultdata/test.csv");
		BufferedReader readerTest = Files.newReader(testFile, Charsets.UTF_8);
		readerTest.readLine();
		String lineTest = null;
		PrintWriter writer = new PrintWriter("insultdata/submissionCF");
		writer.write("Insult,ID\n");
		while ((lineTest = readerTest.readLine()) != null) {
			
			String[] linesTest = lineTest.split(",",2);
//			if (linesTest.length != 2) {
//				continue;
//			}

			String insultTest = linesTest[1];
			Reader inTest = new StringReader(insultTest);
			Multiset<String> words = ConcurrentHashMultiset.create();
			getWords(analyzer, inTest, words);
			Vector vTest = new RandomAccessSparseVector(FEATURE);
			for (String word : words) {
				System.out.print(word+" ");
				encoder.addToVector(word,Math.log1p(words.count(word)), vTest);
			}
			System.out.println();
			 Vector r = crossFL.classifyFull(vTest);
			double testValue = crossFL.classifyScalar(vTest);
			System.out.println(testValue+","+linesTest[0]+","+r.get(0)+","+r.get(1)+"\n");
			writer.write(r.get(0)+","+linesTest[0]+"\n");
			
		}
		readerTest.close();
		writer.flush();
		writer.close();
		
	}

	public static void getWords(Analyzer analyzer, Reader reader, Multiset<String> words) throws IOException {
		// List<String> words = new ArrayList<String>();
		Tokenizer tokenizer = new StandardTokenizer(Version.LUCENE_46, reader);
		TokenStream result = new StandardFilter(Version.LUCENE_46, tokenizer);
		
		result = new StopFilter(Version.LUCENE_46, result, StandardAnalyzer.STOP_WORDS_SET);
		//ts = new LengthFilter(Version.LUCENE_46, ts, 3, 50);
		result.addAttribute(CharTermAttribute.class);
		result.reset();
		while (result.incrementToken()) {
			String s = result.getAttribute(CharTermAttribute.class).toString();
			words.add(s);
		}
		result.end();
		Closeables.close(result, true);
		// return words;
	}
}
