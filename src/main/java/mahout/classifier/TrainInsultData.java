package mahout.classifier;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Reader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import mahout.classifier.utils.Utils;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.util.Version;
import org.apache.mahout.classifier.sgd.AdaptiveLogisticRegression;
import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.L2;
import org.apache.mahout.classifier.sgd.ModelSerializer;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.stats.GlobalOnlineAuc;
import org.apache.mahout.math.stats.OnlineAuc;
import org.apache.mahout.vectorizer.encoders.Dictionary;
import org.apache.mahout.vectorizer.encoders.FeatureVectorEncoder;
import org.apache.mahout.vectorizer.encoders.StaticWordValueEncoder;

import com.google.common.base.Charsets;
import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Lists;
import com.google.common.collect.Multiset;
import com.google.common.io.Files;

public class TrainInsultData {
	private static final Analyzer analyzer = new StandardAnalyzer(Version.LUCENE_46);
	private static final FeatureVectorEncoder encoder = new StaticWordValueEncoder("insult");

	public static void main(String[] args) throws IOException {

		int FEATURE = 20000;
		File trainFile = new File("insultdata/train.csv");
		BufferedReader reader = Files.newReader(trainFile, Charsets.UTF_8);
		reader.readLine();
		String line = null;

		// Vector v = new RandomAccessSparseVector(FEATURE);
		List<Vector> vectorList = new ArrayList<Vector>();
		List<Integer> valueList = new ArrayList<Integer>();
		List order = Lists.newArrayList();
		Dictionary dict = new Dictionary();
		dict.intern("0");
		dict.intern("1");
		// Reading and making AUC
		while ((line = reader.readLine()) != null) {

			String[] lines = line.split(",", 3);
			if (lines.length != 3) {
				continue;
			}
			order.add(order.size());
			String insult = lines[2];
			Reader in = new StringReader(insult);
			Multiset<String> words = ConcurrentHashMultiset.create();
			Utils.getWords(analyzer, in, words);
			Vector v = new RandomAccessSparseVector(FEATURE);
			for (String word : words) {
				encoder.addToVector(word, Math.log1p(words.count(word)), v);
			}
			String insultValue = lines[0];
			vectorList.add(v);
			valueList.add(dict.intern(insultValue));
		}
		reader.close();
		Random random = RandomUtils.getRandom();
		// Collections.shuffle(order, random);
		List<Integer> train = order.subList(0, (int) order.size());
		List<Integer> test = order.subList((int) (order.size() * .75), order.size());

		// System.out.println(vectorList.size());
		// System.out.println(order.size());
		// System.out.println(train.size());
		AdaptiveLogisticRegression learningAlgorithm = new AdaptiveLogisticRegression(2, FEATURE, new L2(), 6, 5);
		learningAlgorithm.setInterval(80000);
		learningAlgorithm.setAveragingWindow(1000);

		for (int pass = 0; pass < 100; pass++) {
			// Collections.shuffle(train, random);
			for (int i : train) {
				learningAlgorithm.train(valueList.get(i), vectorList.get(i));
			}
		}

		learningAlgorithm.close();

		String modelFile = "ModelFileSGD-ALR";
		ModelSerializer.writeBinary(modelFile, learningAlgorithm.getBest().getPayload().getLearner().getModels().get(0));
		OnlineLogisticRegression model = ModelSerializer.readBinary(new FileInputStream(modelFile), OnlineLogisticRegression.class);

		// Testing AUC.
		OnlineAuc auc = new GlobalOnlineAuc();
		for (int i : test) {
			double score = model.classifyFull(vectorList.get(i)).get(1);
			auc.addSample(valueList.get(i), score);
		}
		System.out.println(auc.auc());

//		File testFile = new File("insultdata/test.csv");
//		BufferedReader readerTest = Files.newReader(testFile, Charsets.UTF_8);
//		readerTest.readLine();
//		String lineTest = null;
//		PrintWriter writer = new PrintWriter("insultdata/submission");
//		writer.write("Insult,ID\n");
//		while ((lineTest = readerTest.readLine()) != null) {
//
//			String[] linesTest = lineTest.split(",", 2);
//			// if (linesTest.length != 2) {
//			// continue;
//			// }
//
//			String insultTest = linesTest[1];
//			Reader inTest = new StringReader(insultTest);
//			Multiset<String> words = ConcurrentHashMultiset.create();
//			Utils.getWords(analyzer, inTest, words);
//			Vector vTest = new RandomAccessSparseVector(FEATURE);
//			for (String word : words) {
//				// System.out.print(word+" ");
//				encoder.addToVector(word, Math.log1p(words.count(word)), vTest);
//			}
//			// System.out.println();
//			Vector r = model.classifyFull(vTest);
//			// double testValue = model.classifyScalar(vTest);
//			// System.out.println(testValue+","+linesTest[0]+","+r.get(0)+","+r.get(1)+"\n");
//			writer.write(r.get(1) + "," + linesTest[0] + "\n");
//
//		}
//		readerTest.close();
//		writer.flush();
//		writer.close();

	}

}
