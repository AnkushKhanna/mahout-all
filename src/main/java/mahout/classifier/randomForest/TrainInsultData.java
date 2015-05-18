package mahout.classifier.randomForest;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import mahout.classifier.utils.Utils;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.util.Version;
import org.apache.mahout.classifier.df.DecisionForest;
import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.data.DataLoader;
import org.apache.mahout.classifier.df.data.DescriptorException;
import org.apache.mahout.classifier.df.data.Instance;
import org.apache.mahout.classifier.sgd.AdaptiveLogisticRegression;
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

	public static void main(String[] args) throws IOException, DescriptorException {

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
		List<Integer> train = order.subList(0, (int) (order.size() * .75));
		List<Integer> test = order.subList((int) (order.size() * .75), order.size());
		String[] trainDataValues = new String[train.size()];

		StringBuilder desc = new StringBuilder();
		desc.append("L ");
		for (int f = 0; f < FEATURE; f++) {
			desc.append("N");
			if (f != FEATURE - 1) {
				desc.append(" ");
			}
		}

		for (int i : train) {
			StringBuilder sb = new StringBuilder();
			Vector v = vectorList.get(i);
			if(valueList.get(i)!=0 && valueList.get(i)!=1){
				System.out.println("SAY SOMETHING GOES WRONG::"+valueList.get(i));
			}
			sb.append(valueList.get(i) + " ");
			for (int f = 0; f < FEATURE; f++) {
				sb.append(v.get(f));
				if (f != FEATURE - 1) {
					sb.append(" ");
				}
			}
			trainDataValues[i] = sb.toString();
		}

		Data data = DataLoader.loadData(DataLoader.generateDataset(desc.toString(), false, trainDataValues), trainDataValues);
		int numberOfTrees = 85;
		DecisionForest forest = Utils.buildForest(numberOfTrees, data);

		// Testing AUC.
		OnlineAuc auc = new GlobalOnlineAuc();
		String[] testDataValues = new String[test.size()];
		List<Integer> testValue = new ArrayList<Integer>();
		int count = 0;
		for (int i : test) {
			StringBuilder sb = new StringBuilder();
			sb.append("1 ");
			Vector v = vectorList.get(i);
			for (int f = 0; f < FEATURE; f++) {
				sb.append(v.get(f));
				if (f != FEATURE - 1) {
					sb.append(" ");
				}
			}

			testDataValues[count] = sb.toString();
			count++;
			testValue.add(valueList.get(i));
		}

		
		System.out.println("PASSED THIS");
		Data testData = DataLoader.loadData(data.getDataset(), testDataValues);
		Random rng = RandomUtils.getRandom();
		for (int i = 0; i < testData.size(); i++) {
			Instance oneSample = testData.get(i);
			double classify = forest.classify(testData.getDataset(), rng, oneSample);
			System.out.println(classify);
			auc.addSample(testValue.get(i), classify);
		}
		System.out.println(auc.auc());

		// File testFile = new File("insultdata/test.csv");
		// BufferedReader readerTest = Files.newReader(testFile,
		// Charsets.UTF_8);
		// readerTest.readLine();
		// String lineTest = null;
		// PrintWriter writer = new PrintWriter("insultdata/submission");
		// writer.write("Insult,ID\n");
		// while ((lineTest = readerTest.readLine()) != null) {
		//
		// String[] linesTest = lineTest.split(",", 2);
		// // if (linesTest.length != 2) {
		// // continue;
		// // }
		//
		// String insultTest = linesTest[1];
		// Reader inTest = new StringReader(insultTest);
		// Multiset<String> words = ConcurrentHashMultiset.create();
		// Utils.getWords(analyzer, inTest, words);
		// Vector vTest = new RandomAccessSparseVector(FEATURE);
		// for (String word : words) {
		// // System.out.print(word+" ");
		// encoder.addToVector(word, Math.log1p(words.count(word)), vTest);
		// }
		// // System.out.println();
		// Vector r = model.classifyFull(vTest);
		// // double testValue = model.classifyScalar(vTest);
		// //
		// System.out.println(testValue+","+linesTest[0]+","+r.get(0)+","+r.get(1)+"\n");
		// writer.write(r.get(1) + "," + linesTest[0] + "\n");
		//
		// }
		// readerTest.close();
		// writer.flush();
		// writer.close();

	}
}
