package mahout.recommender.overfit;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;

import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.RMSRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.recommender.svd.FilePersistenceStrategy;
import org.apache.mahout.cf.taste.impl.recommender.svd.PersistenceStrategy;
import org.apache.mahout.cf.taste.impl.recommender.svd.RatingSGDFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.SVDRecommender;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.common.RandomUtils;

public class CheckOverfit {
	public static void main(String[] args) throws Exception {
		RandomUtils.useTestSeed();

		int[] iterations = new int[] { 100 };
		int[] NOFs = new int[] { 20 };
		for (int NOF : NOFs) {
			for (int i : iterations) {
				final int iteration = i;
				final int k = NOF;
				final double preventOverfitting = 0.01;
				System.out.println("ITERATIONS::" + iteration);
				System.out.println("K::" + k);
				File modelFile = null;
				if (modelFile == null || !modelFile.exists())
					modelFile = new File("train_clean2.csv");
				if (!modelFile.exists()) {
					System.err.println("Please, specify name of file, or put file 'input.csv' into current directory!");
					System.exit(1);
				}
				DataModel model = new FileDataModel(modelFile);

				RecommenderEvaluator evaluator = new RMSRecommenderEvaluator();
				// Build the same recommender for testing that we did last time:

				//PersistenceStrategy persisit = new FilePersistenceStrategy(new File("PF_TEST_k" + k + "_i_" + iteration + "RatingSGDFactorizer.txt"));
				PersistenceStrategy persisit = new FilePersistenceStrategy(new File("PF_TEST_k" + k + "_i_" + iteration +"_o_"+preventOverfitting+ "RatingSGDFactorizer.txt"));
				SVDRecommender recommender = new SVDRecommender(model, new RatingSGDFactorizer(model, k, iteration), persisit);

				// Use 80% of the data to train; test using the other 20%.
				BufferedReader reader = new BufferedReader(new FileReader("train_clean2.csv"));

				String line = null;
				double score = 0;
				double R = 0;
				double predScoreTotal = 0;
				double scoreTotal = 0;
				while ((line = reader.readLine()) != null && R < 2000) {
					String[] values = line.split(",");
					double rating = 3f;
					try {
						rating = recommender.estimatePreference(Long.parseLong(values[0]), Long.parseLong(values[1]));
					} catch (NoSuchItemException e) {
						System.out.println("*********** : " + values[0]);

						PreferenceArray userPreferenceArray = model.getPreferencesFromUser(Long.valueOf(values[0]));
						float ratingSum = 0;
						int count = 0;
						for (Preference p : userPreferenceArray) {
							ratingSum = p.getValue();
							count++;
						}
						rating = ratingSum / count;

						// e.printStackTrace();
					} catch (NoSuchUserException e) {
						System.out.println("*********** : " + values[0]);
						// e.printStackTrace();
					}
					predScoreTotal+= rating;
					scoreTotal+= Double.parseDouble(values[2]);
					score += Math.pow(rating - Double.parseDouble(values[2]), 2);
					R++;
				}
				score = Math.sqrt(score) / R;
				System.out.println(scoreTotal+"   "+predScoreTotal);
				reader.close();

				System.out.println(score);

			}
		}
	}
}
