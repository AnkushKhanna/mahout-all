package mahout.recommender;

import java.io.File;

import mahout.recommender.svd.factorizer.RatingSGDWithExtralFeatureFactorizer;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.RMSRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.recommender.svd.FilePersistenceStrategy;
import org.apache.mahout.cf.taste.impl.recommender.svd.PersistenceStrategy;
import org.apache.mahout.cf.taste.impl.recommender.svd.RatingSGDFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.SVDPlusPlusFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.SVDRecommender;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.common.RandomUtils;

public class EvaluateRecommender {
	public static void main(String[] args) throws Exception {
		RandomUtils.useTestSeed();

		int[] iterations = new int[] {100};
		int[] NOFs = new int[] {20};
		for (int NOF : NOFs) {
			for (int i : iterations) { 
				final int iteration = i;
				final int k = NOF;
				final double preventOverfitting = 0.1;
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
				RecommenderBuilder recommenderBuilder = new RecommenderBuilder() {
					public Recommender buildRecommender(DataModel model) throws TasteException {
					//	PersistenceStrategy persisit = new FilePersistenceStrategy(new File("PF_TEST_k" + k + "_i_" + iteration+ "SGDFACT.txt"));
						//return new SVDRecommender(model, new RatingSGDWithExtralFeatureFactorizer(model, k,iteration));
						return new SVDRecommender(model,  new RatingSGDFactorizer(model, k, 0.01, 0.1, 0.01,iteration,1));
					}
				};

				// Use 80% of the data to train; test using the other 20%.
				long startTime = System.currentTimeMillis();
				double score = evaluator.evaluate(recommenderBuilder, null, model, 0.8, 1);
				System.out.println(score);

				long stopTime = System.currentTimeMillis();
				long time = (stopTime - startTime);
				System.out.println(time);
			}
		}
	}
}
