package mahout.recommender.svd.factorizer;

import java.util.HashMap;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.recommender.svd.Factorization;
import org.apache.mahout.cf.taste.impl.recommender.svd.RatingSGDFactorizer;
import org.apache.mahout.cf.taste.model.DataModel;

public class RatingSGDWithExtralFeatureFactorizer extends RatingSGDFactorizer {

	private HashMap<Long, Integer> userSeenMap = new HashMap<Long, Integer>();

	public RatingSGDWithExtralFeatureFactorizer(DataModel dataModel, int numFeatures, int numIterations) throws TasteException {
		super(dataModel, numFeatures, numIterations);
	}

	protected void updateParameters(long userID, long itemID, float rating, double currentLearningRate) {
		int userIndex = userIndex(userID);
		int itemIndex = itemIndex(itemID);

		double[] userVector = userVectors[userIndex];
		double[] itemVector = itemVectors[itemIndex];
		double prediction = predictRating(userIndex, itemIndex);
		double err = rating - prediction;
		double modifiedLR = currentLearningRate;
		Integer n = userSeenMap.get(userID);
		if (n != null) {
			modifiedLR = currentLearningRate / (2*(Math.sqrt(n)));
			userSeenMap.put(userID, ++n);
		}else{
			userSeenMap.put(userID, 1);
		}

		// adjust user bias
		userVector[USER_BIAS_INDEX] += biasLearningRate * modifiedLR * (err - biasReg * preventOverfitting * userVector[USER_BIAS_INDEX]);

		// adjust item bias
		itemVector[ITEM_BIAS_INDEX] += biasLearningRate * currentLearningRate * (err - biasReg * preventOverfitting * itemVector[ITEM_BIAS_INDEX]);

		// adjust features
		for (int feature = FEATURE_OFFSET; feature < numFeatures; feature++) {
			double userFeature = userVector[feature];
			double itemFeature = itemVector[feature];

			double deltaUserFeature = err * itemFeature - preventOverfitting * userFeature;
			userVector[feature] += modifiedLR * deltaUserFeature;

			double deltaItemFeature = err * userFeature - preventOverfitting * itemFeature;
			itemVector[feature] += currentLearningRate * deltaItemFeature;
		}
	}

	private double predictRating(int userID, int itemID) {
		double sum = 0;
		for (int feature = 0; feature < numFeatures; feature++) {
			sum += userVectors[userID][feature] * itemVectors[itemID][feature];
		}
		return sum;
	}
}
