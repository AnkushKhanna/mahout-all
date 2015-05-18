package mahout.recommender;
import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.RMSRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.svd.ALSWRFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.FilePersistenceStrategy;
import org.apache.mahout.cf.taste.impl.recommender.svd.ParallelSGDFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.PersistenceStrategy;
import org.apache.mahout.cf.taste.impl.recommender.svd.RatingSGDFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.SVDPlusPlusFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.SVDRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.common.RandomUtils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.Calendar;
public class EvaluatorIntro {
	private EvaluatorIntro() {
	  }

	  public static void main(String[] args) throws Exception {
	    RandomUtils.useTestSeed();

		File modelFile = null;
		if (args.length > 0)
			modelFile = new File(args[0]);
		if(modelFile == null || !modelFile.exists())
			modelFile = new File("train_clean2.csv");
		if(!modelFile.exists()) {
			System.err.println("Please, specify name of file, or put file 'input.csv' into current directory!");
			System.exit(1);
		}
	    DataModel model = new FileDataModel(modelFile);

	    
	    RecommenderEvaluator evaluator =
	      new RMSRecommenderEvaluator();
	    // Build the same recommender for testing that we did last time:
//	    RecommenderBuilder recommenderBuilder = new RecommenderBuilder() {
//	      public Recommender buildRecommender(DataModel model) throws TasteException {
//	       //UserSimilarity similarity = new PearsonCorrelationSimilarity(model);
//	        //UserNeighborhood neighborhood = new NearestNUserNeighborhood(5, similarity, model);
//	    	// return new GenericUserBasedRecommender(model, neighborhood, similarity)
//	        return new SVDRecommender(model, new ALSWRFactorizer(model, 20, 0.05, 1000));
//	      }
//	    };
	    //TODO SGD for i=100 and parallel with the values as specified.
	    // Use 70% of the data to train; test using the other 30%.
	   // double score = evaluator.evaluate(recommenderBuilder, null, model, 0.8, 0.1);
	    long startTime = System.currentTimeMillis();
	    int iteration = 500;
	    int k = 100;
	    PersistenceStrategy persisit = new FilePersistenceStrategy(new File("persistFactorizer_k"+k+"_i_"+iteration+"SGDFactorizer.txt"));
	    Recommender recommender = new SVDRecommender(model, new RatingSGDFactorizer(model, k, iteration), persisit);
	    //Recommender recommender = new SVDRecommender(model, new ParallelSGDFactorizer(model, k, 0.01, iteration), persisit);
	    //Recommender recommender = new SVDRecommender(model, new ALSWRFactorizer(model, k, 0.01, iteration), persisit);
	   
	   
	    
	    long stopTime = System.currentTimeMillis();
	    long time = (stopTime-startTime);
	    System.out.println("TOTAL TIME:::"+time);
	    
	    
	    BufferedReader reader = new BufferedReader(new FileReader("test_clean2.csv"));
	    PrintWriter writer = new PrintWriter("solution.csv");
	    PrintWriter writerItemNotFound = new PrintWriter("itemNotFound.csv");
	    String line = null;
	    writerItemNotFound.write("ID,user,movie");
	    writer.write("ID,rating");
	    while ((line = reader.readLine()) != null) {
	        String[] values = line.split(",");
	        double rating = 3f;
	        try{
	        	rating = recommender.estimatePreference(Long.parseLong(values[1]), Long.parseLong(values[2]));
	    	}catch(NoSuchItemException e){
	    		System.out.println("*********** : "+values[0]);
	    		writerItemNotFound.write("\n");
	    		PreferenceArray userPreferenceArray = model.getPreferencesFromUser(Long.valueOf(values[1]));
	    		float ratingSum = 0;
	    		int count = 0;
	    		for(Preference p : userPreferenceArray){
	    			ratingSum = p.getValue();
	    			count++;
	    		}
	    		rating = ratingSum/count;
	    		writerItemNotFound.write(values[0]+","+values[1]+","+values[2]);
	    		//e.printStackTrace();
	    	}catch(NoSuchUserException e){
	    		System.out.println("*********** : "+values[0]);
	    		//e.printStackTrace();
	    	}
	        writer.write("\n");
	        writer.write(values[0]+","+rating);
	    }
	    reader.close();
	    writer.flush();
	    writer.close();
	    
	    writerItemNotFound.flush();
	    writerItemNotFound.close();
	   //System.out.println(score);
	    
	   
	  }
}
