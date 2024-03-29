package mahout.classifier;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.mahout.classifier.sgd.AdaptiveLogisticRegression;
import org.apache.mahout.classifier.sgd.CrossFoldLearner;
import org.apache.mahout.classifier.sgd.ModelDissector;
import org.apache.mahout.classifier.sgd.ModelSerializer;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.ep.State;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.vectorizer.encoders.Dictionary;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Multiset;
public final class SGDHelper {

  private static final String[] LEAK_LABELS = {"none", "month-year", "day-month-year"};

  private SGDHelper() {
  }

  public static void dissect(int leakType,
                             Dictionary dictionary,
                             AdaptiveLogisticRegression learningAlgorithm,
                             Iterable<File> files, Multiset<String> overallCounts) throws IOException {
    CrossFoldLearner model = learningAlgorithm.getBest().getPayload().getLearner();
    model.close();

    Map<String, Set<Integer>> traceDictionary = Maps.newTreeMap();
    ModelDissector md = new ModelDissector();

    NewsgroupHelper helper = new NewsgroupHelper();
    helper.getEncoder().setTraceDictionary(traceDictionary);
    helper.getBias().setTraceDictionary(traceDictionary);

    for (File file : permute(files, helper.getRandom()).subList(0, 500)) {
      String ng = file.getParentFile().getName();
      int actual = dictionary.intern(ng);

      traceDictionary.clear();
      Vector v = helper.encodeFeatureVector(file, actual, leakType, overallCounts);
      md.update(v, traceDictionary, model);
    }

    List<String> ngNames = Lists.newArrayList(dictionary.values());
    List<ModelDissector.Weight> weights = md.summary(100);
    System.out.println("============");
    System.out.println("Model Dissection");
    for (ModelDissector.Weight w : weights) {
      System.out.printf("%s\t%.1f\t%s\t%.1f\t%s\t%.1f\t%s%n",
                        w.getFeature(), w.getWeight(), ngNames.get(w.getMaxImpact() + 1),
                        w.getCategory(1), w.getWeight(1), w.getCategory(2), w.getWeight(2));
    }
  }

  public static List<File> permute(Iterable<File> files, Random rand) {
    List<File> r = Lists.newArrayList();
    for (File file : files) {
      int i = rand.nextInt(r.size() + 1);
      if (i == r.size()) {
        r.add(file);
      } else {
        r.add(r.get(i));
        r.set(i, file);
      }
    }
    return r;
  }

  static void analyzeState(SGDInfo info, int leakType, int k, State<AdaptiveLogisticRegression.Wrapper,
      CrossFoldLearner> best) throws IOException {
    int bump = info.getBumps()[(int) Math.floor(info.getStep()) % info.getBumps().length];
    int scale = (int) Math.pow(10, Math.floor(info.getStep() / info.getBumps().length));
    double maxBeta;
    double nonZeros;
    double positive;
    double norm;

    double lambda = 0;
    double mu = 0;

    if (best != null) {
      CrossFoldLearner state = best.getPayload().getLearner();
      info.setAverageCorrect(state.percentCorrect());
      info.setAverageLL(state.logLikelihood());

      OnlineLogisticRegression model = state.getModels().get(0);
      // finish off pending regularization
      model.close();

      Matrix beta = model.getBeta();
      maxBeta = beta.aggregate(Functions.MAX, Functions.ABS);
      nonZeros = beta.aggregate(Functions.PLUS, new DoubleFunction() {
        @Override
        public double apply(double v) {
          return Math.abs(v) > 1.0e-6 ? 1 : 0;
        }
      });
      positive = beta.aggregate(Functions.PLUS, new DoubleFunction() {
        @Override
        public double apply(double v) {
          return v > 0 ? 1 : 0;
        }
      });
      norm = beta.aggregate(Functions.PLUS, Functions.ABS);

      lambda = best.getMappedParams()[0];
      mu = best.getMappedParams()[1];
    } else {
      maxBeta = 0;
      nonZeros = 0;
      positive = 0;
      norm = 0;
    }
    if (k % (bump * scale) == 0) {
      if (best != null) {
        File modelFile = new File(System.getProperty("java.io.tmpdir"), "news-group-" + k + ".model");
        ModelSerializer.writeBinary(modelFile.getAbsolutePath(), best.getPayload().getLearner().getModels().get(0));
      }

      info.setStep(info.getStep() + 0.25);
      System.out.printf("%.2f\t%.2f\t%.2f\t%.2f\t%.8g\t%.8g\t", maxBeta, nonZeros, positive, norm, lambda, mu);
      System.out.printf("%d\t%.3f\t%.2f\t%s%n",
        k, info.getAverageLL(), info.getAverageCorrect() * 100, LEAK_LABELS[leakType % 3]);
    }
  }

}
