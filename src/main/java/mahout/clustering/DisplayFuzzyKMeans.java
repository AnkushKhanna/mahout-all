package mahout.clustering;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.io.IOException;
import java.util.Collection;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.ClusterClassifier;
import org.apache.mahout.clustering.fuzzykmeans.FuzzyKMeansDriver;
import org.apache.mahout.clustering.fuzzykmeans.SoftCluster;
import org.apache.mahout.clustering.iterator.ClusterIterator;
import org.apache.mahout.clustering.iterator.FuzzyKMeansClusteringPolicy;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.math.Vector;

import com.google.common.collect.Lists;

public class DisplayFuzzyKMeans extends DisplayClustering {
  
  DisplayFuzzyKMeans() {
    initialize();
    this.setTitle("Fuzzy k-Means Clusters (>" + (int) (significance * 100) + "% of population)");
  }
  
  // Override the paint() method
  @Override
  public void paint(Graphics g) {
    plotSampleData((Graphics2D) g);
    plotClusters((Graphics2D) g);
  }
  
  public static void main(String[] args) throws Exception {
    DistanceMeasure measure = new ManhattanDistanceMeasure();
    
    Path samples = new Path("inputDir-Display/part-m-00000");
    Path output = new Path("newsClustersFuzzy/clusters/clusteredPoints/part-m-00000");
    Configuration conf = new Configuration();
    HadoopUtil.delete(conf, output);
    HadoopUtil.delete(conf, samples);
    RandomUtils.useTestSeed();
    DisplayClustering.generateSamples();
    writeSampleData(samples);
    boolean runClusterer = true;
    int maxIterations = 10;
    float threshold = 0.001F;
    float m = 1.1F;
    if (runClusterer) {
      runSequentialFuzzyKClusterer(conf, samples, output, measure, maxIterations, m, threshold);
    } else {
      int numClusters = 3;
      runSequentialFuzzyKClassifier(conf, samples, output, measure, numClusters, maxIterations, m, threshold);
    }
    new DisplayFuzzyKMeans();
  }
  
  private static void runSequentialFuzzyKClassifier(Configuration conf, Path samples, Path output,
      DistanceMeasure measure, int numClusters, int maxIterations, float m, double threshold) throws IOException {
    Collection<Vector> points = Lists.newArrayList();
    for (int i = 0; i < numClusters; i++) {
      points.add(SAMPLE_DATA.get(i).get());
    }
    List<Cluster> initialClusters = Lists.newArrayList();
    int id = 0;
    for (Vector point : points) {
      initialClusters.add(new SoftCluster(point, id++, measure));
    }
    ClusterClassifier prior = new ClusterClassifier(initialClusters, new FuzzyKMeansClusteringPolicy(m, threshold));
    Path priorPath = new Path(output, "classifier-0");
    prior.writeToSeqFiles(priorPath);
    
    ClusterIterator.iterateSeq(conf, samples, priorPath, output, maxIterations);
    loadClustersWritable(output);
  }
  
  private static void runSequentialFuzzyKClusterer(Configuration conf, Path samples, Path output,
      DistanceMeasure measure, int maxIterations, float m, double threshold) throws IOException,
      ClassNotFoundException, InterruptedException {
    Path clustersIn = new Path(output, "random-seeds");
    RandomSeedGenerator.buildRandom(conf, samples, clustersIn, 3, measure);
    FuzzyKMeansDriver.run(samples, clustersIn, output, threshold, maxIterations, m, true, true, threshold,
        true);
    
    loadClustersWritable(output);
  }
}
