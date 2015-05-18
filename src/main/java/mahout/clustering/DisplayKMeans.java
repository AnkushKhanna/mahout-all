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
import org.apache.mahout.clustering.iterator.ClusterIterator;
import org.apache.mahout.clustering.iterator.KMeansClusteringPolicy;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.math.Vector;

import com.google.common.collect.Lists;
public class DisplayKMeans extends DisplayClustering {
	  
	  DisplayKMeans() {
	    initialize();
	    this.setTitle("k-Means Clusters (>" + (int) (significance * 100) + "% of population)");
	  }
	  
	  public static void main(String[] args) throws Exception {
	    DistanceMeasure measure = new ManhattanDistanceMeasure();
	    Path samples = new Path("inputDir-Display/part-m-00000");
	    Path output = new Path("newsClusters/clusters/clusteredPoints/part-m-00000");
	    Configuration conf = new Configuration();
	    HadoopUtil.delete(conf, samples);
	    HadoopUtil.delete(conf, output);
	    
	    RandomUtils.useTestSeed();
	    generateSamples();
	    writeSampleData(samples);
	    boolean runClusterer = false;
	    double convergenceDelta = 0.001;
	    int numClusters = 3;
	    int maxIterations = 10;
	    if (runClusterer) {
	      runSequentialKMeansClusterer(conf, samples, output, measure, numClusters, maxIterations, convergenceDelta);
	    } else {
	      runSequentialKMeansClassifier(conf, samples, output, measure, numClusters, maxIterations, convergenceDelta);
	    }
	    new DisplayKMeans();
	  }
	  
	  private static void runSequentialKMeansClassifier(Configuration conf, Path samples, Path output,
	      DistanceMeasure measure, int numClusters, int maxIterations, double convergenceDelta) throws IOException {
	    Collection<Vector> points = Lists.newArrayList();
	    for (int i = 0; i < numClusters; i++) {
	      points.add(SAMPLE_DATA.get(i).get());
	    }
	    List<Cluster> initialClusters = Lists.newArrayList();
	    int id = 0;
	    for (Vector point : points) {
	      initialClusters.add(new org.apache.mahout.clustering.kmeans.Kluster(point, id++, measure));
	    }
	    ClusterClassifier prior = new ClusterClassifier(initialClusters, new KMeansClusteringPolicy(convergenceDelta));
	    Path priorPath = new Path(output, Cluster.INITIAL_CLUSTERS_DIR);
	    prior.writeToSeqFiles(priorPath);
	    
	    ClusterIterator.iterateSeq(conf, samples, priorPath, output, maxIterations);
	    loadClustersWritable(output);
	  }
	  
	  private static void runSequentialKMeansClusterer(Configuration conf, Path samples, Path output,
	    DistanceMeasure measure, int numClusters, int maxIterations, double convergenceDelta)
	    throws IOException, InterruptedException, ClassNotFoundException {
	    Path clustersIn = new Path(output, "random-seeds");
	    RandomSeedGenerator.buildRandom(conf, samples, clustersIn, numClusters, measure);
	    KMeansDriver.run(samples, clustersIn, output, convergenceDelta, maxIterations, true, 0.0, true);
	    loadClustersWritable(output);
	  }
	  
	  // Override the paint() method
	  @Override
	  public void paint(Graphics g) {
	    plotSampleData((Graphics2D) g);
	    plotClusters((Graphics2D) g);
	  }
	}

