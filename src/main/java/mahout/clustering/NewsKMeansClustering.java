package mahout.clustering;

import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.lucene.analysis.Analyzer;
import org.apache.mahout.clustering.Cluster;
//import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.Kluster;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.CosineDistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.distance.TanimotoDistanceMeasure;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.measures.CosineSimilarity;
import org.apache.mahout.vectorizer.DictionaryVectorizer;
import org.apache.mahout.vectorizer.DocumentProcessor;
import org.apache.mahout.vectorizer.common.PartialVectorMerger;
import org.apache.mahout.vectorizer.tfidf.TFIDFConverter;

public class NewsKMeansClustering {

	public static void main(String args[]) throws Exception {

		int minSupport = 5;
		int minDf = 1;
		int maxDFPercent = 99;
		int maxNGramSize = 1;
		int minLLRValue = 50;
		int reduceTasks = 1;
		int chunkSize = 200;
		float norm = 2;
		boolean sequentialAccessOutput = true;

		String inputDir = "inputDir";

		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(conf);
		/*
		 * SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, new
		 * Path(inputDir, "documents.seq"), Text.class, Text.class); for
		 * (Document d : Database) { writer.append(new Text(d.getID()), new
		 * Text(d.contents())); } writer.close();
		 */

	String outputDir = "newsClusters";
		HadoopUtil.delete(conf, new Path(outputDir));
		Path tokenizedPath = new Path(outputDir,
				DocumentProcessor.TOKENIZED_DOCUMENT_OUTPUT_FOLDER);
		MyAnalyzer analyzer = new MyAnalyzer();
		DocumentProcessor.tokenizeDocuments(new Path(inputDir), analyzer
				.getClass().asSubclass(Analyzer.class), tokenizedPath, conf);
		analyzer.close();
		DictionaryVectorizer.createTermFrequencyVectors(tokenizedPath,
				new Path(outputDir),
				DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER, conf,
				minSupport, maxNGramSize, minLLRValue, -1, false, reduceTasks,
				chunkSize, sequentialAccessOutput, false);

		Pair<Long[], List<org.apache.hadoop.fs.Path>> df = TFIDFConverter
				.calculateDF(new Path(outputDir,
						DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER),
						new Path(outputDir), conf, chunkSize);

		TFIDFConverter.processTfIdf(new Path(outputDir,
				DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER), new Path(
				outputDir), conf, df, 1, 100, norm, false,
				sequentialAccessOutput, false, reduceTasks);

		Path vectorsFolder = new Path(outputDir, "tfidf-vectors");
		Path canopyCentroids = new Path(outputDir, "canopy-centroids");
		Path clusterOutput = new Path(outputDir, "clusters");

//		SequenceFile.Reader readerVect = new SequenceFile.Reader(fs, new Path(vectorsFolder+"/part-r-00000"), conf);
//		SequenceFile.Writer writerVect = new SequenceFile.Writer(fs, conf, new Path(canopyCentroids+"/part-m-00000"), Text.class,Kluster.class);
//		Text key1 = new Text();
//		VectorWritable value1 = new VectorWritable();
//		int i=0;
//		while (readerVect.next(key1,value1) && i<10) { 
//			Kluster cluster = new Kluster(value1.get(),i,new EuclideanDistanceMeasure());
//			writerVect.append(key1, cluster);
//			i++;
//			System.out.println(i);
//		}
//		readerVect.close();
//		writerVect.close();
//		
		RandomSeedGenerator.buildRandom(conf, vectorsFolder, canopyCentroids, 20, new CosineDistanceMeasure());
		//CanopyDriver.run(vectorsFolder, canopyCentroids,
			//	new CosineDistanceMeasure(), 250, 120, false, 0.5, false);
		KMeansDriver.run(conf, vectorsFolder, canopyCentroids, clusterOutput, 0.01, 20, true, 1.0, false);

		SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path(
				clusterOutput +"/"+ Kluster.CLUSTERED_POINTS_DIR + "/part-m-00000"),
				conf);

		IntWritable key = new IntWritable();
		WeightedPropertyVectorWritable value = new WeightedPropertyVectorWritable();
		while (reader.next(key, value)) {
			 System.out.println(value.toString() + " belongs to cluster "
                     + key.toString());
		}
		reader.close();
	}
}
