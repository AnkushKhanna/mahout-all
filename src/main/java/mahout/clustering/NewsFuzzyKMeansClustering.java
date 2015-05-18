package mahout.clustering;

import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.lucene.analysis.Analyzer;
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.clustering.fuzzykmeans.FuzzyKMeansDriver;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.Kluster;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.vectorizer.DictionaryVectorizer;
import org.apache.mahout.vectorizer.DocumentProcessor;
import org.apache.mahout.vectorizer.tfidf.TFIDFConverter;

public class NewsFuzzyKMeansClustering {
	public static void main(String args[]) throws Exception {

		int minSupport = 5;
		int minDf = 5;
		int maxDFPercent = 5;
		int maxNGramSize = 1;
		int minLLRValue = 5;
		int reduceTasks = 1;
		int chunkSize = 200;
		int norm = 2;
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

	String outputDir = "newsClustersFuzzy";
		HadoopUtil.delete(conf, new Path(outputDir));
		Path tokenizedPath = new Path(outputDir,
				DocumentProcessor.TOKENIZED_DOCUMENT_OUTPUT_FOLDER);
		MyAnalyzer analyzer = new MyAnalyzer();
		DocumentProcessor.tokenizeDocuments(new Path(inputDir), analyzer
				.getClass().asSubclass(Analyzer.class), tokenizedPath, conf);

		DictionaryVectorizer.createTermFrequencyVectors(tokenizedPath,
				new Path(outputDir),
				DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER, conf,
				minSupport, maxNGramSize, minLLRValue, 2, true, reduceTasks,
				chunkSize, sequentialAccessOutput, false);

		Pair<Long[], List<org.apache.hadoop.fs.Path>> df = TFIDFConverter
				.calculateDF(new Path(outputDir,
						DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER),
						new Path(outputDir), conf, chunkSize);

		TFIDFConverter.processTfIdf(new Path(outputDir,
				DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER), new Path(
				outputDir), conf, df, minDf, maxDFPercent, norm, true,
				sequentialAccessOutput, false, reduceTasks);

		Path vectorsFolder = new Path(outputDir, "tfidf-vectors");
		Path canopyCentroids = new Path(outputDir, "canopy-centroids");
		Path clusterOutput = new Path(outputDir, "clusters");

		CanopyDriver.run(vectorsFolder, canopyCentroids,
				new EuclideanDistanceMeasure(), 250, 120, false, 0.0, false);
		FuzzyKMeansDriver.run(conf, vectorsFolder, new Path(canopyCentroids,
				"clusters-0-final"), clusterOutput, 0.01, 20, 2, true, true, 0.5, sequentialAccessOutput);

		SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path(
				clusterOutput +"/"+ Kluster.CLUSTERED_POINTS_DIR + "/part-m-0"),
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
