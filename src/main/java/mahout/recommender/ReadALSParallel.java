package mahout.recommender;

import org.apache.hadoop.conf.Configurable;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.map.OpenIntObjectHashMap;

public class ReadALSParallel {
	private OpenIntObjectHashMap<Vector> U;
	private OpenIntObjectHashMap<Vector> M;
	
	public static void main(String[] args) {
		ReadALSParallel als =new ReadALSParallel();
		
	} 

	public ReadALSParallel() {

		Path pathToU = new Path("oParallelALS/U/");
		Path pathToM = new Path("oParallelALS/M/");
		Configuration conf = new Configuration();
		U = this.readMatrixByRows(pathToU,conf);
		M = this.readMatrixByRows(pathToM, conf);
	}

	private OpenIntObjectHashMap<Vector> readMatrixByRows(Path dir, Configuration conf) {
		OpenIntObjectHashMap<Vector> matrix = new OpenIntObjectHashMap<Vector>();
		for (Pair<IntWritable, VectorWritable> pair : new SequenceFileDirIterable<IntWritable, VectorWritable>(
				dir, PathType.LIST, PathFilters.partFilter(), conf)) {
			int rowIndex = pair.getFirst().get();
			Vector row = pair.getSecond().get();
			matrix.put(rowIndex, row);
		}
		return matrix;
	}

}
