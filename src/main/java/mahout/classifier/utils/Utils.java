package mahout.classifier.utils;

import java.io.IOException;
import java.io.Reader;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.StopFilter;
import org.apache.lucene.analysis.miscellaneous.LengthFilter;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.standard.StandardFilter;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;
import org.apache.mahout.classifier.df.DecisionForest;
import org.apache.mahout.classifier.df.builder.DefaultTreeBuilder;
import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.ref.SequentialBuilder;
import org.apache.mahout.common.RandomUtils;

import com.google.common.collect.Multiset;
import com.google.common.io.Closeables;

public class Utils {
	public static void getWords(Analyzer analyzer, Reader reader, Multiset<String> words) throws IOException {
		// List<String> words = new ArrayList<String>();
		Tokenizer tokenizer = new StandardTokenizer(Version.LUCENE_46, reader);
		TokenStream result = new StandardFilter(Version.LUCENE_46, tokenizer);

		result = new StopFilter(Version.LUCENE_46, result, StandardAnalyzer.STOP_WORDS_SET);
		//result = new LengthFilter(Version.LUCENE_46, result, 2, 50);
		result.addAttribute(CharTermAttribute.class);
		result.reset();
		while (result.incrementToken()) {
			String s = result.getAttribute(CharTermAttribute.class).toString();
			words.add(s);
		}
		result.end();
		Closeables.close(result, true);
		// return words;
	}

	public static DecisionForest buildForest(int numberOfTrees, Data data) {
		int m = (int) Math.floor((Math.log(data.getDataset().nbAttributes())/Math.log(2)) + 1);
		System.out.println(m);
		DefaultTreeBuilder treeBuilder = new DefaultTreeBuilder();
		treeBuilder.setM(m);
		return new SequentialBuilder(RandomUtils.getRandom(), treeBuilder, data.clone()).build(numberOfTrees);
	}
}
