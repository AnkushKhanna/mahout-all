package mahout.clustering;

import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.LowerCaseFilter;
import org.apache.lucene.analysis.core.StopFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.en.PorterStemFilter;
import org.apache.lucene.analysis.miscellaneous.LengthFilter;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.standard.StandardFilter;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;
import org.apache.mahout.math.function.TimesFunction;

public class MyAnalyzer extends Analyzer {

	private final Pattern alphabets = Pattern.compile("[a-z]+");

	@Override
	protected TokenStreamComponents createComponents(String fieldName,
			Reader reader) {
		try {
			Tokenizer tokenizer = new StandardTokenizer(Version.LUCENE_46, reader);
			TokenStream result = new StandardFilter(Version.LUCENE_46, tokenizer);
		    result = new LowerCaseFilter(Version.LUCENE_46, result);
		    result = new StopFilter(Version.LUCENE_46, result, StandardAnalyzer.STOP_WORDS_SET);
		    result = new LengthFilter(Version.LUCENE_46, result,3, 50);
		    result = new PorterStemFilter(result);
		    return new TokenStreamComponents(tokenizer, result);
//			TokenStream result = new StandardTokenizer(Version.LUCENE_46,
//					reader);
//			result = new StandardFilter(Version.LUCENE_46, result);
//			result = new LowerCaseFilter(Version.LUCENE_46, result);
//			result = new StopFilter(Version.LUCENE_46, result,
//					StandardAnalyzer.STOP_WORDS_SET);
//			result = new LengthFilter(Version.LUCENE_46, result,5, 50);
//
//			CharTermAttribute termAtt = result
//					.addAttribute(CharTermAttribute.class);
//			StringBuilder buf = new StringBuilder();
//
//			result.reset();
//			while (result.incrementToken()) {
//				
//				if (termAtt.length() < 3)
//					continue;
//				String word = new String(termAtt.buffer(), 0, termAtt.length());
//				Matcher m = alphabets.matcher(word);
//
//				if (m.matches()) {
//					buf.append(word).append(" ");
//				}
//			}
//			
//			Tokenizer tokenizer = new WhitespaceTokenizer(Version.LUCENE_46, new StringReader(buf.toString()));
//			result.end();
//			result.close();
//			return new TokenStreamComponents(tokenizer);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
		// new WhitespaceTokenizer(Version.LUCENE_46, new
		// StringReader(buf.toString())).;
	}
}