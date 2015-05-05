package org.deeplearning4j.word2vec;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.deeplearning4j.plot.Tsne;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.SerializationUtils;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.util.Collection;

/**
 * Created by agibsonccc on 10/9/14.
 */
public class CustomWord2VecExample {


    public static void main(String[] args) throws Exception {
       WeightLookupTable pair = SerializationUtils.readObject(new File(args[0]));



        BarnesHutTsne tsne = new BarnesHutTsne.Builder().setMaxIter(1000).stopLyingIteration(250)
                .learningRate(500).useAdaGrad(false).theta(0.5).setMomentum(0.5)
                .normalize(true).usePca(false).build();

        pair.plotVocab(tsne);




    }

}
