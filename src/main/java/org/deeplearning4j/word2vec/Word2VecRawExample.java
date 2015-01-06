package org.deeplearning4j.word2vec;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.plot.Tsne;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.util.Collection;

/**
 * Created by agibsonccc on 10/9/14.
 */
public class Word2VecRawExample {


    public static void main(String[] args) throws Exception {
        ClassPathResource resource = new ClassPathResource("raw_sentences.txt");
        SentenceIterator iter = new LineSentenceIterator(resource.getFile());
        iter.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                return sentence.toLowerCase();
            }
        });

        //DocumentIterator iter = new FileDocumentIterator(resource.getFile());
        TokenizerFactory t = new DefaultTokenizerFactory();
        final EndingPreProcessor preProcessor = new EndingPreProcessor();
        t.setTokenPreProcessor(new TokenPreProcess() {
            @Override
            public String preProcess(String token) {
                token = token.toLowerCase();
                String base = preProcessor.preProcess(token);
                base = base.replaceAll("\\d","d");
                if(base.endsWith("ly") || base.endsWith("ing"))
                    System.out.println();
                return base;
            }
        });

        int layerSize = 300;

        Word2Vec vec = new Word2Vec.Builder().sampling(1e-5)
                .minWordFrequency(5).batchSize(1000).useAdaGrad(false).layerSize(layerSize)
                .iterations(3).learningRate(0.025).minLearningRate(1e-2).negativeSample(10)
                .iterate(iter).tokenizerFactory(t).build();
        vec.fit();


        double sim = vec.similarity("day","night");
        Collection<String> similar = vec.wordsNearest("day",20);
        System.out.println(similar);

        //vec.plotTsne();
        Tsne tsne = new Tsne.Builder().setMaxIter(200)
                .learningRate(500).useAdaGrad(false)
                .normalize(false).usePca(false).build();
        vec.lookupTable().plotVocab(tsne);
        WordVectorSerializer.writeTsneFormat(vec, tsne.getY(), new File("coords.csv"));



    }

}
