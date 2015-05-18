package org.deeplearning4j.recursive;

import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.writable.Writable;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.rntn.RNTN;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.nn.layers.feedforward.autoencoder.recursive.Tree;
import org.deeplearning4j.text.corpora.treeparser.TreeVectorizer;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.deeplearning4j.text.invertedindex.LuceneInvertedIndex;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareFileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;


/**
 * Recursive Neural Tensor Network (RNTN)
 *
 * Created by willow on 5/11/15.
 */
public class RNTNExample {
    private static final Logger log = LoggerFactory.getLogger(RNTNExample.class);

    public static void main(String[] args) throws Exception {

        //Swap with Rotten Tomatoes example or Twitter ?
//        https://github.com/SkymindIO/deeplearning4j-nlp-examples/blob/master/src/main/java/org/deeplearning4j/rottentomatoes/rntn/RNTNTrain.java
//        RottenTomatoesWordVectorDataFetcher fetcher = new RottenTomatoesWordVectorDataFetcher();
//        RottenTomatoesLabelAwareSentenceIterator iter = new RottenTomatoesLabelAwareSentenceIterator();
//        RNTN t = new RNTN.Builder()
//                .setActivationFunction(Activations.hardTanh()).setFeatureVectors(fetcher.getVec())
//                .setUseTensors(true).build();
//
//        TreeVectorizer vectorizer = new TreeVectorizer(new TreeParser());

        // use word2vec as a lookup - feed rntn consitnuency tables - parse - sentence iterator that iterates through corpus
        // get corpus and feed into sentence iterator, fit vectors, loop and fit rntn
        String fileName = "tweets_clean.txt";

        LabelAwareSentenceIterator iter = new LabelAwareFileSentenceIterator(new File(fileName));

        Word2Vec vec;
        TokenizerFactory tokenizerFactory = new UimaTokenizerFactory(false);
        VocabCache cache = new InMemoryLookupCache();
        InvertedIndex index = new LuceneInvertedIndex.Builder()
                .indexDir(new File("rntn-index")).cache(cache).build();
        WeightLookupTable lookupTable = new InMemoryLookupTable.Builder().cache(cache)
                .vectorLength(100).build();


//      load data
        CSVRecordReader data = new CSVRecordReader();
         data.initialize(new FileSplit(new File(fileName)));
        Collection<Writable> next = data.next();
//        RecordReader image = new ImageRecordReader(numRows,numColumns,true);
//        image.initialize(new FileSplit(new File(baseDirectory, "pod")));
//        DataSetIterator iter = new RecordReaderDataSetIterator(image, batchSize, numRows * numColumns, 2);

//      normalize, shape & split data
//        DataSet all = iter.next();
//        all.scale();
//        all.shuffle();
//        DataSetIterator allIter = new ListDataSetIterator(all.asList(),100);



                // vectorize data
        TreeVectorizer vectorizer = new TreeVectorizer();

        vec = new Word2Vec.Builder()
                .vocabCache(cache).index(index)
                .iterate(iter).tokenizerFactory(tokenizerFactory)
                .lookupTable(lookupTable).build();
        vec.fit();


//      build model

        RNTN rntn = new RNTN.Builder().setActivationFunction("tanh")
                .setAdagradResetFrequency(1)
                .setCombineClassification(true).setFeatureVectors(vec)
                .setRandomFeatureVectors(false)
                .setUseTensors(false).build();

//      train model
        while(iter.hasNext()) {
            // this is looped with fit
            List<Tree> trees = vectorizer.getTreesWithLabels(iter.nextSentence(), iter.currentLabel(), Arrays.asList("0", "1", "2", "3", "4"));

            rntn.fit(trees);


//      test model

        // RNTN evalu will eval per node - each sentence is a parse tree

        // rntn eval - positive and negative sentiment
//        Evaluation eval = new Evaluation();
//        INDArray predictedOutput = rntn.output(...);
//        ???
//        log.info("Score " + eval.stats());

        }

    }
}
