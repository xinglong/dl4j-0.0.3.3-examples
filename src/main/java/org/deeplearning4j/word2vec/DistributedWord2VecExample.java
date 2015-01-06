package org.deeplearning4j.word2vec;

import org.deeplearning4j.bagofwords.vectorizer.TextVectorizer;
import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.Huffman;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.nn.conf.Configuration;
import org.deeplearning4j.plot.Tsne;
import org.deeplearning4j.scaleout.actor.runner.DeepLearning4jDistributed;
import org.deeplearning4j.scaleout.api.statetracker.StateTracker;
import org.deeplearning4j.scaleout.api.workrouter.WorkRouter;
import org.deeplearning4j.scaleout.job.JobIterator;
import org.deeplearning4j.scaleout.perform.WorkerPerformerFactory;
import org.deeplearning4j.scaleout.perform.models.word2vec.Word2VecPerformer;
import org.deeplearning4j.scaleout.perform.models.word2vec.Word2VecPerformerFactory;
import org.deeplearning4j.scaleout.perform.models.word2vec.iterator.Word2VecJobIterator;
import org.deeplearning4j.scaleout.statetracker.hazelcast.HazelCastStateTracker;
import org.deeplearning4j.scaleout.workrouter.IterativeReduceWorkRouter;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.deeplearning4j.text.invertedindex.LuceneInvertedIndex;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.springframework.core.io.ClassPathResource;

import java.util.Collection;

/**
 * Created by agibsonccc on 12/1/14.
 */
public class DistributedWord2VecExample {

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



        VocabCache cache = new InMemoryLookupCache();



        WeightLookupTable table =  new InMemoryLookupTable
                .Builder().vectorLength(300).useAdaGrad(false).cache(cache)
                .lr(0.025f).build();

        InvertedIndex invertedIndex = new LuceneInvertedIndex.Builder()
                .cache(cache)
                .build();

        TextVectorizer vectorizer = new TfidfVectorizer.Builder().index(invertedIndex)
                .cache(cache).iterate(iter)
                .tokenize(t).build();

        vectorizer.fit();


        Huffman huffman = new Huffman(cache.vocabWords());
        huffman.build();

        Configuration conf = new Configuration();
        conf.set(WorkerPerformerFactory.WORKER_PERFORMER, Word2VecPerformerFactory.class.getName());
        conf.set(WorkRouter.WORK_ROUTER, IterativeReduceWorkRouter.class.getName());
        Word2VecPerformer.configure((InMemoryLookupTable) table, invertedIndex, conf);

        StateTracker stateTracker = new HazelCastStateTracker();
        JobIterator iter2 = new Word2VecJobIterator(vectorizer,table,cache,stateTracker,10000);

        DeepLearning4jDistributed distributed = new DeepLearning4jDistributed(iter2,stateTracker);
        distributed.setup(conf);
        distributed.train();

        stateTracker.shutdown();
        distributed.shutdown();

        Tsne tsne = new Tsne.Builder().setMaxIter(200)
                .learningRate(500).useAdaGrad(false)
                .normalize(false).usePca(false).build();


        Word2Vec vec = new Word2Vec.Builder()
                .lookupTable(table).vocabCache(cache)
                .build();


        double sim = vec.similarity("day","night");
        Collection<String> similar = vec.wordsNearest("day",20);
        System.out.println(similar);


        table.plotVocab(tsne);



    }


}
