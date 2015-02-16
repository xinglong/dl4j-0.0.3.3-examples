package org.deeplearning4j.mnist.full;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ComposableIterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.plot.iterationlistener.NeuralNetPlotterIterationListener;
import org.nd4j.linalg.api.activation.Activations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.jcublas.SimpleJCublas;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;

/**
 * Created by agibsonccc on 9/11/14.
 */
public class DBNExample {

    private static Logger log = LoggerFactory.getLogger(DBNExample.class);


    public static void main(String[] args) throws Exception {
        RandomGenerator gen = new MersenneTwister(123);
        SimpleJCublas.init();
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().momentum(0.5)
                .momentumAfter(Collections.singletonMap(3, 0.9)).layerFactory(LayerFactories.getFactory(RBM.class)).optimizationAlgo(OptimizationAlgorithm.ITERATION_GRADIENT_DESCENT)
                .iterations(1).weightInit(WeightInit.SIZE).applySparsity(true).sparsity(0.1).iterationListener(new IterationListener() {
                    @Override
                    public void iterationDone(Model model, int iteration) {
                        
                    }
                })
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).rng(gen)
                .learningRate(1e-1f).nIn(784).nOut(10).list(4)
                .hiddenLayerSizes(new int[]{500, 250, 200})
                .override(new ClassifierOverride(3))
                .build();


        MultiLayerNetwork d = new MultiLayerNetwork(conf);
        DataSetIterator iter = new MultipleEpochsIterator(3,new MnistDataSetIterator(10,60000));
        while(iter.hasNext())
            d.fit(iter.next());



        iter.reset();

        Evaluation eval = new Evaluation();

        while(iter.hasNext()) {

            DataSet d2 = iter.next();
            INDArray predict2 = d.output(d2.getFeatureMatrix());

            eval.eval(d2.getLabels(), predict2);

        }

        log.info(eval.stats());


    }

}
