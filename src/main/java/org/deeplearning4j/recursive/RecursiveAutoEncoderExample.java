package org.deeplearning4j.recursive;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.layers.feedforward.autoencoder.recursive.RecursiveAutoEncoder;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;

/**
 * Created by willow on 5/11/15.
 */
public class RecursiveAutoEncoderExample {

    public static void main(String[] args) throws Exception {

//      load data

        MnistDataFetcher fetcher = new MnistDataFetcher(true);
        fetcher.fetch(100);
        DataSet d2 = fetcher.next();
        INDArray input = d2.getFeatureMatrix();

//      build model

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .momentum(0.9f)
                .optimizationAlgo(OptimizationAlgorithm.ITERATION_GRADIENT_DESCENT)
                .corruptionLevel(0.3).weightInit(WeightInit.VI)
                .iterations(10)
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                .learningRate(1e-1f).nIn(784).nOut(600)
                .layer(new org.deeplearning4j.nn.conf.layers.RecursiveAutoEncoder())
                .build();

//      train model

        RecursiveAutoEncoder model = LayerFactories.getFactory(conf).create(conf,
                Arrays.<IterationListener>asList(new ScoreIterationListener(10)));
        model.setParams(model.params());
        model.fit(input);

        // Generative Model - unsupervised and requires different evaluation technique

    }
}
