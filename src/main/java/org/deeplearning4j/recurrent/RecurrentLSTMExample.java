package org.deeplearning4j.recurrent;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.layers.recurrent.LSTM;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;

/**
 * Created by willow on 5/11/15.
 */

public class RecurrentLSTMExample {

    public static void main(String[] args) throws Exception {

//      load data
        MnistDataFetcher fetcher = new MnistDataFetcher(true);
        fetcher.fetch(1000);
        DataSet d2 = fetcher.next();
        INDArray input = d2.getFeatureMatrix();

//      build model

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().activationFunction("tanh")
                .layer(new org.deeplearning4j.nn.conf.layers.LSTM()).optimizationAlgo(OptimizationAlgorithm.LBFGS)
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                .nIn(784).nOut(10).build();
        LSTM l = LayerFactories.getFactory(conf.getLayer()).create(conf, Arrays.<IterationListener>asList(new ScoreIterationListener(10)));

//      train model

        l.fit(input);

    // Generative Model - unsupervised and its time series based which requires different evaluation technique

    }

}
