package org.deeplearning4j.mnist.dimensionalityreduction;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Created by agibsonccc on 9/11/14.
 */
public class DimensionalityReduction {

    private static Logger log = LoggerFactory.getLogger(DimensionalityReduction.class);


    public static void main(String[] args) throws Exception {
        RandomGenerator gen = new MersenneTwister(123);
        LayerFactory l = LayerFactories.getFactory(RBM.class);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().weightInit(WeightInit.VI)
                .iterations(5).layerFactory(l)
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).rng(gen)
                .learningRate(1e-1f).nIn(784).nOut(10).list(4)
                .hiddenLayerSizes(new int[]{600, 500, 400})
                .build();




        MultiLayerNetwork network = new MultiLayerNetwork(conf);


        DataSetIterator iter = new MultipleEpochsIterator(10,new MnistDataSetIterator(1000,1000));
        network.fit(iter);
        iter.reset();
        //dimensionality reduced matrix
        INDArray output = network.output(iter.next().getFeatureMatrix());


        iter.reset();


    }

}
