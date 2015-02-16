package org.deeplearning4j.iris;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.distributions.Distributions;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.layers.factory.DefaultLayerFactory;
import org.deeplearning4j.nn.layers.factory.PretrainLayerFactory;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.activation.Activations;
import org.nd4j.linalg.api.ndarray.DimensionSlice;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.SliceOp;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by agibsonccc on 9/12/14.
 */
public class IrisExample {


    private static Logger log = LoggerFactory.getLogger(IrisExample.class);

    public static void main(String[] args) {
        RandomGenerator gen = new MersenneTwister(123);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(100).layerFactory(new PretrainLayerFactory(RBM.class))
                .weightInit(WeightInit.SIZE).dist(Distributions.normal(gen,1e-1))
                .activationFunction(Activations.tanh()).momentum(0.9).dropOut(0.8)
                .optimizationAlgo(OptimizationAlgorithm.GRADIENT_DESCENT)
                .constrainGradientToUnitNorm(true).k(5).regularization(true).l2(2e-4)
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN).hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                .rng(gen)
                .learningRate(1e-1f)
                .nIn(4).nOut(3).list(2).useDropConnect(false)
                .hiddenLayerSizes(new int[]{3})
                .override(new ClassifierOverride(1)).build();





        MultiLayerNetwork d = new MultiLayerNetwork(conf);


        DataSetIterator iter = new IrisDataSetIterator(150, 150);

        DataSet next = iter.next();

        next.normalizeZeroMeanZeroUnitVariance();
        next.shuffle();

        SplitTestAndTrain testAndTrain = next.splitTestAndTrain(110);
        DataSet train = testAndTrain.getTrain();

        d.fit(train);


        DataSet test = testAndTrain.getTest();


        Evaluation eval = new Evaluation();
        INDArray output = d.output(test.getFeatureMatrix());
        eval.eval(test.getLabels(),output);
        log.info("Score " + eval.stats());
    }
}
