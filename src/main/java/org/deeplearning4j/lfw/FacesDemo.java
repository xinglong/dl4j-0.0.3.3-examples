package org.deeplearning4j.lfw;


import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.plot.iterationlistener.NeuralNetPlotterIterationListener;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;


/**
 * Created by agibsonccc on 10/2/14.
 */
public class FacesDemo {
    private static Logger log = LoggerFactory.getLogger(FacesDemo.class);


    public static void main(String[] args) throws Exception {

        DataSetIterator fetcher = new LFWDataSetIterator(28,28);


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN).layer(new org.deeplearning4j.nn.conf.layers.RBM())
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED).weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(0, 1))
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-3f).nIn(fetcher.inputColumns()).nOut(fetcher.totalOutcomes())
                .list(4).hiddenLayerSizes(new int[]{600, 250, 200}).override(new ClassifierOverride(3)).build();

        MultiLayerNetwork d = new MultiLayerNetwork(conf);
        d.setListeners(Arrays.asList((IterationListener) new NeuralNetPlotterIterationListener(1)));

        while(fetcher.hasNext()) {
            DataSet next = fetcher.next();
            next.normalizeZeroMeanZeroUnitVariance();
            d.fit(next);

        }



    }


}
