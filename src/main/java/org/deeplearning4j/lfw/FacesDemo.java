package org.deeplearning4j.lfw;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.distributions.Distributions;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.activation.Activations;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Created by agibsonccc on 10/2/14.
 */
public class FacesDemo {
    private static Logger log = LoggerFactory.getLogger(FacesDemo.class);


    public static void main(String[] args) throws Exception {
        RandomGenerator gen = new MersenneTwister(123);

        DataSetIterator fetcher = new LFWDataSetIterator(28,28);
        LayerFactory l = LayerFactories.getFactory(RBM.class);


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN).render(10).layerFactory(l)
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED).weightInit(WeightInit.DISTRIBUTION).dist(Distributions.normal(gen,1e-4))
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).rng(gen)
                .learningRate(1e-3f).nIn(fetcher.inputColumns()).nOut(fetcher.totalOutcomes())
                .list(4).hiddenLayerSizes(new int[]{500, 250, 200}).override(new NeuralNetConfiguration.ConfOverride() {
            @Override
            public void override(int i, NeuralNetConfiguration.Builder builder) {
                if(i == 3) {
                    builder.layerFactory(LayerFactories.getFactory(OutputLayer.class));
                    builder.weightInit(WeightInit.ZERO);
                    builder.activationFunction(Activations.softMaxRows());
                    builder.lossFunction(LossFunctions.LossFunction.MCXENT);

                }
            }
        }).build();

        MultiLayerNetwork d = new MultiLayerNetwork(conf);

        while(fetcher.hasNext()) {
            DataSet next = fetcher.next();
            next.normalizeZeroMeanZeroUnitVariance();
            d.fit(next);

        }


    }


}
