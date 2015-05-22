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
import org.deeplearning4j.util.ImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
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

//        d.setListeners(Arrays.asList((IterationListener) new NeuralNetPlotterIterationListener(1)));

        while(fetcher.hasNext()) {
            DataSet next = fetcher.next();
            next.normalizeZeroMeanZeroUnitVariance();
            d.fit(next);

        }

        File savedNetworkFile = saveNetwork(d);
        MultiLayerNetwork d1 = loadNetwork(savedNetworkFile);

        System.out.print(d);
        System.out.println("========");
        System.out.print(d1);

//        File testFile = new File("/home/xinglong/local/data/bw/image_recognition/lfw/Jeanne_Moreau/Jeanne_Moreau_0001.jpg");
//        ImageLoader imageLoader = new ImageLoader();
//        INDArray testImage = imageLoader.asMatrix(testFile);
//        System.out.println(testImage);


    }

    public static File saveNetwork(MultiLayerNetwork network) throws IOException {
        File file = new File("/tmp/nn.ser");
        OutputStream outputStream= new FileOutputStream("/tmp/nn.ser");
        OutputStream buffer = new BufferedOutputStream(outputStream);
        ObjectOutput output = new ObjectOutputStream(buffer);

        output.writeObject(network);
        output.close();
        return file;
    }

    public static MultiLayerNetwork loadNetwork(File networkFile) throws IOException, ClassNotFoundException {
        InputStream inputFile = new FileInputStream(networkFile);
        InputStream buffer = new BufferedInputStream(inputFile);
        ObjectInput input = new ObjectInputStream(buffer);

        MultiLayerNetwork network = (MultiLayerNetwork) input.readObject();
        input.close();

        return network;
    }


}
