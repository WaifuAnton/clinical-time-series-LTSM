package org.app;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.classification.ROC;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;

public class Main {
    private static final String DATA_PATH = "src/main/resources/";

    private static final int NB_TRAIN_EXAMPLES = 3300; // number of training examples
    private static final int NB_TEST_EXAMPLES = 700; // number of testing examples

    private static final int NB_INPUTS = 86;
    private static final int NB_EPOCHS = 10;
    private static final double LEARNING_RATE = 0.0025;
    private static final int BATCH_SIZE = 32;
    private static final int LSTM_LAYER_SIZE = 200;
    private static final int NUM_LABEL_CLASSES = 2;

    public static void main(String[] args) throws IOException, InterruptedException {
        long start = System.currentTimeMillis();

        String path = FilenameUtils.concat(DATA_PATH, "physionet2012/"); // set parent directory

        String featureBaseDir = FilenameUtils.concat(path, "sequence"); // set feature directory
        String mortalityBaseDir = FilenameUtils.concat(path, "mortality"); // set label directory

// Load training data

        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader(1, ",");
        trainFeatures.initialize( new NumberedFileInputSplit(featureBaseDir + "/%d.csv", 0, NB_TRAIN_EXAMPLES - 1));

        SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
        trainLabels.initialize(new NumberedFileInputSplit(mortalityBaseDir + "/%d.csv", 0, NB_TRAIN_EXAMPLES - 1));

        DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels,
                32, 2, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);


// Load testing data
        SequenceRecordReader testFeatures = new CSVSequenceRecordReader(1, ",");
        testFeatures.initialize(new NumberedFileInputSplit(featureBaseDir + "/%d.csv", NB_TRAIN_EXAMPLES, NB_TRAIN_EXAMPLES + NB_TEST_EXAMPLES - 1));

        SequenceRecordReader testLabels = new CSVSequenceRecordReader();
        testLabels.initialize(new NumberedFileInputSplit(mortalityBaseDir + "/%d.csv", NB_TRAIN_EXAMPLES, NB_TRAIN_EXAMPLES  + NB_TEST_EXAMPLES - 1));

        DataSetIterator testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels,
                BATCH_SIZE, 2, false,SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(System.currentTimeMillis())
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(LEARNING_RATE))
                .weightInit(WeightInit.XAVIER)
                .dropOut(0.25)
                .graphBuilder()
                .addInputs("trainFeatures")
                .setOutputs("predictMortality")
                .addLayer("L1", new LSTM.Builder()
                                .nIn(NB_INPUTS)
                                .nOut(LSTM_LAYER_SIZE)
                                .forgetGateBiasInit(1)
                                .activation(Activation.TANH)
                                .build(),
                        "trainFeatures")
                .addLayer("predictMortality", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(LSTM_LAYER_SIZE).nOut(NUM_LABEL_CLASSES).build(),"L1")
                .build();

        ComputationGraph model = new ComputationGraph(conf);
        model.fit(trainData, NB_EPOCHS);

        ROC roc = new ROC(100);

        while (testData.hasNext()) {
            DataSet batch = testData.next();
            INDArray[] output = model.output(batch.getFeatures());
            roc.evalTimeSeries(batch.getLabels(), output[0]);
        }

        System.out.println(("FINAL TEST AUC: " + roc.calculateAUC()));
        System.out.println(System.currentTimeMillis() - start);
    }
}
