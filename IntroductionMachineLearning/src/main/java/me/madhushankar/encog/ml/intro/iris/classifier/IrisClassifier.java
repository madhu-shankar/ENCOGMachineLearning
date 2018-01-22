package me.madhushankar.encog.ml.intro.iris.classifier;

import org.encog.app.analyst.AnalystFileFormat;
import org.encog.app.analyst.EncogAnalyst;
import org.encog.app.analyst.csv.normalize.AnalystNormalizeCSV;
import org.encog.app.analyst.csv.segregate.SegregateCSV;
import org.encog.app.analyst.csv.segregate.SegregateTargetPercent;
import org.encog.app.analyst.csv.shuffle.ShuffleCSV;
import org.encog.app.analyst.wizard.AnalystWizard;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.mathutil.Equilateral;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.util.csv.CSVFormat;
import org.encog.util.simple.EncogUtility;

import java.util.Iterator;

/**
 * Created by mdhu on 1/16/18
 */
public class IrisClassifier {
    private final ClassLoader classLoader = getClass().getClassLoader();

    public static void main(String[] args) {
        IrisClassifier irisClassifier = new IrisClassifier();
        irisClassifier.step1();
        irisClassifier.step2();
        irisClassifier.step3();
        irisClassifier.step4();
        irisClassifier.step5();
        irisClassifier.step6();
    }

    private void step6() {
        evaluateNetwork();
    }

    private void evaluateNetwork() {
        BasicNetwork network = (BasicNetwork) EncogDirectoryPersistence.loadObject(IrisClassifierConfig.getIrisTrainedNetworkFile(classLoader));
        EncogAnalyst analyst = new EncogAnalyst();
        analyst.load(IrisClassifierConfig.getAnalystFile(classLoader));

        MLDataSet testData = EncogUtility.loadCSV2Memory(IrisClassifierConfig.getNormalizedTestDataFile(classLoader).getPath(),
                                                         network.getInputCount(), network.getOutputCount(),
                                                         true, CSVFormat.ENGLISH, false);

        int count = 0;
        int correctPredictionCount = 0;

        final Iterator<MLDataPair> testDataIterator = testData.iterator();
        while (testDataIterator.hasNext()) {
            final MLDataPair mlDataPair = testDataIterator.next();

            count++;
            final MLData predictedOutput = network.compute(mlDataPair.getInput());

            final double sepal_l = analyst.getScript().getNormalize().getNormalizedFields().get(0).deNormalize(mlDataPair.getInput().getData(0));
            final double sepal_w = analyst.getScript().getNormalize().getNormalizedFields().get(1).deNormalize(mlDataPair.getInput().getData(1));
            final double petal_l = analyst.getScript().getNormalize().getNormalizedFields().get(2).deNormalize(mlDataPair.getInput().getData(2));
            final double petal_w = analyst.getScript().getNormalize().getNormalizedFields().get(3).deNormalize(mlDataPair.getInput().getData(3));

            int classCounts = analyst.getScript().getNormalize().getNormalizedFields().get(4).getClasses().size();
            double rangeMin = analyst.getScript().getNormalize().getNormalizedFields().get(4).getNormalizedLow();
            double rangeMax = analyst.getScript().getNormalize().getNormalizedFields().get(4).getNormalizedHigh();
            final Equilateral equilateral = new Equilateral(classCounts, rangeMax, rangeMin);

            final int predictedClassDecoded = equilateral.decode(predictedOutput.getData());
            final String predictedClassName = analyst.getScript().getNormalize().getNormalizedFields().get(4).getClasses().get(predictedClassDecoded).getName();

            final int idealClassDecoded = equilateral.decode(mlDataPair.getIdeal().getData());
            final String idealClassName = analyst.getScript().getNormalize().getNormalizedFields().get(4).getClasses().get(idealClassDecoded).getName();

            if (predictedClassName.equals(idealClassName)) {
                correctPredictionCount++;
            }
            System.out.println(String.format("Count :%s Properties [%s,%s,%s,%s] ,Ideal : %s Predicted : %s ",
                                                 count, sepal_l, sepal_w, petal_l, petal_w, idealClassName, predictedClassName));
        }
        System.out.println("Total Test Count : " + count);
        System.out.println("Total Correct Prediction Count : " + correctPredictionCount);
        System.out.println("% Success : " +((correctPredictionCount * 100.0) / count));

    }

    private void step5() {
        trainNetwork();
    }

    private void trainNetwork() {
        BasicNetwork network = (BasicNetwork) EncogDirectoryPersistence.loadObject(IrisClassifierConfig.getIrisTrainedNetworkFile(classLoader));
        final MLDataSet normalizedTrainingData = EncogUtility.loadCSV2Memory(IrisClassifierConfig.getNormalizedTrainingDataFile(classLoader).getPath(),
                                                                  network.getInputCount(), network.getOutputCount(),
                                                                  true, CSVFormat.ENGLISH, false);
        final ResilientPropagation resilientPropagation = new ResilientPropagation(network, normalizedTrainingData);

        double epoch = 1;
        do {
            resilientPropagation.iteration();
            System.out.println(String.format("Iteration %s error %s", epoch, resilientPropagation.getError()));
            epoch++;
        }while (resilientPropagation.getError() > 0.01);

        EncogDirectoryPersistence.saveObject(IrisClassifierConfig.getIrisTrainedNetworkFile(classLoader), network);
    }

    private void step4() {
        createNetwork();
    }

    private void createNetwork() {
        BasicNetwork network = new BasicNetwork();
        /*
         * We don't need activation function for the input layer.
         * So it can be set as either null or LinearActivationFunction.
         * Since we have 4 inputs, we have used 4 input neurons.
         */
        network.addLayer(new BasicLayer(new ActivationLinear(), true, 4));
        /*
         * number of neurons in the hidden layer can decided by trial and error.
         * This process is called Network Pruning.
         */
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 6));
        /*
         * Since we have used equilateral encoding for the species, there are max two values(for 3 species i.e., n-1),
         * we are having 2 neurons in the output layer. Also, we have two columns for species in our normalized data.
         */
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 2));
        network.getStructure().finalizeStructure();
        /*
         * Randomly initialize all connection weights.
         */
        network.reset();
        /*
         * Persist the network to disk.
         */
        EncogDirectoryPersistence.saveObject(IrisClassifierConfig.getIrisTrainedNetworkFile(classLoader), network);
    }

    private void step3() {
        normalize();
    }

    private void normalize() {
        EncogAnalyst analyst = new EncogAnalyst();
        AnalystWizard wizard = new AnalystWizard(analyst);
        /*Pass the base file to wizard to understand the types of data, like data range, all kind of labels for a type etc.*/
        wizard.wizard(IrisClassifierConfig.getBaseDataFile(classLoader), true, AnalystFileFormat.DECPNT_COMMA);

        /*Normalize Training data*/
        AnalystNormalizeCSV analystNormalizeCSV = new AnalystNormalizeCSV();
        analystNormalizeCSV.analyze(IrisClassifierConfig.getBaseTrainingDataFile(classLoader), true, CSVFormat.ENGLISH, analyst);
        analystNormalizeCSV.setProduceOutputHeaders(true);
        analystNormalizeCSV.normalize(IrisClassifierConfig.getNormalizedTrainingDataFile(classLoader));

        /*Normalize Test data*/
        analystNormalizeCSV.analyze(IrisClassifierConfig.getBaseTestDataFile(classLoader), true, CSVFormat.ENGLISH, analyst);
        analystNormalizeCSV.setProduceOutputHeaders(true);
        analystNormalizeCSV.normalize(IrisClassifierConfig.getNormalizedTestDataFile(classLoader));

        /*
         * Save the analyst file to re-use while de-normalizing results.
         * The Analyst file is nothing but a set of tasks that are to be done to normalize/de-normalize.
         */
        analyst.save(IrisClassifierConfig.getAnalystFile(classLoader));
    }

    private void step2() {
        segregate();
    }

    private void segregate() {
        SegregateCSV segregateCSV = new SegregateCSV();
        segregateCSV.getTargets().add(new SegregateTargetPercent(IrisClassifierConfig.getBaseTrainingDataFile(classLoader), 75));
        segregateCSV.getTargets().add(new SegregateTargetPercent(IrisClassifierConfig.getBaseTestDataFile(classLoader), 25));
        segregateCSV.setProduceOutputHeaders(true);
        segregateCSV.analyze(IrisClassifierConfig.getShuffledBaseDataFile(classLoader), true, CSVFormat.ENGLISH);
        segregateCSV.process();
    }

    private void step1() {
        shuffle();
    }

    private void shuffle() {
        ShuffleCSV shuffleCSV = new ShuffleCSV();
        shuffleCSV.setProduceOutputHeaders(true);
        shuffleCSV.analyze(IrisClassifierConfig.getBaseDataFile(classLoader), true, CSVFormat.ENGLISH);
        shuffleCSV.process(IrisClassifierConfig.getShuffledBaseDataFile(classLoader));
    }
}
