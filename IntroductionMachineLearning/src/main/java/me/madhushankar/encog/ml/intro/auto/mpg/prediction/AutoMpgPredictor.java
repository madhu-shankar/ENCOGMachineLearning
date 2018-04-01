package me.madhushankar.encog.ml.intro.auto.mpg.prediction;

import org.encog.app.analyst.AnalystFileFormat;
import org.encog.app.analyst.EncogAnalyst;
import org.encog.app.analyst.csv.normalize.AnalystNormalizeCSV;
import org.encog.app.analyst.csv.segregate.SegregateCSV;
import org.encog.app.analyst.csv.segregate.SegregateTargetPercent;
import org.encog.app.analyst.csv.shuffle.ShuffleCSV;
import org.encog.app.analyst.wizard.AnalystWizard;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.util.arrayutil.NormalizationAction;
import org.encog.util.csv.CSVFormat;
import org.encog.util.simple.EncogUtility;

/**
 * Created by mdhu on 1/22/18
 */
public class AutoMpgPredictor {

    private ClassLoader classLoader = getClass().getClassLoader();

    public static void main(String[] args) {
        AutoMpgPredictor autoMpgPredictor = new AutoMpgPredictor();
        autoMpgPredictor.step1();
        autoMpgPredictor.step2();
        autoMpgPredictor.step3();
        autoMpgPredictor.step4();
        autoMpgPredictor.step5();
        autoMpgPredictor.step6();
    }

    private void step6() {
        evaluateNetwork();
    }

    private void evaluateNetwork() {
        BasicNetwork network = (BasicNetwork) EncogDirectoryPersistence.loadObject(AutoMpgPredictorConfig.getAutoMpgTrainedNetworkFile(classLoader));
        MLDataSet testDataSet = EncogUtility.loadCSV2Memory(AutoMpgPredictorConfig.getNormalizedTestDataFile(classLoader).getPath(), network.getInputCount(), network.getOutputCount(), true, CSVFormat.ENGLISH, false);

        EncogAnalyst analyst = new EncogAnalyst();
        analyst.load(AutoMpgPredictorConfig.getAnalystFile(classLoader));

        for (MLDataPair testDataPair : testDataSet) {
            MLData normalizedComputedOutput = network.compute(testDataPair.getInput());
            double computedOutput = analyst.getScript().getNormalize().getNormalizedFields().get(8).deNormalize(normalizedComputedOutput.getData(0));
            double idealOutput = analyst.getScript().getNormalize().getNormalizedFields().get(8).deNormalize(testDataPair.getIdeal().getData(0));
            System.out.println(String.format("Ideal: %s , Computed: %s", idealOutput, computedOutput));
        }

    }

    private void step5() {
        trainNetwork();
    }

    private void trainNetwork() {
        BasicNetwork network = (BasicNetwork) EncogDirectoryPersistence.loadObject(AutoMpgPredictorConfig.getAutoMpgTrainedNetworkFile(classLoader));
        MLDataSet trainingDataSet = EncogUtility.loadCSV2Memory(AutoMpgPredictorConfig.getNormalizedTrainingDataFile(classLoader).getPath(), network.getInputCount(), network.getOutputCount(), true, CSVFormat.ENGLISH, false);

        ResilientPropagation resilientPropagation = new ResilientPropagation(network, trainingDataSet);

        int epoch = 1;
        do {
            resilientPropagation.iteration();
            System.out.println(String.format("Iteration: %s Error %s", epoch, resilientPropagation.getError()));
            epoch++;
        } while (resilientPropagation.getError() > 0.01);
        EncogDirectoryPersistence.saveObject(AutoMpgPredictorConfig.getAutoMpgTrainedNetworkFile(classLoader), network);
    }

    private void step4() {
        createNetwork();
    }

    private void createNetwork() {
        BasicNetwork network = new BasicNetwork();
        /*Input layer with 22 inputs*/
        network.addLayer(new BasicLayer(new ActivationLinear(), true, 22));
        /*Hidden Layer*/
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 6));
        /*Output Layer*/
        network.addLayer(new BasicLayer(new ActivationTANH(), false, 1));
        /*Finalize the network structure*/
        network.getStructure().finalizeStructure();
        /*Initialize network with random weights*/
        network.reset();

        EncogDirectoryPersistence.saveObject(AutoMpgPredictorConfig.getAutoMpgTrainedNetworkFile(classLoader), network);
    }


    private void step3() {
        normalize();
    }

    private void normalize() {
        EncogAnalyst analyst = new EncogAnalyst();
        AnalystWizard wizard = new AnalystWizard(analyst);
        /*Pass the base file to wizard to understand the types of data, like data range, all kind of labels for a type etc.*/
        wizard.wizard(AutoMpgPredictorConfig.getBaseDataFile(classLoader), true, AnalystFileFormat.DECPNT_COMMA);

        //Cylinders
        analyst.getScript().getNormalize().getNormalizedFields().get(0).setAction(NormalizationAction.Equilateral);
        //Displacement
        analyst.getScript().getNormalize().getNormalizedFields().get(1).setAction(NormalizationAction.Normalize);
        //Horse-Power
        analyst.getScript().getNormalize().getNormalizedFields().get(2).setAction(NormalizationAction.Normalize);
        //Weight
        analyst.getScript().getNormalize().getNormalizedFields().get(3).setAction(NormalizationAction.Normalize);
        //Acceleration
        analyst.getScript().getNormalize().getNormalizedFields().get(4).setAction(NormalizationAction.Normalize);
        //Year
        analyst.getScript().getNormalize().getNormalizedFields().get(5).setAction(NormalizationAction.Equilateral);
        //Origin
        analyst.getScript().getNormalize().getNormalizedFields().get(6).setAction(NormalizationAction.Equilateral);
        //Name
        analyst.getScript().getNormalize().getNormalizedFields().get(7).setAction(NormalizationAction.Ignore);
        //mpg
        analyst.getScript().getNormalize().getNormalizedFields().get(8).setAction(NormalizationAction.Normalize);


        /*Normalize Training data*/
        AnalystNormalizeCSV analystNormalizeCSV = new AnalystNormalizeCSV();
        analystNormalizeCSV.analyze(AutoMpgPredictorConfig.getBaseTrainingDataFile(classLoader), true, CSVFormat.ENGLISH, analyst);
        analystNormalizeCSV.setProduceOutputHeaders(true);
        analystNormalizeCSV.normalize(AutoMpgPredictorConfig.getNormalizedTrainingDataFile(classLoader));

        /*Normalize Test data*/
        analystNormalizeCSV.analyze(AutoMpgPredictorConfig.getBaseTestDataFile(classLoader), true, CSVFormat.ENGLISH, analyst);
        analystNormalizeCSV.setProduceOutputHeaders(true);
        analystNormalizeCSV.normalize(AutoMpgPredictorConfig.getNormalizedTestDataFile(classLoader));

        /*
         * Save the analyst file to re-use while de-normalizing results.
         * The Analyst file is nothing but a set of tasks that are to be done to normalize/de-normalize.
         */
        analyst.save(AutoMpgPredictorConfig.getAnalystFile(classLoader));
    }

    private void step2() {
        segregate();
    }

    private void segregate() {
        SegregateCSV segregateCSV = new SegregateCSV();
        segregateCSV.getTargets().add(new SegregateTargetPercent(AutoMpgPredictorConfig.getBaseTrainingDataFile(classLoader), 75));
        segregateCSV.getTargets().add(new SegregateTargetPercent(AutoMpgPredictorConfig.getBaseTestDataFile(classLoader), 25));
        segregateCSV.setProduceOutputHeaders(true);
        segregateCSV.analyze(AutoMpgPredictorConfig.getShuffledBaseDataFile(classLoader), true, CSVFormat.ENGLISH);
        segregateCSV.process();
    }

    private void step1() {
        shuffle();
    }

    private void shuffle() {
        ShuffleCSV shuffleCSV = new ShuffleCSV();
        shuffleCSV.setProduceOutputHeaders(true);
        shuffleCSV.analyze(AutoMpgPredictorConfig.getBaseDataFile(classLoader), true, CSVFormat.ENGLISH);
        shuffleCSV.process(AutoMpgPredictorConfig.getShuffledBaseDataFile(classLoader));
    }
}
