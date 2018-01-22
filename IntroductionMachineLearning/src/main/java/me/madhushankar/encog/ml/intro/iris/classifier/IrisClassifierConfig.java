package me.madhushankar.encog.ml.intro.iris.classifier;

import java.io.File;

/**
 * Created by mdhu on 1/16/18
 */
public class IrisClassifierConfig {

    private static final String BASE_DATA_FILE_NAME = "META-INF/IrisData.csv";
    public static final String SHUFFLED_BASE_DATA_FILENAME = "META-INF/ShuffledIrisData.csv";
    private static final String BASE_TRAINING_DATA_FILE = "META-INF/BaseTrainingData.csv";
    private static final String BASE_TEST_DATA_FILE = "META-INF/BaseTestData.csv";
    private static final String NORMALIZED_TRAINING_DATA_FILE = "META-INF/NormalizedTrainingData.csv";
    private static final String NORMALIZED_TEST_DATA_FILE = "META-INF/NormalizedTestData.csv";
    private static final String ANALYST_FILE = "META-INF/AnalystFile.ega";
    private static final String IRIS_TRAINED_NETWORK_FILE = "META-INF/IrisTrainedNetwork.eg";

    static File getBaseDataFile(ClassLoader classLoader) {
        return new File(classLoader.getResource(BASE_DATA_FILE_NAME).getFile());
    }

    static File getShuffledBaseDataFile(ClassLoader classLoader) {
        return new File(classLoader.getResource(SHUFFLED_BASE_DATA_FILENAME).getFile());
    }

    static File getBaseTrainingDataFile(ClassLoader classLoader) {
        return new File(classLoader.getResource(BASE_TRAINING_DATA_FILE).getFile());
    }

    static File getBaseTestDataFile(ClassLoader classLoader) {
        return new File(classLoader.getResource(BASE_TEST_DATA_FILE).getFile());
    }

    static File getNormalizedTrainingDataFile(ClassLoader classLoader) {
        return new File(classLoader.getResource(NORMALIZED_TRAINING_DATA_FILE).getFile());
    }

    static File getNormalizedTestDataFile(ClassLoader classLoader) {
        return new File(classLoader.getResource(NORMALIZED_TEST_DATA_FILE).getFile());
    }

    static File getAnalystFile(ClassLoader classLoader) {
        return new File(classLoader.getResource(ANALYST_FILE).getFile());
    }

    static File getIrisTrainedNetworkFile(ClassLoader classLoader) {
        return new File(classLoader.getResource(IRIS_TRAINED_NETWORK_FILE).getFile());
    }
}
