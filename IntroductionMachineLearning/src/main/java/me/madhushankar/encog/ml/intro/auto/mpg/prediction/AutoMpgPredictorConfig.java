package me.madhushankar.encog.ml.intro.auto.mpg.prediction;

import java.io.File;

/**
 * Created by mdhu on 1/22/18
 */
public class AutoMpgPredictorConfig {
    private static final String BASE_DATA_FILE_NAME = "META-INF/AutoMPG/AutoMPG.csv";
    private static final String SHUFFLED_BASE_DATA_FILENAME = "META-INF/AutoMPG/ShuffledAutoMPGData.csv";
    private static final String BASE_TRAINING_DATA_FILE = "META-INF/AutoMPG/BaseTrainingData.csv";
    private static final String BASE_TEST_DATA_FILE = "META-INF/AutoMPG/BaseTestData.csv";
    private static final String NORMALIZED_TRAINING_DATA_FILE = "META-INF/AutoMPG/NormalizedTrainingData.csv";
    private static final String NORMALIZED_TEST_DATA_FILE = "META-INF/AutoMPG/NormalizedTestData.csv";
    private static final String ANALYST_FILE = "META-INF/AutoMPG/AnalystFile.ega";
    private static final String AUTO_MPG_TRAINED_NETWORK_FILE = "META-INF/AutoMPG/AutoMPGTrainedNetwork.eg";
    private static final String VALIDATION_RESULT = "META-INF/AutoMPG/AutoMPGValidationResult.csv";

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

    static File getAutoMpgTrainedNetworkFile(ClassLoader classLoader) {
        return new File(classLoader.getResource(AUTO_MPG_TRAINED_NETWORK_FILE).getFile());
    }

    static File getValidationResultFile(ClassLoader classLoader) {
        return new File(classLoader.getResource(VALIDATION_RESULT).getFile());
    }
}
