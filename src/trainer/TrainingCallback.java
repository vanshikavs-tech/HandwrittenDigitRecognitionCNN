package trainer;

/**
 * Callback interface to decouple the training engine from console or GUI progress bars.
 * Employs the Observer / Listener pattern.
 */
public interface TrainingCallback {

    default void onEpochStart(int epoch, int totalEpochs) {}

    default void onBatchEnd(int batch, int totalBatches, double currentLoss) {}

    default void onEpochEnd(int epoch, int totalEpochs, double avgLoss, double accuracy) {}

    default void onTrainingComplete(double finalLoss, double finalAccuracy) {}
}
