package model;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import network.NeuralNetwork;

/**
 * ModelSerializer handles persisting and restoring trained neural networks to and from disk.
 * Allows trained CNN models to be loaded instantly in the GUI without retraining.
 */
public class ModelSerializer {

    public static final String DEFAULT_MODEL_PATH = "models/digit-cnn.model";

    /**
     * Saves a trained NeuralNetwork to disk.
     */
    public static void saveModel(NeuralNetwork network, String filePath) throws IOException {
        File file = new File(filePath);
        File parent = file.getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }

        try (ObjectOutputStream oos = new ObjectOutputStream(
                new BufferedOutputStream(new FileOutputStream(file)))) {
            oos.writeObject(network);
            System.out.println("Model successfully saved to: " + file.getAbsolutePath());
        }
    }

    /**
     * Loads a trained NeuralNetwork from disk.
     */
    public static NeuralNetwork loadModel(String filePath) throws IOException, ClassNotFoundException {
        File file = new File(filePath);
        if (!file.exists()) {
            throw new IOException("Model file not found: " + file.getAbsolutePath());
        }

        try (ObjectInputStream ois = new ObjectInputStream(
                new BufferedInputStream(new FileInputStream(file)))) {
            NeuralNetwork network = (NeuralNetwork) ois.readObject();
            System.out.println("Model successfully loaded from: " + file.getAbsolutePath());
            return network;
        }
    }

    /**
     * Checks if default model file exists.
     */
    public static boolean defaultModelExists() {
        return new File(DEFAULT_MODEL_PATH).exists();
    }
}
