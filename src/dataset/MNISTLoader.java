package dataset;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.zip.GZIPInputStream;
import math.Tensor;

/**
 * MNISTLoader loads and parses handwritten digit datasets.
 * Features:
 *   1. Parses standard binary IDX files (train-images-idx3-ubyte, train-labels-idx1-ubyte)
 *   2. Auto-downloads and decompresses from standard web mirrors if missing
 *   3. Generates high-quality synthetic digit samples for instant offline testing and demos
 */
public class MNISTLoader {

    private static final String DEFAULT_DATA_DIR = "data/mnist";
    private static final String BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/";

    private static final String TRAIN_IMAGES = "train-images-idx3-ubyte";
    private static final String TRAIN_LABELS = "train-labels-idx1-ubyte";
    private static final String TEST_IMAGES = "t10k-images-idx3-ubyte";
    private static final String TEST_LABELS = "t10k-labels-idx1-ubyte";

    /**
     * Loads MNIST training set (up to maxSamples, or all if maxSamples <= 0).
     */
    public static List<DataSample> loadTrainingData(int maxSamples) throws IOException {
        ensureDataFilesDownloaded();
        File imgFile = new File(DEFAULT_DATA_DIR, TRAIN_IMAGES);
        File lblFile = new File(DEFAULT_DATA_DIR, TRAIN_LABELS);
        return loadFromIdxFiles(imgFile, lblFile, maxSamples);
    }

    /**
     * Loads MNIST test set (up to maxSamples, or all if maxSamples <= 0).
     */
    public static List<DataSample> loadTestData(int maxSamples) throws IOException {
        ensureDataFilesDownloaded();
        File imgFile = new File(DEFAULT_DATA_DIR, TEST_IMAGES);
        File lblFile = new File(DEFAULT_DATA_DIR, TEST_LABELS);
        return loadFromIdxFiles(imgFile, lblFile, maxSamples);
    }

    /**
     * Parses standard IDX formatted image and label files.
     */
    public static List<DataSample> loadFromIdxFiles(File imageFile, File labelFile, int maxSamples) throws IOException {
        if (!imageFile.exists() || !labelFile.exists()) {
            throw new IOException("Dataset files not found: " + imageFile + " / " + labelFile);
        }

        try (DataInputStream imgIn = new DataInputStream(new BufferedInputStream(new FileInputStream(imageFile)));
             DataInputStream lblIn = new DataInputStream(new BufferedInputStream(new FileInputStream(labelFile)))) {

            int imgMagic = imgIn.readInt();
            if (imgMagic != 2051) {
                throw new IOException("Invalid image file magic number: " + imgMagic);
            }

            int lblMagic = lblIn.readInt();
            if (lblMagic != 2049) {
                throw new IOException("Invalid label file magic number: " + lblMagic);
            }

            int numImages = imgIn.readInt();
            int numLabels = lblIn.readInt();
            int rows = imgIn.readInt();
            int cols = imgIn.readInt();

            if (numImages != numLabels) {
                throw new IOException("Image count (" + numImages + ") != Label count (" + numLabels + ")");
            }

            int countToLoad = (maxSamples > 0) ? Math.min(maxSamples, numImages) : numImages;
            List<DataSample> samples = new ArrayList<>(countToLoad);

            byte[] pixelBuffer = new byte[rows * cols];

            for (int i = 0; i < countToLoad; i++) {
                imgIn.readFully(pixelBuffer);
                int label = lblIn.readUnsignedByte();

                Tensor imgTensor = new Tensor(1, rows, cols);
                int pIdx = 0;
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < cols; c++) {
                        // Normalize unsigned byte [0, 255] to double [0.0, 1.0]
                        double val = (pixelBuffer[pIdx++] & 0xFF) / 255.0;
                        imgTensor.setValue(0, r, c, val);
                    }
                }

                samples.add(new DataSample(imgTensor, label));
            }

            System.out.printf("Loaded %d samples from %s (%dx%d)%n", countToLoad, imageFile.getName(), rows, cols);
            return samples;
        }
    }

    /**
     * Checks if MNIST files are present; if not, downloads and extracts them.
     */
    public static void ensureDataFilesDownloaded() {
        File dir = new File(DEFAULT_DATA_DIR);
        if (!dir.exists()) {
            dir.mkdirs();
        }

        String[] files = {TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS};
        for (String filename : files) {
            File target = new File(dir, filename);
            if (!target.exists() || target.length() == 0) {
                try {
                    System.out.println("Downloading " + filename + ".gz from official mirror...");
                    downloadAndExtractGz(BASE_URL + filename + ".gz", target);
                    System.out.println("Extracted to " + target.getAbsolutePath());
                } catch (Exception e) {
                    System.err.println("Note: Could not download " + filename + ": " + e.getMessage());
                }
            }
        }
    }

    private static void downloadAndExtractGz(String urlString, File targetFile) throws IOException {
        URL url = URI.create(urlString).toURL();
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestProperty("User-Agent", "Mozilla/5.0");
        conn.setConnectTimeout(8000);
        conn.setReadTimeout(15000);

        try (InputStream in = new GZIPInputStream(conn.getInputStream());
             FileOutputStream out = new FileOutputStream(targetFile)) {
            byte[] buffer = new byte[8192];
            int read;
            while ((read = in.read(buffer)) != -1) {
                out.write(buffer, 0, read);
            }
        }
    }

    /**
     * Generates a high quality synthetic digit dataset (28x28) completely offline.
     * Uses Java 2D graphics with multiple font families, antialiasing, and translations.
     * Guarantees 100% offline verification in seconds.
     */
    public static List<DataSample> createSyntheticDataset(int samplesPerDigit) {
        List<DataSample> samples = new ArrayList<>();
        Random rand = new Random(42);

        String[] fontFamilies = {Font.SANS_SERIF, Font.SERIF, Font.MONOSPACED, "Dialog"};
        int[] fontStyles = {Font.BOLD, Font.PLAIN};

        for (int digit = 0; digit <= 9; digit++) {
            for (int s = 0; s < samplesPerDigit; s++) {
                BufferedImage img = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
                Graphics2D g = img.createGraphics();

                // Background black
                g.setColor(Color.BLACK);
                g.fillRect(0, 0, 28, 28);

                // Foreground white
                g.setColor(Color.WHITE);
                g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

                String fontName = fontFamilies[rand.nextInt(fontFamilies.length)];
                int style = fontStyles[rand.nextInt(fontStyles.length)];
                int fontSize = 18 + rand.nextInt(5); // 18 to 22

                g.setFont(new Font(fontName, style, fontSize));

                int offsetX = 7 + rand.nextInt(5) - 2; // subtle variation
                int offsetY = 20 + rand.nextInt(3) - 1;

                g.drawString(String.valueOf(digit), offsetX, offsetY);
                g.dispose();

                // Convert to Tensor
                Tensor tensor = new Tensor(1, 28, 28);
                for (int r = 0; r < 28; r++) {
                    for (int c = 0; c < 28; c++) {
                        int rgb = img.getRGB(c, r);
                        int gray = rgb & 0xFF; // red/gray value
                        tensor.setValue(0, r, c, gray / 255.0);
                    }
                }

                samples.add(new DataSample(tensor, digit));
            }
        }

        return samples;
    }
}
