package dataset;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import javax.imageio.ImageIO;
import math.Tensor;
import preprocessing.ImagePreprocessor;

/**
 * Loads a folder-organized Devanagari handwritten digit dataset.
 * Expected layout: data/devanagari/{train,test}/{0,1,...,9}/*.(png|jpg|bmp)
 * Folder names are numeric labels; model classes remain 0..9 internally.
 */
public final class DevanagariDatasetLoader {
    private DevanagariDatasetLoader() {}

    public static List<DataSample> loadTrainingData(int maxSamples) throws IOException {
        File csv = new File("data/devanagari-csv/train.csv");
        if (csv.isFile()) return loadCsv(csv, maxSamples);
        return loadSplit(new File("data/devanagari/train"), maxSamples);
    }

    public static List<DataSample> loadTestData(int maxSamples) throws IOException {
        File csv = new File("data/devanagari-csv/test.csv");
        if (csv.isFile()) return loadCsv(csv, maxSamples);
        return loadSplit(new File("data/devanagari/test"), maxSamples);
    }

    /** Reads the public DHDD CSV format: label followed by 1024 grayscale pixels (32x32). */
    private static List<DataSample> loadCsv(File csv, int maxSamples) throws IOException {
        List<DataSample> samples = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(csv), 1024 * 1024)) {
            String line;
            int lineNumber = 0;
            while ((line = reader.readLine()) != null) {
                lineNumber++;
                if (line.trim().isEmpty()) continue;
                String[] fields = line.split(",", -1);
                if (fields.length != 1025) {
                    throw new IOException("Invalid DHDD CSV row " + lineNumber + " in " + csv.getPath()
                        + ": expected 1025 columns (label + 1024 pixels), found " + fields.length);
                }
                int label;
                try { label = Integer.parseInt(fields[0].trim()); }
                catch (NumberFormatException e) { continue; } // Skip a possible header row.
                if (label < 0 || label > 9) throw new IOException("Invalid digit label " + label + " at row " + lineNumber);
                BufferedImage image = new BufferedImage(32, 32, BufferedImage.TYPE_BYTE_GRAY);
                for (int i = 0; i < 1024; i++) {
                    int pixel;
                    try { pixel = Integer.parseInt(fields[i + 1].trim()); }
                    catch (NumberFormatException e) { throw new IOException("Invalid pixel at row " + lineNumber + ", column " + (i + 2), e); }
                    pixel = Math.max(0, Math.min(255, pixel));
                    int rgb = (pixel << 16) | (pixel << 8) | pixel;
                    image.setRGB(i % 32, i / 32, 0xFF000000 | rgb);
                }
                samples.add(new DataSample(toTensor(image), label));
                if (maxSamples > 0 && samples.size() >= maxSamples) break;
            }
        }
        if (samples.isEmpty()) throw new IOException("No valid rows found in " + csv.getPath());
        System.out.printf("Loaded %d Devanagari digit samples from %s%n", samples.size(), csv.getPath());
        return samples;
    }

    public static List<DataSample> loadSplit(File splitDir, int maxSamples) throws IOException {
        if (!splitDir.isDirectory()) {
            throw new IOException("Dataset folder not found: " + splitDir.getPath()
                + "\nCreate data/devanagari/train/0..9 and data/devanagari/test/0..9 folders; add handwritten digit images.");
        }
        List<DataSample> samples = new ArrayList<>();
        String[] extensions = {"png", "jpg", "jpeg", "bmp", "gif"};
        for (int label = 0; label <= 9; label++) {
            File classDir = new File(splitDir, Integer.toString(label));
            if (!classDir.isDirectory()) continue;
            File[] files = classDir.listFiles((dir, name) -> {
                String lower = name.toLowerCase();
                for (String ext : extensions) if (lower.endsWith("." + ext)) return true;
                return false;
            });
            if (files == null) continue;
            java.util.Arrays.sort(files, (a, b) -> a.getName().compareToIgnoreCase(b.getName()));
            for (File file : files) {
                if (maxSamples > 0 && samples.size() >= maxSamples) break;
                BufferedImage image = ImageIO.read(file);
                if (image == null) continue;
                samples.add(new DataSample(toTensor(image), label));
            }
        }
        if (samples.isEmpty()) throw new IOException("No readable images found in " + splitDir.getPath());
        Collections.shuffle(samples, new java.util.Random(20261026L));
        System.out.printf("Loaded %d Devanagari digit images from %s%n", samples.size(), splitDir.getPath());
        return samples;
    }

    private static Tensor toTensor(BufferedImage source) {
        // ImagePreprocessor expects bright strokes on a dark background. Many datasets
        // use dark ink on white paper, so estimate background from the image border and invert.
        BufferedImage normalized = new BufferedImage(source.getWidth(), source.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        long borderSum = 0;
        int borderCount = 0;
        for (int y = 0; y < source.getHeight(); y++) {
            for (int x = 0; x < source.getWidth(); x++) {
                int rgb = source.getRGB(x, y);
                int gray = (int) Math.round(0.299 * ((rgb >> 16) & 255)
                    + 0.587 * ((rgb >> 8) & 255) + 0.114 * (rgb & 255));
                if (x == 0 || y == 0 || x == source.getWidth() - 1 || y == source.getHeight() - 1) {
                    borderSum += gray;
                    borderCount++;
                }
            }
        }
        boolean lightBackground = borderCount > 0 && borderSum / borderCount > 127;
        for (int y = 0; y < source.getHeight(); y++) {
            for (int x = 0; x < source.getWidth(); x++) {
                int rgb = source.getRGB(x, y);
                int gray = (int) Math.round(0.299 * ((rgb >> 16) & 255)
                    + 0.587 * ((rgb >> 8) & 255) + 0.114 * (rgb & 255));
                if (lightBackground) gray = 255 - gray;
                normalized.setRGB(x, y, new Color(gray, gray, gray).getRGB());
            }
        }
        return ImagePreprocessor.preprocess(normalized);
    }
}
