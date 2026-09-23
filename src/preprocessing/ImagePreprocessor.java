package preprocessing;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import math.Tensor;

/**
 * ImagePreprocessor converts raw user drawings into MNIST-compliant 28x28 tensors.
 * Performs:
 *   1. Bounding-box detection
 *   2. Aspect-ratio preserving scaling into a 20x20 bounding box
 *   3. Center-of-mass centering onto a 28x28 canvas (exact MNIST standard)
 *   4. Pixel value normalization to [0.0, 1.0]
 */
public class ImagePreprocessor {

    private static final int TARGET_SIZE = 28;
    private static final int INNER_BOUND = 20;

    /**
     * Preprocesses an input BufferedImage into a 1x28x28 Tensor.
     */
    public static Tensor preprocess(BufferedImage inputImage) {
        int width = inputImage.getWidth();
        int height = inputImage.getHeight();

        // 1. Convert to grayscale luminance array
        double[][] gray = new double[height][width];
        int minX = width, maxX = -1, minY = height, maxY = -1;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = inputImage.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                double lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
                gray[y][x] = lum;

                // Threshold detection for drawn strokes
                if (lum > 0.08) {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
        }

        // If canvas is blank, return empty 28x28 tensor
        if (maxX < minX || maxY < minY) {
            return Tensor.zeros(1, TARGET_SIZE, TARGET_SIZE);
        }

        int boxW = maxX - minX + 1;
        int boxH = maxY - minY + 1;

        // 2. Extract bounding box into a cropped sub-image
        BufferedImage cropped = new BufferedImage(boxW, boxH, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < boxH; y++) {
            for (int x = 0; x < boxW; x++) {
                int val = (int) (gray[minY + y][minX + x] * 255);
                int rgb = (val << 16) | (val << 8) | val;
                cropped.setRGB(x, y, rgb);
            }
        }

        // 3. Scale bounding box proportionally to fit within 20x20
        int scaledW, scaledH;
        if (boxW > boxH) {
            scaledW = INNER_BOUND;
            scaledH = Math.max(1, (int) Math.round((double) boxH / boxW * INNER_BOUND));
        } else {
            scaledH = INNER_BOUND;
            scaledW = Math.max(1, (int) Math.round((double) boxW / boxH * INNER_BOUND));
        }

        Image scaledImg = cropped.getScaledInstance(scaledW, scaledH, Image.SCALE_SMOOTH);
        BufferedImage scaledBuffered = new BufferedImage(scaledW, scaledH, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g2d = scaledBuffered.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g2d.drawImage(scaledImg, 0, 0, null);
        g2d.dispose();

        // 4. Place onto a 28x28 canvas centered by center of mass
        double[][] scaledData = new double[scaledH][scaledW];
        double totalMass = 0.0;
        double massCenterX = 0.0;
        double massCenterY = 0.0;

        for (int y = 0; y < scaledH; y++) {
            for (int x = 0; x < scaledW; x++) {
                double val = (scaledBuffered.getRGB(x, y) & 0xFF) / 255.0;
                scaledData[y][x] = val;
                totalMass += val;
                massCenterX += x * val;
                massCenterY += y * val;
            }
        }

        if (totalMass > 0) {
            massCenterX /= totalMass;
            massCenterY /= totalMass;
        } else {
            massCenterX = scaledW / 2.0;
            massCenterY = scaledH / 2.0;
        }

        // Center of 28x28 canvas is (13.5, 13.5) or approx 14
        int targetCenter = TARGET_SIZE / 2;
        int shiftX = (int) Math.round(targetCenter - massCenterX);
        int shiftY = (int) Math.round(targetCenter - massCenterY);

        Tensor finalTensor = Tensor.zeros(1, TARGET_SIZE, TARGET_SIZE);

        for (int y = 0; y < scaledH; y++) {
            int targetY = y + shiftY;
            if (targetY < 0 || targetY >= TARGET_SIZE) continue;

            for (int x = 0; x < scaledW; x++) {
                int targetX = x + shiftX;
                if (targetX < 0 || targetX >= TARGET_SIZE) continue;

                finalTensor.setValue(0, targetY, targetX, scaledData[y][x]);
            }
        }

        return finalTensor;
    }
}
