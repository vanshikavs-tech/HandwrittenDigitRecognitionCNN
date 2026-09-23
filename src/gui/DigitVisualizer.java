package gui;

import math.Tensor;

/**
 * DigitVisualizer renders 28x28 grayscale image tensors into clean ASCII art
 * for direct visualization in terminal consoles and test outputs.
 */
public class DigitVisualizer {

    private static final char[] ASCII_SHADES = {' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'};

    /**
     * Prints a 28x28 single-channel Tensor to the console using ASCII shading.
     */
    public static void printAscii(Tensor tensor) {
        if (tensor.getHeight() != 28 || tensor.getWidth() != 28) {
            System.out.println("Tensor size: " + tensor.getHeight() + "x" + tensor.getWidth());
        }

        System.out.println("+----------------------------+");
        for (int r = 0; r < tensor.getHeight(); r++) {
            StringBuilder sb = new StringBuilder("|");
            for (int c = 0; c < tensor.getWidth(); c++) {
                double val = tensor.getValue(0, r, c);
                int idx = (int) Math.round(val * (ASCII_SHADES.length - 1));
                idx = Math.max(0, Math.min(ASCII_SHADES.length - 1, idx));
                sb.append(ASCII_SHADES[idx]);
            }
            sb.append("|");
            System.out.println(sb.toString());
        }
        System.out.println("+----------------------------+");
    }
}
