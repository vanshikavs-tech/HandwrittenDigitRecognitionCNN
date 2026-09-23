package gui;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.GridLayout;
import java.awt.Insets;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.List;
import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.SwingConstants;
import javax.swing.SwingUtilities;
import javax.swing.SwingWorker;
import javax.swing.UIManager;
import dataset.DataSample;
import dataset.MNISTLoader;
import math.Tensor;
import model.ModelSerializer;
import network.NeuralNetwork;
import preprocessing.ImagePreprocessor;
import trainer.Trainer;
import utils.ModelBuilder;

/**
 * MainWindow is the primary graphical user interface for the Handwritten Digit Recognition System.
 * Integrates:
 *   1. Interactive drawing canvas
 *   2. Live 28x28 preprocessed tensor preview
 *   3. Real-time digit prediction and confidence scoring
 *   4. 10-class probability distribution bar chart
 *   5. Training controls, model persistence, and status feedback
 */
public class MainWindow extends JFrame {

    private NeuralNetwork network;
    private DrawingPanel drawingPanel;
    private PreviewPanel previewPanel;

    private JLabel digitLabel;
    private JLabel confidenceLabel;
    private JLabel statusLabel;

    private JProgressBar[] probBars;
    private JLabel[] probLabels;

    private JCheckBox livePredictionCheck;
    private JButton predictButton;
    private JButton clearButton;
    private JButton trainDemoButton;
    private JButton trainMnistButton;
    private JButton saveButton;
    private JButton loadButton;

    public MainWindow(NeuralNetwork initialNetwork) {
        this.network = initialNetwork != null ? initialNetwork : ModelBuilder.buildCNN();
        initUI();
    }

    private void initUI() {
        setTitle("Handwritten Digit Recognition CNN - Java AOOP Project");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout(12, 12));
        getContentPane().setBackground(new Color(245, 247, 250));

        // 1. Header Banner
        JPanel headerPanel = createHeaderPanel();
        add(headerPanel, BorderLayout.NORTH);

        // 2. Central Content Panel (Left: Draw, Center: Result & Preview, Right: Probability Bars)
        JPanel contentPanel = new JPanel(new GridLayout(1, 3, 16, 16));
        contentPanel.setBackground(new Color(245, 247, 250));
        contentPanel.setBorder(BorderFactory.createEmptyBorder(10, 16, 10, 16));

        JPanel leftPanel = createLeftDrawingPanel();
        JPanel centerPanel = createCenterResultPanel();
        JPanel rightPanel = createRightProbabilitiesPanel();

        contentPanel.add(leftPanel);
        contentPanel.add(centerPanel);
        contentPanel.add(rightPanel);
        add(contentPanel, BorderLayout.CENTER);

        // 3. Footer Toolbar
        JPanel footerPanel = createFooterPanel();
        add(footerPanel, BorderLayout.SOUTH);

        pack();
        setLocationRelativeTo(null);
        setResizable(false);
    }

    private JPanel createHeaderPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setBackground(new Color(30, 41, 59));
        panel.setBorder(BorderFactory.createEmptyBorder(12, 20, 12, 20));

        JLabel titleLabel = new JLabel("HANDWRITTEN DIGIT RECOGNITION CNN");
        titleLabel.setFont(new Font("Segoe UI", Font.BOLD, 18));
        titleLabel.setForeground(Color.WHITE);

        JLabel subtitleLabel = new JLabel("Built from Scratch in Pure Java | AOOP Final Project");
        subtitleLabel.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        subtitleLabel.setForeground(new Color(203, 213, 225));

        panel.add(titleLabel, BorderLayout.WEST);
        panel.add(subtitleLabel, BorderLayout.EAST);
        return panel;
    }

    private JPanel createLeftDrawingPanel() {
        JPanel card = createCardPanel("1. DRAW DIGIT HERE");
        card.setLayout(new BoxLayout(card, BoxLayout.Y_AXIS));

        drawingPanel = new DrawingPanel();
        drawingPanel.setAlignmentX(CENTER_ALIGNMENT);
        drawingPanel.setBorder(BorderFactory.createLineBorder(new Color(71, 85, 105), 2));
        drawingPanel.setOnDrawingChanged(() -> {
            if (livePredictionCheck.isSelected() && drawingPanel.hasDrawn()) {
                performPrediction();
            }
        });

        JPanel btnPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 10, 8));
        btnPanel.setOpaque(false);

        clearButton = new JButton("Clear Canvas");
        clearButton.setFont(new Font("Segoe UI", Font.BOLD, 12));
        clearButton.addActionListener(e -> {
            drawingPanel.clear();
            previewPanel.updateTensor(null);
            resetPredictionDisplay();
        });

        predictButton = new JButton("Predict");
        predictButton.setFont(new Font("Segoe UI", Font.BOLD, 12));
        predictButton.setBackground(new Color(37, 99, 235));
        predictButton.setForeground(Color.WHITE);
        predictButton.addActionListener(e -> performPrediction());

        btnPanel.add(clearButton);
        btnPanel.add(predictButton);

        livePredictionCheck = new JCheckBox("Live Real-Time Prediction", true);
        livePredictionCheck.setOpaque(false);
        livePredictionCheck.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        livePredictionCheck.setAlignmentX(CENTER_ALIGNMENT);

        card.add(Box.createVerticalStrut(5));
        card.add(drawingPanel);
        card.add(Box.createVerticalStrut(8));
        card.add(btnPanel);
        card.add(livePredictionCheck);
        card.add(Box.createVerticalGlue());

        return card;
    }

    private JPanel createCenterResultPanel() {
        JPanel card = createCardPanel("2. CNN PREDICTION");
        card.setLayout(new BoxLayout(card, BoxLayout.Y_AXIS));

        // Result display
        JPanel resultBox = new JPanel();
        resultBox.setLayout(new BoxLayout(resultBox, BoxLayout.Y_AXIS));
        resultBox.setOpaque(false);
        resultBox.setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createLineBorder(new Color(226, 232, 240), 1),
            BorderFactory.createEmptyBorder(12, 12, 12, 12)
        ));

        JLabel title = new JLabel("PREDICTED DIGIT", SwingConstants.CENTER);
        title.setFont(new Font("Segoe UI", Font.BOLD, 13));
        title.setForeground(new Color(100, 116, 139));
        title.setAlignmentX(CENTER_ALIGNMENT);

        digitLabel = new JLabel("-", SwingConstants.CENTER);
        digitLabel.setFont(new Font("Segoe UI", Font.BOLD, 68));
        digitLabel.setForeground(new Color(15, 23, 42));
        digitLabel.setAlignmentX(CENTER_ALIGNMENT);

        confidenceLabel = new JLabel("Confidence: - %", SwingConstants.CENTER);
        confidenceLabel.setFont(new Font("Segoe UI", Font.BOLD, 14));
        confidenceLabel.setForeground(new Color(71, 85, 105));
        confidenceLabel.setAlignmentX(CENTER_ALIGNMENT);

        resultBox.add(title);
        resultBox.add(Box.createVerticalStrut(4));
        resultBox.add(digitLabel);
        resultBox.add(Box.createVerticalStrut(4));
        resultBox.add(confidenceLabel);

        // Preprocessed 28x28 Preview
        JLabel previewTitle = new JLabel("Preprocessed 28x28 Tensor Input:", SwingConstants.CENTER);
        previewTitle.setFont(new Font("Segoe UI", Font.BOLD, 12));
        previewTitle.setForeground(new Color(71, 85, 105));
        previewTitle.setAlignmentX(CENTER_ALIGNMENT);

        previewPanel = new PreviewPanel();
        previewPanel.setAlignmentX(CENTER_ALIGNMENT);
        previewPanel.setPreferredSize(new Dimension(140, 140));
        previewPanel.setMaximumSize(new Dimension(140, 140));
        previewPanel.setBorder(BorderFactory.createLineBorder(new Color(148, 163, 184), 1));

        card.add(Box.createVerticalStrut(5));
        card.add(resultBox);
        card.add(Box.createVerticalStrut(14));
        card.add(previewTitle);
        card.add(Box.createVerticalStrut(6));
        card.add(previewPanel);
        card.add(Box.createVerticalGlue());

        return card;
    }

    private JPanel createRightProbabilitiesPanel() {
        JPanel card = createCardPanel("3. PROBABILITY DISTRIBUTION");
        card.setLayout(new GridBagLayout());

        probBars = new JProgressBar[10];
        probLabels = new JLabel[10];

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.insets = new Insets(3, 4, 3, 4);

        for (int i = 0; i < 10; i++) {
            gbc.gridx = 0;
            gbc.gridy = i;
            gbc.weightx = 0.15;
            JLabel digitName = new JLabel("Digit " + i + ":");
            digitName.setFont(new Font("Segoe UI", Font.BOLD, 12));
            digitName.setForeground(new Color(30, 41, 59));
            card.add(digitName, gbc);

            gbc.gridx = 1;
            gbc.weightx = 0.65;
            JProgressBar bar = new JProgressBar(0, 100);
            bar.setValue(0);
            bar.setStringPainted(false);
            bar.setPreferredSize(new Dimension(110, 16));
            bar.setForeground(new Color(59, 130, 246));
            probBars[i] = bar;
            card.add(bar, gbc);

            gbc.gridx = 2;
            gbc.weightx = 0.20;
            JLabel valLabel = new JLabel(" 0.0%");
            valLabel.setFont(new Font("Consolas", Font.PLAIN, 12));
            valLabel.setForeground(new Color(71, 85, 105));
            probLabels[i] = valLabel;
            card.add(valLabel, gbc);
        }

        return card;
    }

    private JPanel createFooterPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setBackground(Color.WHITE);
        panel.setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createMatteBorder(1, 0, 0, 0, new Color(226, 232, 240)),
            BorderFactory.createEmptyBorder(8, 16, 8, 16)
        ));

        statusLabel = new JLabel("Status: Ready | Network Loaded");
        statusLabel.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        statusLabel.setForeground(new Color(71, 85, 105));

        JPanel actions = new JPanel(new FlowLayout(FlowLayout.RIGHT, 8, 0));
        actions.setOpaque(false);

        trainDemoButton = new JButton("Train Quick Demo");
        trainDemoButton.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        trainDemoButton.addActionListener(e -> trainQuickDemo());

        trainMnistButton = new JButton("Train on MNIST");
        trainMnistButton.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        trainMnistButton.addActionListener(e -> trainOnMnist());

        saveButton = new JButton("Save Model");
        saveButton.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        saveButton.addActionListener(e -> saveModelToFile());

        loadButton = new JButton("Load Model");
        loadButton.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        loadButton.addActionListener(e -> loadModelFromFile());

        actions.add(trainDemoButton);
        actions.add(trainMnistButton);
        actions.add(saveButton);
        actions.add(loadButton);

        panel.add(statusLabel, BorderLayout.WEST);
        panel.add(actions, BorderLayout.EAST);
        return panel;
    }

    private JPanel createCardPanel(String title) {
        JPanel panel = new JPanel();
        panel.setBackground(Color.WHITE);
        panel.setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createLineBorder(new Color(226, 232, 240), 1),
            BorderFactory.createCompoundBorder(
                BorderFactory.createTitledBorder(
                    BorderFactory.createEmptyBorder(),
                    title,
                    0, 0,
                    new Font("Segoe UI", Font.BOLD, 12),
                    new Color(71, 85, 105)
                ),
                BorderFactory.createEmptyBorder(6, 10, 10, 10)
            )
        ));
        return panel;
    }

    private void performPrediction() {
        if (!drawingPanel.hasDrawn()) {
            return;
        }

        BufferedImage canvasImg = drawingPanel.getImage();
        Tensor inputTensor = ImagePreprocessor.preprocess(canvasImg);
        previewPanel.updateTensor(inputTensor);

        double[] probs = network.predictProbabilities(inputTensor);
        int predictedDigit = 0;
        double maxProb = -1.0;

        for (int i = 0; i < probs.length; i++) {
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                predictedDigit = i;
            }
        }

        digitLabel.setText(String.valueOf(predictedDigit));
        confidenceLabel.setText(String.format("Confidence: %.1f%%", maxProb * 100.0));

        // Update probability bars
        for (int i = 0; i < 10; i++) {
            int pct = (int) Math.round(probs[i] * 100.0);
            probBars[i].setValue(pct);
            probLabels[i].setText(String.format("%5.1f%%", probs[i] * 100.0));

            if (i == predictedDigit) {
                probBars[i].setForeground(new Color(16, 185, 129)); // Emerald Green for top
                probLabels[i].setFont(new Font("Consolas", Font.BOLD, 12));
                probLabels[i].setForeground(new Color(5, 150, 105));
            } else {
                probBars[i].setForeground(new Color(59, 130, 246)); // Blue
                probLabels[i].setFont(new Font("Consolas", Font.PLAIN, 12));
                probLabels[i].setForeground(new Color(100, 116, 139));
            }
        }
    }

    private void resetPredictionDisplay() {
        digitLabel.setText("-");
        confidenceLabel.setText("Confidence: - %");
        for (int i = 0; i < 10; i++) {
            probBars[i].setValue(0);
            probBars[i].setForeground(new Color(59, 130, 246));
            probLabels[i].setText(" 0.0%");
            probLabels[i].setFont(new Font("Consolas", Font.PLAIN, 12));
            probLabels[i].setForeground(new Color(100, 116, 139));
        }
    }

    private void trainQuickDemo() {
        setControlsEnabled(false);
        statusLabel.setText("Status: Generating synthetic digits and training demo CNN...");

        SwingWorker<Void, String> worker = new SwingWorker<>() {
            @Override
            protected Void doInBackground() {
                List<DataSample> dataset = MNISTLoader.createSyntheticDataset(40); // 400 samples
                Trainer trainer = new Trainer(network);
                trainer.train(dataset, 6, 0.02, 16, null);
                return null;
            }

            @Override
            protected void done() {
                setControlsEnabled(true);
                statusLabel.setText("Status: Quick Demo Model trained successfully!");
                JOptionPane.showMessageDialog(MainWindow.this,
                    "Demo CNN successfully trained on synthetic digits!\nDraw digits on the canvas to test predictions.",
                    "Training Complete", JOptionPane.INFORMATION_MESSAGE);
                if (drawingPanel.hasDrawn()) {
                    performPrediction();
                }
            }
        };
        worker.execute();
    }

    private void trainOnMnist() {
        String input = JOptionPane.showInputDialog(
            this,
            "Enter number of MNIST samples to train on (e.g. 1000 for quick training, or 5000):",
            "1000"
        );
        if (input == null || input.trim().isEmpty()) return;

        int samples;
        try {
            samples = Integer.parseInt(input.trim());
        } catch (NumberFormatException e) {
            JOptionPane.showMessageDialog(this, "Invalid number.", "Error", JOptionPane.ERROR_MESSAGE);
            return;
        }

        setControlsEnabled(false);
        statusLabel.setText("Status: Loading MNIST and training CNN (please wait)...");

        SwingWorker<Void, Void> worker = new SwingWorker<>() {
            @Override
            protected Void doInBackground() throws Exception {
                List<DataSample> trainData = MNISTLoader.loadTrainingData(samples);
                Trainer trainer = new Trainer(network);
                trainer.train(trainData, 5, 0.01, 32, null);
                return null;
            }

            @Override
            protected void done() {
                setControlsEnabled(true);
                try {
                    get();
                    statusLabel.setText("Status: Model trained on " + samples + " MNIST samples!");
                    JOptionPane.showMessageDialog(MainWindow.this,
                        "Successfully trained on " + samples + " MNIST images!",
                        "Training Complete", JOptionPane.INFORMATION_MESSAGE);
                } catch (Exception e) {
                    statusLabel.setText("Status: Training failed: " + e.getMessage());
                    JOptionPane.showMessageDialog(MainWindow.this,
                        "Failed to train on MNIST: " + e.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
                }
            }
        };
        worker.execute();
    }

    private void saveModelToFile() {
        try {
            ModelSerializer.saveModel(network, ModelSerializer.DEFAULT_MODEL_PATH);
            statusLabel.setText("Status: Model saved to " + ModelSerializer.DEFAULT_MODEL_PATH);
            JOptionPane.showMessageDialog(this, "Model saved successfully!", "Saved", JOptionPane.INFORMATION_MESSAGE);
        } catch (Exception e) {
            JOptionPane.showMessageDialog(this, "Save error: " + e.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
        }
    }

    private void loadModelFromFile() {
        JFileChooser chooser = new JFileChooser(new File("models"));
        int res = chooser.showOpenDialog(this);
        if (res == JFileChooser.APPROVE_OPTION) {
            try {
                this.network = ModelSerializer.loadModel(chooser.getSelectedFile().getAbsolutePath());
                statusLabel.setText("Status: Model loaded from " + chooser.getSelectedFile().getName());
                resetPredictionDisplay();
                if (drawingPanel.hasDrawn()) {
                    performPrediction();
                }
            } catch (Exception e) {
                JOptionPane.showMessageDialog(this, "Load error: " + e.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
            }
        }
    }

    private void setControlsEnabled(boolean enabled) {
        trainDemoButton.setEnabled(enabled);
        trainMnistButton.setEnabled(enabled);
        predictButton.setEnabled(enabled);
        clearButton.setEnabled(enabled);
        saveButton.setEnabled(enabled);
        loadButton.setEnabled(enabled);
    }

    /**
     * PreviewPanel renders the centered 28x28 grayscale Tensor.
     */
    private static class PreviewPanel extends JPanel {
        private Tensor tensor;

        public PreviewPanel() {
            setBackground(Color.BLACK);
        }

        public void updateTensor(Tensor tensor) {
            this.tensor = tensor;
            repaint();
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            if (tensor == null) {
                g.setColor(Color.DARK_GRAY);
                g.drawString("No drawing", 35, 75);
                return;
            }

            int w = getWidth();
            int h = getHeight();
            int rows = tensor.getHeight();
            int cols = tensor.getWidth();

            double cellW = (double) w / cols;
            double cellH = (double) h / rows;

            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    double val = tensor.getValue(0, r, c);
                    int gray = Math.min(255, Math.max(0, (int) (val * 255)));
                    g.setColor(new Color(gray, gray, gray));
                    g.fillRect((int) (c * cellW), (int) (r * cellH), (int) Math.ceil(cellW), (int) Math.ceil(cellH));
                }
            }
        }
    }
}
