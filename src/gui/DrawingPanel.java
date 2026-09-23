package gui;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.RenderingHints;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.image.BufferedImage;
import javax.swing.JPanel;

/**
 * DrawingPanel provides an interactive drawing canvas for the user to sketch digits.
 * Features antialiased strokes, offscreen buffering, and drawing change events.
 */
public class DrawingPanel extends JPanel {

    public static final int CANVAS_SIZE = 280;
    private static final int BRUSH_RADIUS = 22;

    private BufferedImage canvasImage;
    private Graphics2D g2d;
    private Point lastPoint;
    private boolean hasDrawn = false;
    private Runnable onDrawingChanged;

    public DrawingPanel() {
        setPreferredSize(new Dimension(CANVAS_SIZE, CANVAS_SIZE));
        setBackground(Color.BLACK);

        canvasImage = new BufferedImage(CANVAS_SIZE, CANVAS_SIZE, BufferedImage.TYPE_INT_RGB);
        g2d = canvasImage.createGraphics();
        setupGraphics();
        clear();

        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                lastPoint = e.getPoint();
                drawCircle(lastPoint.x, lastPoint.y);
                hasDrawn = true;
                repaint();
                if (onDrawingChanged != null) {
                    onDrawingChanged.run();
                }
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                lastPoint = null;
                if (onDrawingChanged != null) {
                    onDrawingChanged.run();
                }
            }
        });

        addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                Point currentPoint = e.getPoint();
                if (lastPoint != null) {
                    drawLine(lastPoint.x, lastPoint.y, currentPoint.x, currentPoint.y);
                } else {
                    drawCircle(currentPoint.x, currentPoint.y);
                }
                lastPoint = currentPoint;
                hasDrawn = true;
                repaint();
                if (onDrawingChanged != null) {
                    onDrawingChanged.run();
                }
            }
        });
    }

    private void setupGraphics() {
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2d.setRenderingHint(RenderingHints.KEY_STROKE_CONTROL, RenderingHints.VALUE_STROKE_PURE);
        g2d.setColor(Color.WHITE);
        g2d.setStroke(new BasicStroke(BRUSH_RADIUS, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
    }

    private void drawCircle(int x, int y) {
        g2d.setColor(Color.WHITE);
        g2d.fillOval(x - BRUSH_RADIUS / 2, y - BRUSH_RADIUS / 2, BRUSH_RADIUS, BRUSH_RADIUS);
    }

    private void drawLine(int x1, int y1, int x2, int y2) {
        g2d.setColor(Color.WHITE);
        g2d.drawLine(x1, y1, x2, y2);
    }

    public void clear() {
        g2d.setColor(Color.BLACK);
        g2d.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
        g2d.setColor(Color.WHITE);
        hasDrawn = false;
        repaint();
        if (onDrawingChanged != null) {
            onDrawingChanged.run();
        }
    }

    public BufferedImage getImage() {
        return canvasImage;
    }

    public boolean hasDrawn() {
        return hasDrawn;
    }

    public void setOnDrawingChanged(Runnable listener) {
        this.onDrawingChanged = listener;
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        if (canvasImage != null) {
            g.drawImage(canvasImage, 0, 0, null);
        }
    }
}
