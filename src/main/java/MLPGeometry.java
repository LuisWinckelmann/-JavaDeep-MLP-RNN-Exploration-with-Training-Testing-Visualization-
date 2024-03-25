import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;

import javax.imageio.ImageIO;
import javax.swing.JFrame;
import javax.swing.JPanel;

import misc.BasicLearningListener;
import misc.LearningListener;

public class MLPGeometry {
    
    public static final Color CLASS_1 = new Color(255, 255, 255);
    public static final Color CLASS_2 = new Color(130, 130, 130);
    
    public static double sq(final double x) {
        return x * x;
    }
    
    public static double[][][] generateData(
            final int samples
    ) {
        final double[][] input  = new double[samples][2];
        final double[][] target = new double[samples][1];
        // use internally seed random number generator to ensure reproducibility
        final Random rnd = new Random(100L);

        final double c1x = 0.3;
        final double c1y = 0.3;
        final double r1  = 0.2;
        final double c2x = 0.7;
        final double c2y = 0.7;
        final double r2  = 0.2;
        final double rel = 0.5;
        int size1 = (int)(samples * rel);
        int size2 = samples - size1;
        int size = size1 + size2;
        int i = 0;
        while (size > 0) {
            final double x = rnd.nextDouble();
            final double y = rnd.nextDouble();
            final double l1 = Math.sqrt(sq(c1x - x) + sq(c1y - y));
            final double l2 = Math.sqrt(sq(c2x - x) + sq(c2y - y));
            if (l1 > r1 && l2 > r2) {
                // outer point
                if (size2 > 0) {
                    input[i][0]  = x;
                    input[i][1]  = y;
                    target[i][0] = 0.0;
                    size2--;
                }
            } else {
                // inner point
                if (size1 > 0) {
                    input[i][0]  = x;
                    input[i][1]  = y;
                    target[i][0] = 1.0;
                    size1--;
                }
            }
            i++;
            size--;
        }
        return new double[][][]{input, target};
    }
    
    
    private static void drawSamples(final Graphics gfx, final int w, final int h, double[][] input, double[][] target) {
        final Graphics2D g = (Graphics2D)gfx;
        g.setRenderingHint(
            RenderingHints.KEY_ANTIALIASING,
            RenderingHints.VALUE_ANTIALIAS_ON
        );
        for (int i = 0; i < input.length; i++) {
            int x = (int)((double)(w - 1) *  
                    input[i][0]);
            int y = (int)((double)(h - 1) * 
                    input[i][1]);
            if (target[i][0] > 0.5) {
                g.setColor(CLASS_1);
            } else {
                g.setColor(CLASS_2);
            }
            g.fillOval(x-1, y-1, 3, 3);
        }
    }
    
    public static void main(String[] args) {
        final double[][][] samples = generateData(5000);
        final double[][] input  = samples[0];
        final double[][] target = samples[1];
        final Random rnd = new Random(10000L);
        // set up network (biases are used by default).
        //final MultiLayerPerceptron net = new MultiLayerPerceptron(2, 3,1);
        final MultiLayerPerceptron net = new MultiLayerPerceptron(2, 256,128, 1);
        // System.out.println(net.getWeightsNum());
        // perform training.
        final int epochs = 2000;
        // 0.012:
        final double learningrate = 0.1;
        final double momentumrate = 0.8;
        // generate initial weights.
        net.initializeWeights(rnd, 0.1);
        // setup visualization.
        final String caption = "Non-linear classification problem";
        final JFrame frame = new JFrame(caption);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        final BufferedImage img = new BufferedImage(600, 600, BufferedImage.TYPE_INT_RGB);
        drawSamples(img.getGraphics(), img.getWidth(), img.getHeight(), input, target);
        final JPanel panel = new JPanel() {
            private static final long serialVersionUID = -4307908552010057652L;
            @Override
            protected void paintComponent(Graphics gfx) {
                super.paintComponent(gfx);
                gfx.drawImage(
                    img,  0,  0, 
                    img.getWidth(),  img.getHeight(),  null
                );
            }
        };
        panel.setPreferredSize(new Dimension(img.getWidth(), img.getHeight()));
        frame.add(panel);
        frame.setResizable(false);
        frame.pack();
        frame.setVisible(true);

        final LearningListener listener = new BasicLearningListener() {
            @Override
            public void afterEpoch(int epoch, double trainingerror) {
                super.afterEpoch(epoch, trainingerror);

                final int ep = epoch + 1;
                // Only visualize each every step_size frame
                final int vis_step_size = 5;
                if (ep % vis_step_size != 0 && ep != 1) return;

                final double[] p = new double[2];
                for (int y = 0; y < img.getHeight(); y++) {
                    for (int x = 0; x < img.getWidth(); x++) {
                        p[0] = ((double)x) / ((double)(img.getWidth()));
                        p[1] = ((double)y) / ((double)(img.getHeight()));

                        final double[] o = net.forwardPass(p);
                        final double v = o[0];

                        int vi = ((int)(v * 255));
                        img.setRGB(x, y, ((vi >> 1) << 8) | vi);
                    }
                }
                drawSamples(img.getGraphics(), img.getWidth(), img.getHeight(), input, target);
                double rounded_train_error =  Math.round(trainingerror * 1000.0) / 1000.0;
                try {
                    ImageIO.write(img,"png", new File("./gfx/MLPGeometry/" + String.valueOf(ep)+ ".png" ));
                } catch (IOException e) {
                    e.printStackTrace();
                }
                frame.setTitle(caption + " after epoch " + ep + ", training-error: " + rounded_train_error);
                frame.repaint();
            }
        };
        // perform training.
        net.trainStochastic(
            rnd, 
            input,
            target,
            epochs,
            learningrate,
            momentumrate,
            listener
        );
    }
}
