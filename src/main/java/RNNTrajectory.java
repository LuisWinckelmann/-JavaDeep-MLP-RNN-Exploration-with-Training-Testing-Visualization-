import misc.BasicLearningListener;
import misc.Spiral;
import misc.TrajectoryGenerator;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.Serial;
import java.util.Random;

public class RNNTrajectory {
    
    private static final Random rnd = new Random(100L);

    public static double[][] generateTrajectory(
        final TrajectoryGenerator gen,
        final int timesteps
    ) {
        final double[][] result = new double[timesteps][];
        gen.reset();
        for (int t = 0; t < timesteps; t++) {
            result[t] = gen.next();
        }
        return result;
    }

    public static void main(String[] args) {
        final TrajectoryGenerator gen = new Spiral();
        final int trainlength  = 200; // 100 originally
        final double[][] trainseq = generateTrajectory(
            gen, trainlength
        );
        final double[][][] inputs  = new double[1][trainlength][1];
        final double[][][] targets = new double[][][]{trainseq};
        // set only the first value of the input sequence to 1.0.
        inputs[0][0][0] = 1.0;
        // set up network. biases are used by default, but can be deactivated using net.setBias(layer, false),
        // where layer gives the layer index (1 = is the first hidden layer).
        // works:
        // final RecurrentNeuralNetwork net = new RecurrentNeuralNetwork(1, 150,150, 2);
        final RecurrentNeuralNetwork net = new RecurrentNeuralNetwork(1, 250,250, 2);
        // we disable all biases to test if the implementation really works and isn't solved purely by the bias
        boolean flag = false;
        net.setBias(1, flag);
        net.setBias(2, flag);
        net.setBias(3, flag);
        // Flag to save visualizations to file
        final boolean save_vis  = false;
        // perform training.
        final int epochs = 25000; //
        final double learningrate = 0.000005; //
        final double momentumrate = 0.9; //
        // generate initial weights and prepare the RNN buffer for BPTT over the required number of time steps.
        net.initializeWeights(rnd, 0.1);
        net.rebufferOnDemand(trainlength);
        // start the training.
        final double error = net.trainStochastic(
            rnd, 
            inputs,
            targets,
            epochs,
            learningrate,
            momentumrate,
            new BasicLearningListener()
        );
        System.out.println();
        System.out.println("final error: " + error);
        // for online usage of the trained RNN, we do not need any time buffer -> we can thus reduce the buffer length to 1 time step.
        net.rebufferOnDemand(1);
        net.reset();
        gen.reset();
        final BufferedImage img = new BufferedImage(
            800, 700, BufferedImage.TYPE_INT_ARGB
        );
        final Graphics2D imggfx = (Graphics2D)img.getGraphics();
        imggfx.setRenderingHint(
            RenderingHints.KEY_ANTIALIASING,
            RenderingHints.VALUE_ANTIALIAS_ON
        );
        final Stroke stroke = new BasicStroke(1.5f);
        imggfx.setStroke(stroke);
        imggfx.setBackground(Color.WHITE);
        final JPanel canvas = new JPanel(){
            @Serial
            private static final long serialVersionUID = -5927396264951085674L;
            @Override
            protected void paintComponent(Graphics gfx) {
                super.paintComponent(gfx);
                gfx.drawImage(img, 0, 0, null);
            }
        };
        final Dimension canvasdim = new Dimension(
            img.getWidth(), img.getHeight()
        );
        canvas.setPreferredSize(canvasdim);
        canvas.setSize(canvasdim);

        imggfx.clearRect(0, 0, img.getWidth(), img.getHeight());
        final int[] timestep = {0};

        final JFrame frame = new JFrame("Trajectory generation.");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                super.mousePressed(e);
                synchronized (gen) {
                    gen.reset();
                    net.reset();
                    //
                    imggfx.clearRect(0, 0, img.getWidth(), img.getHeight());
                    timestep[0] = 0;
                }
            }
        });
        frame.add(canvas);
        frame.pack();
        frame.setVisible(true);
        // setup render timer.
        final Timer timer = new Timer((1000 / 30), new ActionListener() {
            private double lastgenx;
            private double lastgeny;
            private double lastnetx;
            private double lastnety;

            final Color colorgen = new Color(20, 80, 220);
            final Color colornet = new Color(220, 80, 20);

            @Override
            public void actionPerformed(ActionEvent e) {

                synchronized (gen) {
                    double[] netout;
                    if (timestep[0] == 0) {
                        netout = net.forwardPass(new double[]{1.0});
                    } else {
                        netout = net.forwardPass(new double[]{0.0});
                    }
                    final double[] genout = gen.next();

                    final double netx = netout[0];
                    final double nety = netout[1];
                    final double genx = genout[0];
                    final double geny = genout[1];

                    if (timestep[0] > 0) {

                        final int w = img.getWidth();
                        final int h = img.getHeight();
                        final int mx = w / 2;
                        final int my = h / 2;

                        final int gx1 = mx + (int)(330 * this.lastgenx);
                        final int gy1 = my - (int)(330 * this.lastgeny);
                        final int gx2 = mx + (int)(330 * genx);
                        final int gy2 = my - (int)(330 * geny);

                        imggfx.setColor(colorgen);
                        imggfx.drawLine(gx1, gy1, gx2, gy2);

                        final int nx1 = mx + (int)(330 * this.lastnetx);
                        final int ny1 = my - (int)(330 * this.lastnety);
                        final int nx2 = mx + (int)(330 * netx);
                        final int ny2 = my - (int)(330 * nety);

                        imggfx.setColor(colornet);
                        imggfx.drawLine(nx1, ny1, nx2, ny2);

                        if (timestep[0] == (trainlength - 1)) {
                            imggfx.fillOval(nx2 - 4, ny2 - 4, 9, 9);
                        }
                    }
                    this.lastgenx = genx;
                    this.lastgeny = geny;
                    this.lastnetx = netx;
                    this.lastnety = nety;
                    timestep[0]++;
                }
                canvas.repaint();
                if (save_vis){
                    try {
                        ImageIO.write(img,"png", new File("./gfx/RNNTrajectory/" + timestep[timestep.length - 1] + ".png" ));
                    } catch (IOException x) {
                        x.printStackTrace();
                    }
                }
            }
        });
        timer.start();
    }
}
