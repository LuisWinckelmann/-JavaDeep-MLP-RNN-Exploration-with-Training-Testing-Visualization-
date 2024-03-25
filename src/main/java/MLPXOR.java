import java.util.Random;

import misc.BasicLearningListener;

public class MLPXOR {
    public static void main(String[] args) {
        final double[][] input = {
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        };
        final double[][] target = {
            {0}, {1}, {1}, {0}
        };
        final Random rnd = new Random(100L);
        // set up network. biases are used by default, but can be deactivated using net.setBias(layer, false),
        // where layer gives the layer index (1 = the first hidden layer).
        final MultiLayerPerceptron net = new MultiLayerPerceptron(2, 150,128, 1);
        // perform training.
        final int epochs = 10000;
        final double learningrate = 1.7;
        final double momentumrate = 0.3;
        // Working parameters:
        // generate initial weights.
        net.initializeWeights(rnd, 0.1);
        net.trainStochastic(
            rnd, 
            input,
            target,
            epochs,
            learningrate,
            momentumrate,
            new BasicLearningListener()
        );
    }
}
