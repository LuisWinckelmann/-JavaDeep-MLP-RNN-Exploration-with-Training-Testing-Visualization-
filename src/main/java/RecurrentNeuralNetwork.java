import java.util.Random;

import misc.LearningListener;
import misc.Tools;

/**
 * Simple implementation of a recurrent neural network with 
 * hyperbolic tangent neurons and trainable biases.
 * 
 * @author Sebastian Otte
 */
public class RecurrentNeuralNetwork {
    
    public final static double BIAS = 1.0;
    
    private final int layersnum;
    private final int inputsize;
    private final int weightsnum;
    private final int[] layer;
    private final double[][][] net;
    private final double[][][] act;
    private final double[][][] bwbuffer;
    private final double[][][] delta;
    private final double[][][][] weights;
    private final boolean[] usebias;
    private final double[][][][] dweights;
    
    private int bufferlength    = 0;
    private int lastinputlength = 0;
    
    private static int[] join(final int i1, final int i2, final int ...in) {
        final int[] result = new int[2 + in.length];
        result[0] = i1;
        result[1] = i2;
        System.arraycopy(in, 0, result, 2, in.length);
        return result;
    }
    public double[][][] getAct() { return this.act; }
    public double[][][] getNet() { return this.net; }
    public double[][][] getDelta() { return this.delta; }
    
    public void setBias(final int layer, final boolean bias) {
        assert(layer > 0 && layer < this.layersnum);
        this.usebias[layer] = bias;
    }
    
    public boolean getBias(final int layer) {
        assert(layer > 0 && layer < this.layersnum);
        return this.usebias[layer];
    }
    
    /**
     * Constructor of the RNN class. The function signature ensures that there is
     * at least an input and an output layer (in this case layern.length would be 0). 
     * Otherwise layer2 would be the first hidden layer and layern would contain
     * at least the output layer and, optionally, further hidden layers. Effectively,
     * the firstly given layer defines the input layer size, the lastly given number
     * defines the output layer size, and the numbers in between define the hidden 
     * layers sizes, accordingly.
     * 
     * @param input Number of input neurons.
     * @param layer2 Layer of hidden/output neurons.
     * @param layern A list of further hidden and the output layer.
     */
    public RecurrentNeuralNetwork(
        final int input, final int layer2, final int... layern 
    ) {
        this.inputsize = input;
        this.layer     = join(input, layer2, layern);
        this.layersnum = this.layer.length;
        // set up buffers.
        this.act      = new double[this.layersnum][][];
        this.net      = new double[this.layersnum][][];
        this.delta    = new double[this.layersnum][][];
        this.bwbuffer = new double[this.layersnum][][];

        this.rebufferOnDemand(1);

        this.usebias  = new boolean[this.layersnum];
        this.weights  = new double[this.layersnum][this.layersnum][][];
        this.dweights = new double[this.layersnum][this.layersnum][][];

        int sumweights = 0;

        for (int l = 0; l < this.layersnum; l++) {
            //
            this.usebias[l] = true;

            /*
             The weights buffer works differently compared to the MLP:
               o its 4-dimension
               o the first 2 indices address the source and the destination
                 layer matrix, respectively.
               o the last 2 indices address the connections in the usual manner
             For instance, this.weights[0][1][4][6] this address connection
             from neuron 4 of layer 0 to neuron 6 of layer 1.
             Note that in this implementation only the forward weight matrices [l][l+1] and
             recurrent weight matrices [l][l] (for non-input non-output layers)
             are defined -> all others are null.
            */
            if (l > 0) {
                // forward weights. the plus + 1 used to model the bias weights.
                this.weights[l-1][l]  = new double[this.layer[l - 1] + 1][this.layer[l]];
                this.dweights[l-1][l] = new double[this.layer[l - 1] + 1][this.layer[l]];
                sumweights += (this.layer[l - 1] + 1) * (this.layer[l]);
                // if the current layer is a hidden layer, also add recurrent connections.
                if (l < (this.layersnum - 1)) {
                    this.weights[l][l]  = new double[this.layer[l]][this.layer[l]];
                    this.dweights[l][l] = new double[this.layer[l]][this.layer[l]];
                    sumweights += (this.layer[l] * this.layer[l]);
                }
            }
        }
        this.weightsnum = sumweights;
    }
    private static double tanh(final double x) {
        return Math.tanh(x);
    }
    private static double tanhDx(final double x) {
        final double tanhx = Math.tanh(x);
        return 1 - (tanhx * tanhx);
    }
    public void reset() {
        for (int l = 0; l < this.layersnum; l++) {
            //
            for (int i = 0; i < this.layer[l]; i++) {
                for (int t = 0; t < this.bufferlength; t++) {
                    this.act[l][i][t]      = 0.0;
                    if (l > 0) {
                        this.net[l][i][t]      = 0.0;
                        this.delta[l][i][t]    = 0.0;
                        this.bwbuffer[l][i][t] = 0.0;
                    }
                }
            }
        }
        this.lastinputlength = 0;
    }
    
    public void rebufferOnDemand(final int sequencelength) {
        if (this.bufferlength != sequencelength) {
            for (int l = 0; l < this.layersnum; l++) {
                if (l > 0) {
                    // we don't need the net buffer for input neurons.
                    this.net[l]      = new double[this.layer[l]][sequencelength];
                    this.delta[l]    = new double[this.layer[l]][sequencelength];
                    this.bwbuffer[l] = new double[this.layer[l]][sequencelength];
                }
                this.act[l] = new double[this.layer[l]][sequencelength];
            }
        }
        this.bufferlength    = sequencelength;
        this.lastinputlength = 0;
    }

    /**
     * Computes the forward pass, i.e., propagates an input 
     * vector through the network to the output layer. This is
     * a wrapper method for one time step online.
     * @param input Input vector.
     * @return Output vector.
     */
    public double[] forwardPass(final double[] input) {
        return forwardPass(new double[][]{input})[0];
    }
    
    /**
     * Computes the forward pass, i.e., propagates a sequence
     * of input vectors through the network to the output layer.
     * @param input Sequence of input vectors.
     * @return Output Sequence of output vectors.
     */
    public double[][] forwardPass(final double[][] input) {
        final int sequencelength = Math.min(this.bufferlength, input.length);
        final double[][] output  = new double[sequencelength][];
        final int outputlayer    = this.layersnum - 1;
        int prevt = 0;
        for (int t = 0; t < sequencelength; t++) {
            // store input.
            assert(input.length == this.inputsize);
            for (int i = 0; i < input[t].length; i++) {
                this.act[0][i][t] = input[t][i];
            }
            // compute output layer-wise.
            // start with the first hidden layer or the output layer if there is no hidden layer.
            for (int l = 1; l < this.layersnum; l++) {
                // first compute all the net (integrate the inputs) values and activations.
                final int layersize    = this.layer[l];
                final int prelayersize = this.layer[l - 1];

                final double[][] ff_weights = this.weights[l - 1][l];
                final double[][] fb_weights = this.weights[l][l];

                for (int j = 0; j < layersize; j++) {
                    // eventually initialize netjt with the weighted bias.
                    double netjt = 0;
                    if (this.usebias[l]) {
                        netjt = BIAS * ff_weights[prelayersize][j];
                    }
                    // integrate feed-forward input.
                    for (int i = 0; i < prelayersize; i++) {
                        netjt += this.act[l - 1][i][t] * ff_weights[i][j];
                    }
                    if (l < outputlayer) {
                        // integrate recurrent input.
                        for (int i = 0; i < layersize; i++) {
                            netjt += this.act[l][i][prevt] * fb_weights[i][j];
                        }
                    }
                    this.net[l][j][t] = netjt;
                }
                // now we compute the activations of the neurons in the current layer.
                if (l < outputlayer) {
                    // tanh hidden layer.
                    for (int j = 0; j < layersize; j++) {
                        // System.out.println(this.net[l][j][t]);
                        this.act[l][j][t] = tanh(this.net[l][j][t]);
                    }    
                } else {
                    // linear output layer.
                    for (int j = 0; j < layersize; j++) {
                        this.act[l][j][t] = this.net[l][j][t];
                    }    
                }
            }
            // store output.
            final int outputlayersize = this.layer[outputlayer];
            output[t] = new double[outputlayersize];
            for (int k = 0; k < outputlayersize; k++) {
                output[t][k] = this.act[outputlayer][k][t];
            }
            if (t > 0) prevt = t;
        }
        // Store input length of the current sequence. You can 
        // use this information in the backward pass, i.e., starting the
        // back propagation through time procedure at this index,
        // which can be smaller than the current buffer length
        // depending on the current input sequence.
        this.lastinputlength = sequencelength;
        // return output layer activation as output.
        return output;
    }

    public void backwardPass(final double[][] target) {
        final int outputlayer = this.delta.length - 1;
        final int steps       = this.lastinputlength;
        int t_target = target.length - 1;
        // System.out.println(t_target + " " + steps);
        // compute reversely in time.
        int nextt = steps -1;
        for (int t = (steps - 1); t >= 0; t--) {
            // inject the output/target discrepancy into this.bwbuffer. Note that
            // this.bwbuffer functions analogously to this.net but is used to
            // store into "back-flowing" inputs (deltas).
            if (t_target >= 0) {
                for (int j = 0; j < this.delta[outputlayer].length; j++) {
                    this.bwbuffer[outputlayer][j][t] = (this.act[outputlayer][j][t] - target[t_target][j]);
                }
            }
            // back-propagate the error through the network -- we compute the deltas --
            // starting with the output layer.
            for (int h = 0; h < this.delta[outputlayer].length; h++) {
                this.delta[outputlayer][h][t] = tanhDx(this.net[outputlayer][h][t]) * bwbuffer[outputlayer][h][t];
            }

            for (int l = this.layersnum-2; l >= 0; l--) {      // iteration over all layers (starting from the last one)
                final int layersize = this.layer[l];
                int prelayersize;
                if(l>0) {
                    //Outer derivative, Hidden-Layer:
                    final double[][] ff_weights = this.weights[l][l+1];
                    final double[][] fb_weights = this.weights[l][l];
                    prelayersize = this.layer[l + 1];
                    for (int h = 0; h < layersize; h++) {
                        double sum = 0;
                        for (int k = 0; k < prelayersize; k++) {
                            sum += this.delta[l+1][k][t] * ff_weights[h][k];

                        }
                        for (int k = 0; k < layersize; k++) {
                            sum += this.delta[l][k][nextt] * fb_weights[h][k];
                        }
                        this.bwbuffer[l][h][t] = sum;
                    }
                    // combine outer with inner derivative
                    for (int h = 0; h < layersize; h++) {
                        this.delta[l][h][t] = tanhDx(this.net[l][h][t]) * bwbuffer[l][h][t];
                    }
                }

            }
            if (t < steps - 1) nextt = t;
            t_target--;
        }
        // Compute the weights derivatives.

       for (int l = 1; l < outputlayer; l++){
           final int layersize    = this.layer[l];
           final int prelayersize = this.layer[l + 1];
           // calculate weight derivatives
           for (int j = 0; j < prelayersize; j++) {
               for (int i = 0; i < layersize; i++) {
                   double del_ff = 0.0;
                   double del_fb = 0.0;
                   double del_Fb = 0.0;
                   for (int t = 0; t < steps ; t++) {
                       del_ff += this.act[l][i][t] * this.delta[l+1][j][t];
                       del_fb += this.act[l][i][t] * this.delta[l][i][t];
                       if (this.usebias[l]){
                           del_Fb += BIAS * this.delta[l][i][t];
                       }
                    }
                   this.dweights[l][l+1][i][j] = del_ff;
                   this.dweights[l][l][i][j] = del_fb;
                   if (this.usebias[l]){
                       this.dweights[l][l][outputlayer][j] = del_Fb;
                   }
                }
           }
       }
    }

    /**
     * Initializes the weights randomly and normal distributed with
     * std. dev. 0.1.
     * @param rnd Instance of Random.
     */
    public void initializeWeights(final Random rnd, final double stddev) {
        for (double[][][] weight : this.weights) {
            for (double[][] wll : weight) {
                if (wll != null) {
                    for (int i = 0; i < wll.length; i++) {
                        for (int j = 0; j < wll[i].length; j++) {
                            wll[i][j] = rnd.nextGaussian() * stddev;
                        }
                    }
                }
            }
        }
    }
    
    public int getWeightsNum() {
        return this.weightsnum;
    }

    private static void map(final double[] from, final double[][][][] to) {
        int idx = 0;
        for (double[][][] doubles : to) {
            for (double[][] wll : doubles) {
                if (wll != null) {
                    for (int i = 0; i < wll.length; i++) {
                        for (int j = 0; j < wll[i].length; j++) {
                            wll[i][j] = from[idx++];
                        }
                    }
                }
            }
        }
    }

    private static void map(final double[][][][] from, final double[] to) {
        int idx = 0;
        for (double[][][] doubles : from) {
            for (double[][] wll : doubles) {
                if (wll != null) {
                    for (double[] value : wll) {
                        for (double v : value) {
                            to[idx++] = v;
                        }
                    }
                }
            }
        }
    }

    public void writeWeights(final double[] weights) {
        map(weights, this.weights);
    }

    public void readWeights(final double[] weights) {
        map(this.weights, weights);
    }

    public void readDWeights(final double[] dweights) {
        map(this.dweights, dweights);
    }

    /**
     * Stochastic gradient descent.
     * @param rnd Instance of Random.
     * @param input Input vectors.
     * @param target Target vectors.
     * @param epochs Number of epochs.
     * @param learningrate Value for the learning rate.
     * @param momentumrate Value for the momentum rate.
     * @param listener Listener to observe the training progress.
     * @return The final epoch error.
     */    
    public double trainStochastic(
        final Random rnd, 
        final double[][][] input, 
        final double[][][] target,
        final double epochs,
        final double learningrate,
        final double momentumrate,
        final LearningListener listener
    ) {
        assert(input.length == target.length);
        final double[] weights       = new double[this.weightsnum];
        final double[] dweights      = new double[this.weightsnum];
        final double[] weightsupdate = new double[this.weightsnum];

        this.readWeights(weights);
        // create initial index permutation.
        final int[] indices = new int[input.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        double error = Double.POSITIVE_INFINITY;
        // epoch loop.
        double lr;
        lr = learningrate;
        for (int i = 0; i < epochs; i++) {
            // shuffle indices.
            Tools.shuffle(indices, rnd);
            double errorsum = 0.0;
            // train all samples in online manner, i.e. iterate over all samples, while considering the shuffled order
            // and update the weights immediately after each sample
            for (int k = 0; k < this.weightsnum; k++){
                weightsupdate[k] = 0;
            }
            for (int j: indices){
                double[][] output = forwardPass(input[j]);
                errorsum += RMSE(output,target[j]);
                backwardPass(target[j]);
                this.readWeights(weights);
                this.readDWeights(dweights);

                for (int k = 0; k < this.weightsnum; k++){
                    weights[k] += - lr * dweights[k] + momentumrate * weightsupdate[k];
                }
                this.writeWeights(weights);
                this.reset();
            }
            error = errorsum / (double)(input.length);
            if (listener != null & i%1000 == 0) listener.afterEpoch(i + 1, error);
            if (this.weightsnum >= 0) System.arraycopy(dweights, 0, weightsupdate, 0, this.weightsnum);
        }
        return error;
    }

    /**
     * Computes the RMSE of the current output and a given target vector.
     * @param target Target vector.
     * @return RMSE value.
     */
    public static double RMSE(final double[][] output, final double[][] target) {
        final int length = Math.min(output.length, target.length);

        double error = 0;
        int    ctr   = 0;

        for (int t = 0; t < length; t++) {
            assert(output[t].length > 0);
            assert(target[t].length > 0);
            assert(target[t].length == output.length);

            for (int i = 0; i < target[t].length; i++) {
                final double e = output[t][i] - target[t][i];
                error += (e * e);
                ctr++;
            }
        }
        return Math.sqrt(error / (double)(ctr));
    }

    
}