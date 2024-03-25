package misc;


public class Spiral implements TrajectoryGenerator {

    private double scale; 
    private int t;
    
    public Spiral() {
        this.reset();
    }
    
    @Override
    public int vectorsize() {
        return 2;
    }
    
    @Override
    public void reset() {
        this.scale = 1.0;
        this.t     = 0;
    }

    @Override
    public double[] next() {
        final double[] result = new double[2];
        final double tf = 0.1 * (double)(this.t);
        result[0] = Math.cos(tf) * scale;
        result[1] = Math.sin(tf) * scale;

        this.scale *= 0.99;
        t++;
        return result;
    }
    
}