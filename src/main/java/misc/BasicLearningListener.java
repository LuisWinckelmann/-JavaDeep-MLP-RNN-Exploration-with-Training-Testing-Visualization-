package misc;

public class BasicLearningListener implements LearningListener {
    @Override
    public void afterEpoch(final int epoch, final double trainingerror) {
        System.out.println("epoch: " + epoch + " training error: " + trainingerror);
    }
}