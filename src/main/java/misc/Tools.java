package misc;

import java.util.Random;

public class Tools {
    public static void shuffle(
        final int[] data, 
        final Random rnd
    ) {
       final int size = data.length;
       for (int i = size; i > 1; i--) {
           final int ii = i - 1;
           final int r  = rnd.nextInt(i);
           final int temp = data[ii];
           data[ii] = data[r];
           data[r] = temp;
       }
    }    
}