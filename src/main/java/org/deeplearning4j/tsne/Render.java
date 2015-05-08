package org.deeplearning4j.tsne;

import org.deeplearning4j.plot.dropwizard.RenderApplication;

/**
 * Created by agibsoncccc on 4/16/15.
 */
public class Render {

    public static void main(String[] args) {
        try {
            RenderApplication.main(args);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
