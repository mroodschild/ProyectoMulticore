package org.gitia.cudaexamplemaven;

/*
 * JCublas - Java bindings for CUBLAS, the NVIDIA CUDA BLAS library,
 * to be used with JCuda <br />
 * http://www.jcuda.org
 *
 * Copyright 2009-2011 Marco Hutter - http://www.jcuda.org
 */
import org.gitia.froog.statistics.Clock;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * This is a sample class demonstrating the application of JCublas2 for
 * performing a BLAS 'sgemm' operation, i.e. for computing the matrix <br />
 * C = alpha * A * B + beta * C <br />
 * for single-precision floating point values alpha and beta, and matrices A, B
 * and C of size 1000 x 1000.
 */
public class ND4J {

    public static void main(String args[]) {
        for (int i = 0; i < 10; i++) {
            System.out.println("iteracion:\t" + i);
            testSgemm(28);
            testSgemm(32);
            testSgemm(64);
            testSgemm(128);
            testSgemm(256);
            testSgemm(512);
            testSgemm(1024);
            testSgemm(2048);
            testSgemm(4096);
        }
    }

    /**
     * Test the JCublas sgemm operation for matrices of size n x x
     *
     * @param n The matrix size
     */
    public static void testSgemm(int n) {
        Clock c = new Clock();
        //System.out.println("Creating input data...");
        INDArray w = Nd4j.rand(n, n);
        INDArray x = Nd4j.rand(n, n);
        INDArray b = Nd4j.rand(n, n);

        c.start();
        DGEMM_ND4J(w, x, b);
        c.stop();
        c.printTime("GPU ND4J\t" + n);
    }

    public static INDArray DGEMM_ND4J(INDArray W, INDArray X, INDArray B) {
        double alpha = 1;
        double beta = 1;
        return W.mmul(X).muli(alpha).add(B.muli(beta));
    }

}
