package org.gitia.cudaexamplemaven;

/*
 * JCublas - Java bindings for CUBLAS, the NVIDIA CUDA BLAS library,
 * to be used with JCuda <br />
 * http://www.jcuda.org
 *
 * Copyright 2009-2011 Marco Hutter - http://www.jcuda.org
 */
import java.util.Random;

import org.gitia.froog.statistics.Clock;

/**
 * This is a sample class demonstrating the application of JCublas2 for
 * performing a BLAS 'sgemm' operation, i.e. for computing the matrix <br />
 * C = alpha * A * B + beta * C <br />
 * for single-precision floating point values alpha and beta, and matrices A, B
 * and C of size 1000 x 1000.
 */
public class CPU {

    public static void main(String args[]) {
        // Create a CUBLAS handle

        for (int i = 0; i < 10; i++) {
            System.out.println("iteracion:\t"+i);
            testSgemm(28);
            testSgemm(32);
            testSgemm(64);
            testSgemm(128);
            testSgemm(256);
            testSgemm(512);
            testSgemm(1024);
            testSgemm(2048);
        }

    }

    /**
     * Test the JCublas sgemm operation for matrices of size n x x
     *
     * @param n The matrix size
     */
    public static void testSgemm(int n) {
        Clock c = new Clock();

        double alpha = 1;
        double beta = 1;
        int nn = n * n;

        double h_A[] = createRandomDoubleData(nn);
        double h_B[] = createRandomDoubleData(nn);
        double h_C_ref[] = createRandomDoubleData(nn);

        c.start();
        dgemmJava(n, alpha, h_A, h_B, beta, h_C_ref);
        c.stop();
        c.printTime("CPU\t" + n);
    }

    /**
     * Simple implementation of sgemm, using plain Java
     */
    private static void dgemmJava(int n, double alpha, double A[], double B[],
            double beta, double C[]) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double prod = 0;
                for (int k = 0; k < n; ++k) {
                    prod += A[k * n + i] * B[j * n + k];
                }
                C[j * n + i] = alpha * prod + beta * C[j * n + i];
            }
        }
    }

    /**
     * Creates an array of the specified size, containing some random data
     */
    private static double[] createRandomDoubleData(int n) {
        Random random = new Random();
        double x[] = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = random.nextDouble();
        }
        return x;
    }

}
