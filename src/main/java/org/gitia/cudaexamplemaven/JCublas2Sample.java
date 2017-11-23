package org.gitia.cudaexamplemaven;

/*
 * JCublas - Java bindings for CUBLAS, the NVIDIA CUDA BLAS library,
 * to be used with JCuda <br />
 * http://www.jcuda.org
 *
 * Copyright 2009-2011 Marco Hutter - http://www.jcuda.org
 */
import static jcuda.jcublas.JCublas2.*;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.runtime.JCuda.*;

import java.util.Random;

import jcuda.*;
import jcuda.jcublas.cublasHandle;
import org.ejml.simple.SimpleMatrix;
import org.gitia.froog.statistics.Clock;

/**
 * This is a sample class demonstrating the application of JCublas2 for
 * performing a BLAS 'sgemm' operation, i.e. for computing the matrix <br />
 * C = alpha * A * B + beta * C <br />
 * for single-precision floating point values alpha and beta, and matrices A, B
 * and C of size 1000x1000.
 */
public class JCublas2Sample {

    public static void main(String args[]) {
        // Create a CUBLAS handle

        testSgemm(2000);
    }

    /**
     * Test the JCublas sgemm operation for matrices of size n x x
     *
     * @param n The matrix size
     */
    public static void testSgemm(int n) {
        Clock c = new Clock();
        c.start();
        cublasHandle handle = new cublasHandle();
        cublasCreate(handle);
        c.stop();
        c.printTime("Init Handle");
        double alpha = 0.3;
        double beta = 0.7;
        int nn = n * n;

        System.out.println("Creating input data...");
        double h_A[] = createRandomDoubleData(nn);
        double h_B[] = createRandomDoubleData(nn);
        double h_C[] = createRandomDoubleData(nn);
        double h_C_ref[] = h_C.clone();
        double h_C_SM[] = h_C.clone();

        System.out.println("x:\t" + h_C[0]);
        System.out.println("x ref:\t" + h_C_ref[0]);
        System.out.println("x SM:\t" + h_C_SM[0]);

        c.start();
        System.out.println("Performing Sgemm with SimpleMatrix...");
        h_C_SM = dgemmSimpleMatrix(n, alpha, h_A, h_B, beta, h_C_SM);
        c.stop();
        c.printTime("SimpleMatrix");

        c.start();
        System.out.println("Performing Sgemm with Java...");
        dgemmJava(n, alpha, h_A, h_B, beta, h_C_ref);
        c.stop();
        c.printTime("CPU");

        c.start();
        System.out.println("Performing Sgemm with JCublas...");
        dgemmJCublas(n, alpha, h_A, h_B, beta, h_C, handle);
        c.stop();
        c.printTime("GPU");

        boolean passed = isCorrectResult(h_C, h_C_ref);
        System.out.println("testSgemm " + (passed ? "PASSED" : "FAILED"));

        boolean passed2 = isCorrectResult(h_C, h_C_SM);
        System.out.println("testSgemmSimpleMatrix " + (passed2 ? "PASSED" : "FAILED"));

        System.out.println("x:\t" + h_C[0]);
        System.out.println("x ref:\t" + h_C_ref[0]);
        System.out.println("x SM:\t" + h_C_SM[0]);
        
        System.out.println("x:\t" + h_C[3]);
        System.out.println("x ref:\t" + h_C_ref[3]);
        System.out.println("x SM:\t" + h_C_SM[3]);

        cublasDestroy(handle);
    }

    private static double[] dgemmSimpleMatrix(int n, double alpha, double A[], double B[],
            double beta, double C[]) {
        SimpleMatrix a = new SimpleMatrix(n, n, false, A);
        SimpleMatrix b = new SimpleMatrix(n, n, false, B);
        SimpleMatrix c = new SimpleMatrix(n, n, false, C);
        return a.scale(alpha).mult(b).plus(c.scale(beta)).transpose().getMatrix().getData();
    }

    /**
     * Implementation of sgemm using JCublas
     */
    private static void dgemmJCublas(int n, double alpha, double A[], double B[],
            double beta, double C[], cublasHandle handle) {
        int nn = n * n;

        // Allocate memory on the device
        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_C = new Pointer();
        cudaMalloc(d_A, nn * Sizeof.DOUBLE);
        cudaMalloc(d_B, nn * Sizeof.DOUBLE);
        cudaMalloc(d_C, nn * Sizeof.DOUBLE);

        // Copy the memory from the host to the device
        cublasSetVector(nn, Sizeof.DOUBLE, Pointer.to(A), 1, d_A, 1);
        cublasSetVector(nn, Sizeof.DOUBLE, Pointer.to(B), 1, d_B, 1);
        cublasSetVector(nn, Sizeof.DOUBLE, Pointer.to(C), 1, d_C, 1);

        // Execute sgemm
        Pointer pAlpha = Pointer.to(new double[]{alpha});
        Pointer pBeta = Pointer.to(new double[]{beta});
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                pAlpha, d_A, n, d_B, n, pBeta, d_C, n);
        // Copy the result from the device to the host
        cublasGetVector(nn, Sizeof.DOUBLE, d_C, 1, Pointer.to(C), 1);
        // Clean up
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
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

    /**
     * Compares the given result against a reference, and returns whether the
     * error norm is below a small epsilon threshold
     */
    private static boolean isCorrectResult(double result[], double reference[]) {
        double errorNorm = 0;
        double refNorm = 0;
        for (int i = 0; i < result.length; ++i) {
            double diff = reference[i] - result[i];
            errorNorm += diff * diff;
            refNorm += reference[i] * result[i];
        }
        errorNorm = (double) Math.sqrt(errorNorm);
        refNorm = (double) Math.sqrt(refNorm);
        if (Math.abs(refNorm) < 1e-6) {
            return false;
        }
        return (errorNorm / refNorm < 1e-6f);
    }
}
