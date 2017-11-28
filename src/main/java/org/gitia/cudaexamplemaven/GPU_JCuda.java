package org.gitia.cudaexamplemaven;

/*
 * JCublas - Java bindings for CUBLAS, the NVIDIA CUDA BLAS library,
 * to be used with JCuda <br />
 * http://www.jcuda.org
 *
 * Copyright 2009-2011 Marco Hutter - http://www.jcuda.org
 */
import java.util.Random;
import jcuda.Pointer;
import jcuda.Sizeof;
import static jcuda.jcublas.JCublas2.cublasCreate;
import static jcuda.jcublas.JCublas2.cublasDestroy;
import static jcuda.jcublas.JCublas2.cublasDgemm;
import static jcuda.jcublas.JCublas2.cublasGetVector;
import static jcuda.jcublas.JCublas2.cublasSetVector;
import jcuda.jcublas.cublasHandle;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
import org.ejml.simple.SimpleMatrix;

import org.gitia.froog.statistics.Clock;

/**
 * This is a sample class demonstrating the application of JCublas2 for
 * performing a BLAS 'sgemm' operation, i.e. for computing the matrix <br />
 * C = alpha * A * B + beta * C <br />
 * for single-precision floating point values alpha and beta, and matrices A, B
 * and C of size 1000 x 1000.
 */
public class GPU_JCuda {

    static cublasHandle handle;

    public static void main(String args[]) {
        handle = new cublasHandle();
        cublasCreate(handle);

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
        cublasDestroy(handle);
    }

    /**
     * Test the JCublas sgemm operation for matrices of size n x x
     *
     * @param n The matrix size
     */
    public static void testSgemm(int n) {
        //    Clock c = new Clock();
        //System.out.println("Creating input data...");
        SimpleMatrix w = SimpleMatrix.random(n, n, 0, 1, new Random());
        SimpleMatrix x = SimpleMatrix.random(n, n, 0, 1, new Random());
        SimpleMatrix b = SimpleMatrix.random(n, 1, 0, 1, new Random());

//        c.start();
        //System.out.println("Performing Sgemm with Java...");
        DGEMM_Cuda(w, x, b);
        //      c.stop();
        //c.printTime("GPU\t" + n);
    }

    public static SimpleMatrix DGEMM_Cuda(SimpleMatrix W, SimpleMatrix X, SimpleMatrix B) {
        Clock c = new Clock();
        SimpleMatrix b_aux = new SimpleMatrix(W.numRows(), X.numCols());
        for (int i = 0; i < X.numCols(); i++) {
            b_aux.setColumn(i, 0, B.getMatrix().getData());
        }

        double h_W[] = W.transpose().getMatrix().getData();
        double h_X[] = X.transpose().getMatrix().getData();
        double h_B[] = b_aux.transpose().getMatrix().getData();
        double alpha = 1;
        double beta = 1;

        c.start();
        int size_W = h_W.length;
        int size_X = h_X.length;
        int size_B = h_B.length;

        Pointer d_W = new Pointer();
        Pointer d_X = new Pointer();
        Pointer d_B = new Pointer();
        cudaMalloc(d_W, size_W * Sizeof.DOUBLE);
        cudaMalloc(d_X, size_X * Sizeof.DOUBLE);
        cudaMalloc(d_B, size_B * Sizeof.DOUBLE);

        // Copy the memory from the host to the device
        cublasSetVector(size_W, Sizeof.DOUBLE, Pointer.to(h_W), 1, d_W, 1);
        cublasSetVector(size_X, Sizeof.DOUBLE, Pointer.to(h_X), 1, d_X, 1);
        cublasSetVector(size_B, Sizeof.DOUBLE, Pointer.to(h_B), 1, d_B, 1);

        Pointer pAlpha = Pointer.to(new double[]{alpha});
        Pointer pBeta = Pointer.to(new double[]{beta});

        
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                W.numRows(), X.numCols(), W.numCols(), pAlpha,
                d_W, W.numRows(),
                d_X, X.numRows(), pBeta,
                d_B, B.numRows());
        
        cublasGetVector(size_B, Sizeof.DOUBLE, d_B, 1, Pointer.to(h_B), 1);

        cudaFree(d_W);
        cudaFree(d_X);
        cudaFree(d_B);
        c.stop();
        c.printTime("GPU\t" + W.numCols());
        return new SimpleMatrix(W.numRows(), X.numCols(), false, h_B);
    }

}
