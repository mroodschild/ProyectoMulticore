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
public class CPU_EJML {

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
        }
        cublasDestroy(handle);
    }

    /**
     * Test the JCublas sgemm operation for matrices of size n x x
     *
     * @param n The matrix size
     */
    public static void testSgemm(int n) {
        Clock c = new Clock();
        //System.out.println("Creating input data...");
        SimpleMatrix w = SimpleMatrix.random(n, n, 0, 1, new Random());
        SimpleMatrix x = SimpleMatrix.random(n, n, 0, 1, new Random());
        SimpleMatrix b = SimpleMatrix.random(n, 1, 0, 1, new Random());
        SimpleMatrix B = new SimpleMatrix(w.numRows(), x.numCols());
        
        for (int i = 0; i < x.numCols(); i++) {
            B.setColumn(i, 0, b.getMatrix().getData());
        }
        
        c.start();
        DGEMM_EJML(w, x, B);
        c.stop();
        c.printTime("CPU EJML\t" + n);
    }

    public static SimpleMatrix DGEMM_EJML(SimpleMatrix W, SimpleMatrix X, SimpleMatrix B) {
        double alpha = 1;
        double beta = 1;
        return W.mult(X).scale(alpha).plus(B.scale(beta));
    }

}
