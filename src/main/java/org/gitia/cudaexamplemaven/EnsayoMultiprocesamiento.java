package org.gitia.cudaexamplemaven;

import java.util.Random;
import static jcuda.jcublas.JCublas2.*;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.runtime.JCuda.*;

import jcuda.*;
import jcuda.jcublas.cublasHandle;
import org.ejml.simple.SimpleMatrix;
import org.gitia.froog.statistics.Clock;

public class EnsayoMultiprocesamiento {

    static cublasHandle handle;
    static double h_A[];
    static double h_B[];
    static double h_C[];
    static double h_C_SM[];

    public static void main(String args[]) {
        // Create a CUBLAS handle
        handle = new cublasHandle();
        cublasCreate(handle);
        
        int m = 200;
        int n = 200;
        int k = n;

        SimpleMatrix w = SimpleMatrix.random(m, n, -2, 2, new Random());
        SimpleMatrix x = SimpleMatrix.random(n, k, -2, 2, new Random());
        SimpleMatrix b = SimpleMatrix.random(m, 1, -2, 2, new Random());
        
        Clock c = new Clock();
        c.start();
        DGEMM_Cuda(w, x, b);
        c.stop();
        c.printTime("Cuda");
        c.start();
        DGEMM_EJML(w, x, b);
        c.stop();
        c.printTime("EJML");

        cublasDestroy(handle);
    }

    /**
     * Simple implementation of sgemm, using plain Java
     */
    private static void DGEMM_Java(int n, double A[], double B[],
            double C[], SimpleMatrix W, SimpleMatrix X, SimpleMatrix b) {
        
        SimpleMatrix b_aux = new SimpleMatrix(W.numRows(), X.numCols());
        for (int i = 0; i < X.numCols(); i++) {
            b_aux.setColumn(i, 0, b.getMatrix().getData());
        }
        
        A = W.transpose().getMatrix().getData();
        B = X.transpose().getMatrix().getData();
        C = b_aux.transpose().getMatrix().getData();
        
        double alpha = 1;
        double beta = 1;
        
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

    public static SimpleMatrix DGEMM_Cuda(SimpleMatrix W, SimpleMatrix X, SimpleMatrix B) {
        SimpleMatrix b_aux = new SimpleMatrix(W.numRows(), X.numCols());
        for (int i = 0; i < X.numCols(); i++) {
            b_aux.setColumn(i, 0, B.getMatrix().getData());
        }

        double h_W[] = W.transpose().getMatrix().getData();
        double h_X[] = X.transpose().getMatrix().getData();
        double h_B[] = b_aux.transpose().getMatrix().getData();
        double alpha = 1;
        double beta = 1;

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
        return new SimpleMatrix(W.numRows(), X.numCols(), false, h_B);
    }

    public static SimpleMatrix DGEMM_EJML(SimpleMatrix W, SimpleMatrix X, SimpleMatrix B) {
        SimpleMatrix b_aux = new SimpleMatrix(W.numRows(), X.numCols());
        double alpha = 1;
        double beta = 1;
        for (int i = 0; i < X.numCols(); i++) {
            b_aux.setColumn(i, 0, B.getMatrix().getData());
        }
        return W.mult(X).scale(alpha).plus(b_aux.scale(beta));
    }

}
