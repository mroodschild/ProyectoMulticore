package org.gitia.cudaexamplemaven;

import java.util.Random;
import static jcuda.jcublas.JCublas2.*;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.runtime.JCuda.*;

import jcuda.*;
import jcuda.jcublas.cublasHandle;
import org.ejml.simple.SimpleMatrix;

public class PesosMult {

    static double h_A[];
    static double h_B[];
    static double h_C[];
    static double h_C_SM[];

    public static void main(String args[]) {
        // Create a CUBLAS handle

        SimpleMatrix W = new SimpleMatrix(2, 3, true,
                0.1, 0.2, 0.3,
                0.3, 0.2, 0.1
        );
        SimpleMatrix X = new SimpleMatrix(3, 3, true,
                0.1, 0.2, 0.1,
                0.1, 0.1, 0.3,
                0.5, 0.6, 0.7
        );
        SimpleMatrix B = new SimpleMatrix(2, 1, true, 0.4, 0.6);
        
        SimpleMatrix w = SimpleMatrix.random(4, 6, -2, 2, new Random());
        SimpleMatrix x = SimpleMatrix.random(6, 10, -2, 2, new Random());
        SimpleMatrix b = SimpleMatrix.random(4, 1, -2, 2, new Random());

        outputZ(w, x, b).print();
        outputZ(W, X, B).print();

    }

    public static SimpleMatrix outputZ(SimpleMatrix W, SimpleMatrix X, SimpleMatrix B) {
        cublasHandle handle = new cublasHandle();
        cublasCreate(handle);

        SimpleMatrix resultado = new SimpleMatrix(W.numRows(), X.numCols());
        for (int i = 0; i < X.numCols(); i++) {
            resultado.setColumn(i, 0, B.getMatrix().getData());
        }

        System.out.println("W");
        W.print();
        System.out.println("X");
        X.print();
        System.out.println("B");
        B.print();
        resultado.print();

        System.out.println("WX");
        W.mult(X).plus(resultado).print();

        double h_W[] = W.transpose().getMatrix().getData();
        double h_X[] = X.transpose().getMatrix().getData();

        double h_B[] = resultado.transpose().getMatrix().getData();

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

        // Execute sgemm
        Pointer pAlpha = Pointer.to(new double[]{alpha});
        Pointer pBeta = Pointer.to(new double[]{beta});

        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                W.numRows(), X.numCols(), W.numCols(), pAlpha,
                d_W, W.numRows(),
                d_X, X.numRows(), pBeta,
                d_B, B.numRows());
        // Copy the result from the device to the host
        cublasGetVector(size_B, Sizeof.DOUBLE, d_B, 1, Pointer.to(h_B), 1);
        // Clean up
        cudaFree(d_W);
        cudaFree(d_X);
        cudaFree(d_B);
        cublasDestroy(handle);
        System.out.println(
                "size h_B: " + h_B.length
                + " row " + W.numRows()
                + " cols " + X.numCols() + " row x cols " + W.numRows() * X.numCols());
        return new SimpleMatrix(W.numRows(), X.numCols(), false, h_B);
    }

}
