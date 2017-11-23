/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.gitia.cudaexamplemaven;

import org.ejml.simple.SimpleMatrix;
import org.gitia.froog.transferfunction.Tansig;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import static org.nd4j.linalg.ops.transforms.Transforms.*;

/**
 *
 * @author Matías Rodschild <mroodschild@gmail.com>
 */
public class ND4JExample {

    public static void main(String[] args) {
        int nRows = 2;
        int nColumns = 2;
// Create INDArray of zeros
        INDArray zeros = Nd4j.zeros(nRows, nColumns);
// Create one of all ones
        INDArray ones = Nd4j.ones(nRows, nColumns);
//hstack
        INDArray hstack = Nd4j.hstack(ones, zeros);
        System.out.println("### HSTACK ####");
        System.out.println(hstack);

        double[][] a = {
            {1, 2}, 
            {3, 4}};
        INDArray nd = Nd4j.create(new float[]{1, 2, 3, 4}, new int[]{2, 2});

        INDArray A = Nd4j.create(a);
        System.out.println(nd);
        System.out.println(A);
    }
}
