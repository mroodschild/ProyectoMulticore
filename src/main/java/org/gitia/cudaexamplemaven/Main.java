/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.gitia.cudaexamplemaven;

/**
 *
 * @author Matías Rodschild <mroodschild@gmail.com>
 */
public class Main {
    public static void main(String[] args) {
        CPU.main(args);
        CPU_EJML.main(args);
        GPU_JCuda.main(args);
        ND4J.main(args);
    }
}
