# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 21:02:33 2023

@author: XwGu
"""

import numpy
import math
import scipy.spatial
import scipy.io
import pandas
import os
import numpy.matlib

class SAFLS:
    def __init__(self,Mu0,Gamma0,M0,Omega0): ## create an empty AFIS template
        self.Mu0=Mu0
        self.Gamma0=Gamma0#0.6
        self.M0=M0#0
        self.Omega0=Omega0#1000
    
    def Learning(self,data,y):
        self.L,self.W=data.shape
        self.L,self.Wo=y.shape
        self.IDX=numpy.array([0])
        self.centres=data[0,:].copy().reshape(1,self.W)
        self.prototyps=data[0,:].copy().reshape(1,self.W)
        self.local_X=numpy.power(data[0,:],2).reshape(1,self.W);
        self.local_delta=numpy.zeros([1,self.W])
        self.global_mean=data[0,:].copy()
        self.global_X=numpy.power(data[0,:],2);
        self.supports=numpy.array([1])
        self.numRule=1
        self.sumLambda=1
        self.A=numpy.zeros((self.Wo,self.W+1,1))
        self.C=(numpy.eye(self.W+1)*self.Omega0).reshape(self.W+1,self.W+1,1)
        self.Ye=numpy.zeros((self.L,self.Wo))
        for ii in range(1,self.L):
            datain=data[ii,:].copy().reshape(1,self.W)
            datain1=numpy.append([[1]],datain,axis=1)
            yin=y[ii,:]
            self.global_mean=(self.global_mean*ii+datain)/(ii+1)
            self.global_X=(self.global_X*ii+numpy.power(datain,2))/(ii+1)
            self.global_delta=abs(self.global_X-numpy.power(self.global_mean,2))
            localDensity,centreLambda=self.firingstrength(datain)
            self.Ye[ii,:]=self.outputgeneration(datain1,localDensity)
            if numpy.max(localDensity)<self.Mu0:
                self.centres=numpy.append(self.centres,datain.copy(),axis=0)
                self.prototyps=numpy.append(self.prototyps,datain.copy(),axis=0)
                self.local_X=numpy.append(self.local_X,numpy.power(datain,2),axis=0)
                self.supports=numpy.append(self.supports,1)
                self.numRule=self.numRule+1
                self.A=numpy.append(self.A,numpy.mean(self.A,axis=2).reshape(self.Wo,self.W+1,1),axis=2)
                self.C=numpy.append(self.C,(numpy.eye(self.W+1)*self.Omega0).reshape(self.W+1,self.W+1,1),axis=2)
                self.IDX=numpy.append(self.IDX,ii)
                self.sumLambda=numpy.append(self.sumLambda,0)
                self.local_delta=numpy.append(self.local_delta,numpy.zeros([1,self.W]),axis=0)
                localDensity=numpy.append(localDensity,1)
            else:
                idx=numpy.argmax(localDensity)
                temp=self.supports[idx].copy()
                self.supports[idx]=self.supports[idx]+1
                self.centres[idx,:]=(self.centres[idx,:]*temp+datain)/self.supports[idx]
                self.local_X[idx,:]=(self.local_X[idx,:]*temp+numpy.power(datain,2))/self.supports[idx]
                self.local_delta[idx,:]=abs(self.local_X[idx,:]-numpy.power(self.centres[idx,:],2))
                localDensity[idx]=numpy.exp(numpy.sum(numpy.power((self.prototyps[idx,:]-datain),2))/numpy.sum((self.local_delta[idx,:]+self.global_delta)/2)*-1)
            self.sumLambda=self.sumLambda+localDensity
            utility=numpy.ones((self.numRule,))
            seq0=[i for i in range(0,self.numRule) if ii-self.IDX[i]>0]
            utility[seq0]=self.sumLambda[seq0]/(ii-self.IDX)[seq0]
            seq1=[i for i in range(0,self.numRule) if utility[i]>=self.M0]
            if len(seq1)<self.numRule:
                self.numRule=len(seq1)
                self.centres=self.centres[seq1,:].copy()
                self.prototyps=self.prototyps[seq1,:].copy()
                self.local_X=self.local_X[seq1,:].copy()
                self.supports=self.supports[seq1].copy()
                self.A=self.A[:,:,seq1].copy()
                self.C=self.C[:,:,seq1].copy()
                self.IDX=self.IDX[seq1].copy()
                self.sumLambda=self.sumLambda[seq1].copy()
                self.local_delta=self.local_delta[seq1,:].copy()
                localDensity=localDensity[seq1].copy()
            aseq,localDensity,centreLambda=self.activtingruleselection(localDensity)
            for jj in aseq:
                self.C[:,:,jj]=self.C[:,:,jj]-centreLambda[jj]*self.C[:,:,jj]@datain1.transpose()@datain1@self.C[:,:,jj]/(1+centreLambda[jj]*datain1@self.C[:,:,jj]@datain1.transpose())
                A1=self.A[:,:,jj].transpose()+centreLambda[jj]*self.C[:,:,jj]@datain1.transpose()@(yin-datain1@self.A[:,:,jj].transpose())
                self.A[:,:,jj]=A1.transpose()
            
    def Testing(self,data):
        L1,W1=data.shape 
        Ye=numpy.zeros((L1,self.Wo))
        for ii in range(0,L1):
            datain=data[ii,:].copy().reshape(1,self.W)
            datain1=numpy.append([[1]],datain,axis=1)
            global_mean1=(self.global_mean*self.L+datain)/(self.L+1)
            global_X1=(self.global_X*self.L+numpy.power(datain,2))/(self.L+1)
            self.global_delta=abs(global_X1-numpy.power(global_mean1,2))
            localDensity,centreLambda=self.firingstrength(datain)
            Ye[ii,:]=self.outputgeneration(datain1,localDensity)
        return Ye
            
    def activtingruleselection(self,localDensity):
        seq=numpy.sort(localDensity.copy())
        seq2=numpy.argsort(localDensity.copy())
        LDS=numpy.sum(numpy.triu(numpy.matlib.repmat(seq.reshape(1,self.numRule),self.numRule,1)),axis=1)
        seq1=[i for i in range(0,self.numRule) if LDS[i]>=self.Gamma0*numpy.sum(localDensity)]
        seqa=seq2[seq1[-1]:self.numRule]
        localDensity1=numpy.zeros((self.numRule,))
        localDensity1[seqa]=localDensity[seqa]
        centreLambda1=localDensity1/numpy.sum(localDensity1)
        return seqa,localDensity1,centreLambda1
    
        
    def outputgeneration(self,datain1,localDensity):
        y=numpy.zeros((1,self.Wo))
        aseq,localDensity,centreLambda=self.activtingruleselection(localDensity)
        for ii in aseq:
            y=y+datain1@self.A[:,:,ii].transpose()*centreLambda[ii]
        return y
        

    def firingstrength(self,datain):
        datain1=numpy.sum(numpy.power((self.prototyps-datain),2),axis=1)
        global_delta1=numpy.sum((self.local_delta+self.global_delta)/2,axis=1)
        localDensity=numpy.array(numpy.exp(datain1/global_delta1*-1))
        centreLambda=localDensity/numpy.sum(localDensity)
        return localDensity,centreLambda
    
