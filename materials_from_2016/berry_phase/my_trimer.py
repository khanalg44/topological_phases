#!/usr/bin/env python

from pythtb import *

def main():
    lat=[[1.0,0.0],[0.0,1.0]]
    r=1.0;
    tpio3=2.*np.pi/3.0; tpio4=2.*np.pi/4.0; 
    #tpio3=120; tpio4=90;

    orb=[]
    for i in range(3):
        angle=tpio4+(i+1)*tpio3 # i=1: bottom left atom 
        atom_pos=[r*np.cos(angle), r*np.sin(angle)]
