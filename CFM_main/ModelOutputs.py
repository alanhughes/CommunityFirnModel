#!/usr/bin/env python
'''
Code for isotope diffusion.
'''
import numpy as np 
import json
import scipy.interpolate as interpolate
from constants import *
import os
import sys

class ModelOutputs:
    '''
    Class to handle making the model output files
    '''
    def __init__(self, config, MOd, TWlen, init_time, Glen):
        '''
        Initialize the model output class
        Main variable is Mout_dict, which is a dictionary that contains all
        of the variables that will get written to file.
        '''

        self.c = config
        self.Mout_dict = {}      

        if 'output_bits' not in self.c:
            self.c['output_bits']='float32'
        if 'grid_outputs' not in self.c:
            self.c['grid_outputs'] = False
        self.MOgrid = self.c['grid_outputs']

        if self.MOgrid:
            self.grid_out = np.arange(MOd['z'][0], MOd['z'][-1], self.c['grid_output_res'])

        self.output_list = list(MOd.keys())

        for varname in self.output_list:
            if varname == 'Dcon':
                intkind = 'nearest'
            else:
                intkind = 'linear'
            
            if varname == 'DIP':
                self.Mout_dict[varname] = np.zeros((TWlen+1,8), dtype = self.c['output_bits'])
                self.Mout_dict[varname][0,:]  = np.append(init_time, MOd[varname])
            elif varname == 'BCO':
                self.Mout_dict[varname] = np.zeros((TWlen+1,10), dtype = self.c['output_bits'])
                self.Mout_dict[varname][0,:]  = np.append(init_time, MOd[varname])
            elif varname == 'climate': #[time,bdot,Ts,snowmelt,rain,sublim]
                self.Mout_dict[varname] = np.zeros((TWlen+1,6), dtype = self.c['output_bits'])
                self.Mout_dict[varname][0,:]  = np.append(init_time, MOd[varname])
            # elif varname == 'runoff':
            #     self.Mout_dict[varname] = np.zeros((TWlen+1,2), dtype = self.c['output_bits'])
            #     self.Mout_dict[varname][0,:]  = np.append(init_time, MOd[varname])
            #VV (23/03/2021)
            elif varname == 'refreeze':
                self.Mout_dict[varname] = np.zeros((TWlen+1,2), dtype = self.c['output_bits'])
                self.Mout_dict[varname][0,:]  = np.append(init_time, MOd[varname])
            elif varname == 'runoff':
                self.Mout_dict[varname] = np.zeros((TWlen+1,2), dtype = self.c['output_bits'])
                self.Mout_dict[varname][0,:]  = np.append(init_time, MOd[varname])
            elif varname == 'meltvol':
                self.Mout_dict[varname] = np.zeros((TWlen+1,2), dtype = self.c['output_bits'])
                self.Mout_dict[varname][0,:]  = np.append(init_time, MOd[varname])

            else:
                if self.MOgrid: #gridding outputs
                    if varname == 'z':
                        self.Mout_dict[varname] = np.zeros((len(self.grid_out)+1),dtype=self.c['output_bits'])
                        self.Mout_dict[varname] = np.append(init_time,self.grid_out)
                    else:
                        self.Mout_dict[varname] = np.zeros((TWlen+1,len(self.grid_out)+1),dtype=self.c['output_bits'])

                    if varname in ['LWC', 'refrozen']:
                        self.Mout_dict[varname][0,:] = np.append(init_time,self.RGfun(MOd['z'], MOd[varname], self.grid_out))
                    elif varname != 'z':
                        Ifun = interpolate.interp1d(MOd['z'], MOd[varname], kind = intkind, fill_value='extrapolate')         
                        self.Mout_dict[varname][0,:] = np.append(init_time,Ifun(self.grid_out))
                
                else: #not gridding outputs
                    self.Mout_dict[varname]       = np.zeros((TWlen+1,Glen+1),dtype=self.c['output_bits'])
                    self.Mout_dict[varname][0,:]  = np.append(init_time, MOd[varname])

    def updateMO(self, MOd, mtime, Wtracker):
        '''
        Function to update the output matrices in Mout_dict
        '''

        for varname in self.output_list:
            if varname == 'Dcon':
                intkind = 'nearest'
            else:
                intkind = 'linear'

            if self.MOgrid:
                if varname in ['LWC', 'refrozen']:
                    self.Mout_dict[varname][Wtracker,:] = np.append(mtime,self.RGfun(MOd['z'], MOd[varname], self.grid_out))
                elif ((varname == 'BCO') or (varname == 'DIP') or (varname == 'climate') or (varname == 'runoff') or (varname == 'refreeze') or (varname == 'meltvol')):
                    self.Mout_dict[varname][Wtracker,:] = np.append(mtime,MOd[varname])
                elif varname == 'z':
                    continue
                else:
                    Ifun = interpolate.interp1d(MOd['z'], MOd[varname], kind = intkind,bounds_error=False,fill_value=np.nan)           
                    self.Mout_dict[varname][Wtracker,:] = np.append(mtime,Ifun(self.grid_out))

            else:
                self.Mout_dict[varname][Wtracker,:] = np.append(mtime,MOd[varname])
                

    def RGfun(self, z, var, grid):
        '''
        Function to regrid the variables that can not be linearly interpolated
        e.g. LWC and refrozen need to conserve mass. 

        :param z: the top edge of each node
        :param var: the amount in each node
        :param grid: the top edge of each grid interval
        :return: the amount in each grid interval. An amount is evenly spread over its node.
        The last node is taken to be as thick as the node above it.
        An amount below the end of the last grid interval is not included.
        '''

        node_edges = np.append(z, z[-1] + (z[-1] - z[-2]))
        grid_edges = np.append(grid, grid[-1] + self.c['grid_output_res'])
        amount_above = np.concatenate(([0], np.cumsum(var))) # the amount above each node edge
        return np.diff(np.interp(grid_edges, node_edges, amount_above))



