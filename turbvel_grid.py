######################################################################
#                            LIBRARIES                               #
######################################################################

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

######################################################################

class TurbulentVelocityGrid:
    
    ######################################################################
    #                           CONSTRUCTOR                              #
    ######################################################################
    
    def __init__(self, file_path=None, check=False):
        
        #FILE PATH
        self.file_path = str
        
        #CREATE VARIABLES
        #Raw data
        self.data = []
        #Extracted data
        self.size = int
        self.vx = [[[]]]
        self.vy = [[[]]]
        self.vz = [[[]]]
        
        #SET VARIABLES
        if(not file_path==None):
            self.read_file(file_path, check)
        
    ######################################################################
    #                       READ AND WRITE FUNCTIONS                     #
    ######################################################################
    
    #Reading in data from the turbulent velocity binary file
    def read_file(self, file_path, check):

        #READING THE DATA
        #Note: the create-turbulence-field code has to be run with parameter "flt" for data types to match
        #between the numpy reading done below and the C++ code output
        print("NOTE: Started reading turbulent velocity field data...")
        self.file_path = file_path
        velocity_data = np.fromfile(self.file_path,'float32')
        print("NOTE: Completed reading turbulent velocity field data.")
        

        #SPLITTING DATA INTO INTO VELOCITY COMPONENTS
        self.size = int(np.cbrt(len(velocity_data)/3))
        velocity_arrays = np.array_split(velocity_data, 3)
        n = self.size
        vx_flat= velocity_arrays[0]
        self.vx = np.reshape(vx_flat,(n,n,n),order='C')
        vy_flat = velocity_arrays[1]
        self.vy = np.reshape(vy_flat,(n,n,n),order='C')
        vz_flat = velocity_arrays[2]
        self.vz = np.reshape(vz_flat,(n,n,n),order='C')
        
        
        #CHECKING THE READING OF DATA
        if(check):
            print("NOTE: Printing data for checking below...")
            #Compare these with the dim/2 slices from the C++ code
            #Note that the slices are all for velocity_z and correspond to dim/2 slices along each x,y,z axis
            print("\nPrinting vz dim/2 x-slice now")
            print(self.vz[:,:,int(n/2)])
            print("\nPrinting vz dim/2 y-slice now")
            print(self.vz[:,int(n/2),:])
            print("\nPrinting vz dim/2 z-slice now")
            print(self.vz[int(n/2),:,:])
                  
    
    ######################################################################
    #                         PLOTTING FUNCTIONS                         #
    ######################################################################
                  
    #Plotting heat map of the velocity components
    def plot_velocity_component(self, x=False, y=False, z=False):

        #PLOTTING OPTIONS
        my_cmap = plt.cm.RdBu
                  
        #MAKING THE PLOTS
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111, projection='3d')
        n = self.size
        X, Y, Z = np.meshgrid(np.arange(n),np.arange(n),np.arange(n))
        ax.set_xlabel(r"$\mathrm{X~[grid~points]}$",fontsize='xx-large',labelpad=15)
        ax.set_ylabel(r"$\mathrm{Y~[grid~points]}$",fontsize='xx-large',labelpad=15)
        ax.set_zlabel(r"$\mathrm{Z~[grid~points]}$",fontsize='xx-large',labelpad=15)
        ax.xaxis.set_tick_params(labelsize='xx-large')
        ax.yaxis.set_tick_params(labelsize='xx-large')
        ax.zaxis.set_tick_params(labelsize='xx-large')
        #The axes x,y,z are likely oriented wrongly here.
        if(x):
            print("NOTE: Plotting vx velocity field below")
            #ax.set_title(r"$\mathrm{Heatmap~of~X~velocity~component}$",fontsize='xx-large')
            ax.scatter(X, Y, Z, c=self.vx.flatten(), cmap=my_cmap)
        if(y):
            print("NOTE: Plotting vy velocity field below")
            #ax.set_title(r"$\mathrm{Heatmap~of~Y~velocity~component}$",fontsize='xx-large')
            ax.scatter(X, Y, Z, c=self.vy.flatten(), cmap=my_cmap)
        if(z):
            print("NOTE: Plotting vz velocity field below")
            #ax.set_title(r"$\mathrm{Heatmap~of~Z~velocity~component}$",fontsize='xx-large')
            ax.scatter(X, Y, Z, c=self.vz.flatten(), cmap=my_cmap)