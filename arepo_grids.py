######################################################################
#                            LIBRARIES                               #
######################################################################

#Standard libs
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Own libs
from .units import *
from .physical_functions import *
from .plotting_functions import *
from . import binary_read as rsnap
from . import binary_write as wsnap

######################################################################
#                       CLASS: ArepoVoronoiGrid                      #
######################################################################

class ArepoVoronoiGrid:
    
    ######################################################################
    #                           CONSTRUCTOR                              #
    ######################################################################
    
    def __init__(self, file_path=None, verbose=False):
        
        #FILE PATH
        self.file_path = str
        
        #CREATE VARIABLES FOR THE WHOLE GRID
        #Raw data
        self.data = []
        self.header = []
        #Extracted data
        self.size = int
        self.time = float
        self.ntot = int
        self.ngas = int
        self.nsink = int
        self.gas_ids = []
        self.sink_ids = []
        self.position = []  #in code units
        self.velocity = []  #in code units
        self.mass = []  #in code units
        self.chem = []  #in code units
        self.rho = []  #in code units
        self.utherm = []  #in code units
        
        #SET VARIABLES
        if(not file_path==None):
            self.read_file(file_path,verbose)
        
    ######################################################################
    #                       READ AND WRITE FUNCTIONS                     #
    ######################################################################
    
    #Reading in data from the initial conditions file
    def read_file(self, file_path, verbose=False):

        if(verbose):
            print("Loading AREPO Voronoi grid from \"{}\" ...".format(file_path))
        
        #READING THE DATA AND SETTING VARIABLES
        self.file_path = file_path
        self.data, self.header = rsnap.read_snapshot(self.file_path)
        self.size = int(self.header['boxsize'][0])
        self.time = float(self.header['time'][0])
        self.ntot = int(sum(self.header['num_particles']))
        self.ngas = int(self.header['num_particles'][0])
        self.nsink = int(self.header['num_particles'][5])
        self.gas_ids = np.arange(0,self.ngas)
        self.sink_ids = np.arange(self.ntot-self.nsink,self.ntot)
        self.position = self.data['pos']
        self.velocity = self.data['vel']
        self.mass = self.data['mass']
        self.chem = self.data['chem']
        self.rho = self.data['rho']
        self.utherm = self.data['u_therm']
        
        if(verbose):
            print("Size: {}^3".format(self.size))
            print("# of particles: {}".format(self.ntot))
            print("# of gas particles: {}".format(self.ngas))
            print("# of sink particles: {}".format(self.nsink))
            if(self.header['flag_doubleprecision']):
                  print("Precision: double")
            else:
                  print("Precision: float")
                  
        if(verbose):
            print("Completed loading AREPO Voronoi grid from \"{}\"".format(file_path))
        
    #Reading out data into an initial conditions file
    def write_file(self, file_path, io_flags=None, verbose=False):

        #SETTING THE IO FLAGS
        if(io_flags==None):
            io_flags = {'mc_tracer'           : False,
                        'time_steps'          : False,
                        'sgchem'              : True,
                        'variable_metallicity': False}
        wsnap.io_flags = io_flags

        #WRITING THE FILE
        if(verbose):
            print("Writing AREPO Voronoi grid to \"{}\" ...".format(file_path))
        wsnap.write_snapshot(file_path, self.header, self.data)
        if(verbose):
            print("Completed writing AREPO Voronoi grid to \"{}\"".format(file_path))

    ######################################################################
    #                            CALC PROPERTIES                         #
    ######################################################################

    #Number density. output: 1d array, units: in cm^-3
    def ndensity(self):
        ndensity = calc_number_density(self.rho)
        return ndensity

    #Temperature. output: 1d array, units: in K
    def temperature(self):
        nTOT = calc_chem_density(self.chem, self.ndensity())[4]
        temperature = calc_temperature(self.rho, self.utherm, nTOT)
        return temperature
    
    ######################################################################
    #                           ????? FUNCTIONS                          #
    ######################################################################
    
    def interpolate_density(self):
        
        from scipy.interpolate import LinearNDInterpolator
        
        print("NOTE: Started creating density interpolation function...")
        pos_cgs = self.position[0:self.ngas]*uparsec
        grid_density_interp = NearestNDInterpolator(pos_cgs,self.rho)
        print("NOTE: Completed creating density interpolation function.")
        
        return grid_density_interp
                
    ######################################################################
    #                         PLOTTING FUNCTIONS                         #
    ######################################################################
    
    #PLOTTING IDEA: tricontourf for ndensity? -- Too computing expensive?
        
    def plot_projection(self, projection='z', log=False, ROI=None):
        
        print("NOTE: Plotting projection of voronoi cell centers below...")

        #PLOTTING OPTIONS
        my_cmap = mpl.cm.get_cmap('magma') 
        if(log):
            my_cmap.set_bad((0,0,0))
            my_norm = mpl.colors.LogNorm()
        else:
            my_norm = mpl.colors.Normalize()

        xpos = self.position[:,0]*uparsec
        ypos = self.position[:,1]*uparsec
        zpos = self.position[:,2]*uparsec
        
        #MAKING THE PLOTS
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        ax.tick_params(axis='both', which='major', labelsize='xx-large')
        if(projection=='x' or projection=='X'):
            #ax.set_title(r"$\mathrm{Histogram~of~X~projection~of~cell~centers}$",fontsize='xx-large')
            ax.hist2d(ypos, zpos, cmap=my_cmap, norm=my_norm, bins=1000, weights=self.mass)
            ax.set_xlabel(r"$\mathrm{Y~[parsec]}$",fontsize='xx-large')
            ax.set_ylabel(r"$\mathrm{Z~[parsec]}$",fontsize='xx-large')
        elif(projection=='y' or projection=='Y'):
            #ax.set_title(r"$\mathrm{Histogram~of~Y~projection~of~cell~centers}$",fontsize='xx-large')
            ax.hist2d(xpos, zpos, cmap=my_cmap, norm=my_norm, bins=1000, weights=self.mass)
            ax.set_xlabel(r"$\mathrm{X~[parsec]}$",fontsize='xx-large')
            ax.set_ylabel(r"$\mathrm{Z~[parsec]}$",fontsize='xx-large')
        elif(projection=='z' or projection=='Z'):
            #ax.set_title(r"$\mathrm{Histogram~of~Z~projection~of~cell~centers}$",fontsize='xx-large')
            ax.hist2d(xpos, ypos, cmap=my_cmap, norm=my_norm, bins=1000, weights=self.mass)
            ax.set_xlabel(r"$\mathrm{X~[parsec]}$",fontsize='xx-large')
            ax.set_ylabel(r"$\mathrm{Y~[parsec]}$",fontsize='xx-large')

        if(not ROI==None):
            #This part plots a red square around the ROI (Region of Interest) in the simulation
            #The format of ROI is [xmin,xmax,ymin,ymax]
            xmin, xmax = ROI[0], ROI[1]
            ymin, ymax = ROI[2], ROI[3]
            ax.vlines(x=xmin*uparsec, ymin=ymin*uparsec, ymax=ymax*uparsec, color='red')
            ax.vlines(x=xmax*uparsec, ymin=ymin*uparsec, ymax=ymax*uparsec, color='red')
            ax.hlines(y=ymin*uparsec, xmin=xmin*uparsec, xmax=xmax*uparsec, color='red')
            ax.hlines(y=ymax*uparsec, xmin=xmin*uparsec, xmax=xmax*uparsec, color='red')
            
    ##########################     PDF    ################################
    ######################################################################
    
    def plot_density_PDF(self, color='k', label=None, title=None, fit_spline=False, only_spline=False, fit_gauss=False, save_png=None):

        if(label==None):
            label=self.file_path
            
        plot_density_PDFs(arepo_voronoi_grids=[self],
                          colors=[color],
                          labels=[label],
                          title=title,
                          fit_spline=fit_spline,
                          only_spline=only_spline,
                          fit_gauss=fit_gauss, 
                          save_png=save_png)
        
    def plot_ndensity_PDF(self, color='k', label=None, title=None, fit_spline=False, only_spline=False, save_png=None):
        if(label==None):
            label=self.file_path
            
        plot_ndensity_PDFs(arepo_voronoi_grids=[self],
                          colors=[color],
                          labels=[label],
                          title=title,
                          fit_spline=fit_spline,
                          only_spline=only_spline,
                          save_png=save_png)
        
    ##########################    Sinks   ################################
    ######################################################################
    
    def plot_sink_mass_histogram(self, color='k', label=None, title=None, save_png=None):
        
        if(label==None):
            label=self.file_path
        
        plot_sink_mass_histograms(arepo_voronoi_grids=[self],
                                  colors=[color],
                                  labels=[label],
                                  title=title,
                                  save_png=save_png)

      
    ######################################################################
    #                        CLOUD-BASED FUNCTIONS                       #
    ######################################################################

    #Extracting cloud from the simulation region            
    def create_cloud(self, cloud_size=600, cloud_Tmax=30):
        
        #CREATE VARIABLES FOR THE CLOUD IN THE GRID
        #
        self.cloud_size = int
        self.cloud_Tmax = float
        #
        self.cloud_ids = []
        self.non_cloud_ids = []
        self.cloud_ntot = int
        #
        self.cloud_position = []  #in code units
        self.cloud_velocity = []  #in code units
        self.cloud_mass = []  #in code units
        self.cloud_chem = []  #in code units
        self.cloud_rho = []  #in code units
        self.cloud_utherm = []  #in code units
        #
        self.cloud_ndensity = []  #in cm^-3
        self.cloud_temperature = []  #in K
        
        
        #DEFINING BOUNDARIES OF THE CLOUD (a cubical region containing it)
        self.cloud_size = cloud_size
        cloud_min = int(self.size/2 - self.cloud_size/2)
        cloud_max = int(self.size/2 + self.cloud_size/2)
        
        #SETTING MAXIMUM TEMPERATURE FOR THE CLOUD
        self.cloud_Tmax = cloud_Tmax
    
        
        #EXTRACTING THE CLOUD FROM THE GAS
        print("NOTE: Started extracting cloud particle indices...")
        temps = self.temperature() #For checking temperature of cell
        for i in range(self.ntot):
            if(i%500000==0):
                print(str(i)+"/"+str(self.ntot)+" total cells") #counter
            pos = self.position[i]
            #This is a first filter based on cloud size (the box within which the cloud is)
            if(cloud_min<pos[0]<cloud_max and cloud_min<pos[1]<cloud_max and cloud_min<pos[2]<cloud_max):
                #This is a second filter based on temperature (to remove "outliers" from cloud size box)
                temp = temps[i]
                if(temp<cloud_Tmax):
                    self.cloud_ids.append(i)
        self.non_cloud_ids = np.delete(np.arange(self.ntot), self.cloud_ids)
        print("NOTE: Completed extracting cloud particle indices.")
        self.cloud_ntot = len(self.cloud_ids)
        print("NOTE: The number of cloud particles is "+str(self.cloud_ntot)+".")
        
        
        #SETTING CLOUD VARIABLES FOR EASE OF USE
        self.cloud_position = self.position[self.cloud_ids]
        self.cloud_velocity = self.velocity[self.cloud_ids]
        self.cloud_mass = self.mass[self.cloud_ids]
        self.cloud_chem = self.chem[self.cloud_ids]
        self.cloud_rho = self.rho[self.cloud_ids]
        self.cloud_utherm = self.utherm[self.cloud_ids]
        self.cloud_ndensity = self.ndensity()[self.cloud_ids]
        self.cloud_temperature = self.temperature()[self.cloud_ids]
        
    #Updating velocity in the cloud (requires a turbulent velocity field input)    
    def update_cloud_velocity(self, turbulentvelocitygrid=None, virial_ratio=0.5):
        
        from scipy.interpolate import RegularGridInterpolator
        
        tvg = turbulentvelocitygrid

        #CREATING INTERPOLATION FUNCTION
        print("NOTE: Started creating interpolation functions...")
        #Creating grid for interpolating
        cloud_min = int(self.size/2 - self.cloud_size/2)
        cloud_max = int(self.size/2 + self.cloud_size/2)
        scaling = float(self.cloud_size)/float(tvg.size) #Scaling velocity gridsize to cloud gridsize
        range_vals = np.arange(cloud_min,cloud_max,scaling)
        #Interpolating the data
        interp_x = RegularGridInterpolator([range_vals]*3,tvg.vx,method='linear')
        interp_y = RegularGridInterpolator([range_vals]*3,tvg.vy,method='linear')
        interp_z = RegularGridInterpolator([range_vals]*3,tvg.vz,method='linear')
        print("NOTE: Completed creating interpolation functions.")

        #CREATING VELOCITY ARRAY
        print("NOTE: Started creating velocity array...")

        #Filling with all zero velocity first
        velocity_new = np.zeros((self.ntot,3)) 
        vx_s = interp_x(self.cloud_position)
        vy_s = interp_y(self.cloud_position)
        vz_s = interp_z(self.cloud_position)
        velocity_new[self.cloud_ids] = np.array([vx_s,vy_s,vz_s]).T
        print("NOTE: Completed creating velocity array.")
        #Note that the velocity array is in arbitrary units so will have to be scaled    


        #SCALING VELOCITIES TO A SPECIFIC VIRIAL RATIO
        print("NOTE: Started scaling velocity array to satisfy virial ratio "+str(virial_ratio)+"...")
        #Finding the scaling factor first
        E_k = calc_kinetic_energy(self.cloud_mass, velocity_new[self.cloud_ids])
        E_g = calc_gravitational_potential_energy(self.cloud_mass, self.cloud_rho)
        Q = E_k / E_g  #Virial ratio (0.5 for virial equilibrium)
        scaling = np.sqrt(virial_ratio/Q)
        print("NOTE: The scaling factor is "+str(scaling)+".")
        #Scaling the velocity here
        velocity_new = scaling*velocity_new
        print("NOTE: Completed scaling velocity array to satisfy virial ratio "+str(virial_ratio)+".")


        #CHUCKING VELOCITIES INTO VARIABLES
        self.velocity = velocity_new
        self.cloud_velocity = self.velocity[self.cloud_ids]
        self.data['vel'] = self.velocity
                  
######################################################################
#                        CLASS: ArepoCubicGrid                       #
######################################################################
                  
class ArepoCubicGrid:
    
    ######################################################################
    #                           CONSTRUCTORS                             #
    ######################################################################
                
    def __init__(self, file_path=None, AREPO_bounds=None, verbose=False):
        

        ######################################################################
        #                              Vars                                  #
        ######################################################################
        
        #Data properties
        self.file_path = str
        self.density =[[[]]]      #3D array
        self.nx = int             #number of data-points along x
        self.ny = int             #number of data-points along y
        self.nz = int             #number of data-points along z
        self.ndensity = [[[]]]    #3D array, units: cm^-3

        
        #Scale-related stuff
        self.scale = str          #"pixel" or "parsec"
        self.xmin = float         #minimum x value, units: pixels OR parsec
        self.xmax = float         #maximum x value, units: pixels OR parsec 
        self.ymin = float         #minimum y value, units: pixels OR parsec
        self.ymax = float         #maximum y value, units: pixels OR parsec 
        self.zmin = float         #minimum z value, units: pixels OR parsec
        self.zmax = float         #maximum z value, units: pixels OR parsec 
        self.xlength = float      #length of one grid cube along x, units: parsec OR pixels
        self.ylength = float      #length of one grid cube along y, units: parsec OR pixels
        self.zlength = float      #length of one grid cube along z, units: parsec OR pixels

        ######################################################################
        #                              Init                                  #
        ######################################################################

        #Initialize everything
        if(not file_path==None):
            self.read_from_binary(file_path, verbose)

            if(not AREPO_bounds==None):
                self.set_scale("parsec",AREPO_bounds)

        
    ######################################################################
    #                       READ AND WRITE FUNCTIONS                     #
    ######################################################################

    #Reading in the grid from a binary file
    def read_from_binary(self, file_path, verbose=False):

        if(verbose):
            print("Loading AREPO Cubic grid from \"{}\" ...".format(file_path))
            
        self.file_path = file_path
        #Read 3D data
        self.density = rsnap.read_grid(self.file_path)
        #Save dimensions
        self.nx, self.ny, self.nz = self.density.shape
        #Calculate ndensity for ease of use
        self.ndensity = calc_number_density(self.density)

        #Default scale
        self.set_scale("pixels")

        if(verbose):
            print("Completed loading AREPO Voronoi grid from \"{}\"".format(file_path))

    #Setting a scale for the grid. Useful!
    def set_scale(self, scale, AREPO_bounds=None):

        if(scale=="pc" or scale=="parsec" or scale=="parsecs"):
            self.scale = "parsec"

            if(AREPO_bounds==None):
                print("Please mention AREPO_bounds!")
            else:
                #Setting x, y and z min/max values
                self.xmin, self.xmax = (AREPO_bounds[0]*uparsec, AREPO_bounds[1]*uparsec)
                self.ymin, self.ymax = (AREPO_bounds[2]*uparsec, AREPO_bounds[3]*uparsec)
                self.zmin, self.zmax = (AREPO_bounds[4]*uparsec, AREPO_bounds[5]*uparsec)

                #Actual length of each grid cell
                self.xlength = float(self.xmax - self.xmin)/float(self.nx)
                self.ylength = float(self.ymax - self.ymin)/float(self.ny)
                self.zlength = float(self.zmax - self.zmin)/float(self.nz)                

        elif(scale=="px" or scale=="pixel" or scale=="pixels"):
            self.scale = "pixel"
            self.xmin, self.xmax = (0, self.nx)
            self.ymin, self.ymax = (0, self.ny)
            self.zmin, self.zmax = (0, self.nz)
            self.xlength = 1
            self.ylength = 1
            self.zlength = 1   

        else:
            print("Incorrect scale option!") 
    
    #Writing the grid into a FITS file
    def write_to_FITS(self, FITS_file_path, verbose=False):
        
        from astropy.io import fits
        
        if(verbose):
            print("Writing AREPO Cubic grid to FITS at \"{}\" ...".format(FITS_file_path))
            
        hdu = fits.PrimaryHDU(data=self.density)
        hdu.writeto(FITS_file_path, overwrite=True)
        #Setting overwrite to True so it doesn't complain about "file existing"

        if(verbose):
            print("Completed writing AREPO Cubic grid to FITS at \"{}\"".format(FITS_file_path))

        #if(check):
        #    print("NOTE: Printing data from the FITS file below...")
        #    data = fits.open(grid_FITS_file_path)[0].data
        #    print(data)
            
            
    #Writing a mask to the FITS file that masks things outside (xmin,xmax), (ymin,ymax) and (zmin,zmax)
    def write_mask_to_FITS(self, mask_FITS_file_path, xmin, xmax, ymin, ymax, zmin, zmax, verbose=False):
        
        from astropy.io import fits

        if(verbose):
            print("Writing mask to FITS at \"{}\" ...".format(mask_FITS_file_path))
            
        mask_data = np.full((self.nx,self.ny,self.nz), 1, dtype=int)
        mask_data[xmin:xmax,ymin:ymax,zmin:zmax] = 0
        hdu = fits.PrimaryHDU(data=mask_data)
        hdu.writeto(mask_FITS_file_path, overwrite=True)
        #Setting overwrite to True so it doesn't complain about "file existing"

        if(verbose):
            print("Completed writing mask to FITS at \"{}\"".format(mask_FITS_file_path))

        #if(check):
        #    print("NOTE: Printing mask from the FITS file below...")
        #    data = fits.open(mask_FITS_file_path)[0].data
        #    print(data[xmin:xmin+5,ymin:ymin+5,zmin:zmin+5])
    
    ######################################################################
    #                         PLOTTING FUNCTIONS                         #
    ######################################################################
    
    
    def plot_grid(self, ndensity=False, log=False, mask_below=None, network=None, arepo_voronoi_grid=None, min_sink_mass=0.0, save_png=None):

        print("NOTE: Plotting 3D grid below...")

        #Density or number density
        if(ndensity):
            data = self.ndensity
        else:
            data = self.density

        #Masking below a certain threshold
        if(not mask_below==None):
            data = np.ma.masked_array(data,mask=data<mask_below)
           
        #Log or linear
        if(log):
            data = np.log10(data, out=np.zeros_like(data), where=(data>0))

        #Creating figure
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')

        #Plotting options
        ax.set_xlabel(r"$\mathrm{X~[parsec]}$",fontsize='xx-large',labelpad=12)
        ax.set_ylabel(r"$\mathrm{Y~[parsec]}$",fontsize='xx-large',labelpad=15)
        ax.set_zlabel(r"$\mathrm{Z~[parsec]}$",fontsize='xx-large',labelpad=20)
        ax.tick_params(axis='x', labelsize='xx-large')
        ax.tick_params(axis='y', labelsize='xx-large')
        ax.tick_params(axis='z', labelsize='xx-large', pad=10)
        ax.set_xlim(self.xmin,self.xmax)
        ax.set_ylim(self.ymin,self.ymax)
        ax.set_zlim(self.zmin,self.zmax)
        my_cmap_1 = mpl.cm.get_cmap('viridis')
        my_cmap_2 = mpl.cm.get_cmap('gray')
        alpha_1 = 0.1
        alpha_2 = 0.4

        #CONTOUR PLOTS
        ticks_x = np.linspace(self.xmin, self.xmax, num=self.nx, endpoint=True)
        ticks_y = np.linspace(self.ymin, self.ymax, num=self.ny, endpoint=True)
        ticks_z = np.linspace(self.zmin, self.zmax, num=self.nz, endpoint=True)

        #x projection
        y, z = np.meshgrid(ticks_y,ticks_z)
        contour_x = np.sum(data,axis=0).T #The transpose is CRUCIAL due to how contourf works!
        ax.contourf(contour_x, y, z, zdir='x', levels=20, offset=self.xmin, cmap=my_cmap_2, alpha=alpha_2)
        
        #y projection
        x, z = np.meshgrid(ticks_x,ticks_z)
        contour_y = np.sum(data,axis=1).T #The transpose is CRUCIAL due to how contourf works!
        ax.contourf(x, contour_y, z, zdir='y', levels=20, offset=self.ymax, cmap=my_cmap_2, alpha=alpha_2)
        
        #z projection
        x, y = np.meshgrid(ticks_x,ticks_y)
        contour_z = np.sum(data,axis=2).T #The transpose is CRUCIAL due to how contourf works!
        ax.contourf(x, y, contour_z, zdir='z', levels=20, offset=self.zmin, cmap=my_cmap_2, alpha=alpha_2)

        #3D PLOTS
        X, Y, Z = np.meshgrid(ticks_x,ticks_y,ticks_z)
        plot_data = np.swapaxes(data,0,1).flatten() #The swapping of axes is how matplotlib works. NECESSARY!
        ax.scatter(X, Y, Z, c=plot_data, cmap=my_cmap_1, alpha=alpha_1, s=1)

        #PLOTTING FILAMENTS IF REQUIRED
        if network is None:
            pass
        else:
            if(network.scale != self.scale):
                print("Network and Arepo scales don't match!")
            else:   
                for fil in network.fils:
                    x = fil.samps[:,0]
                    y = fil.samps[:,1]
                    z = fil.samps[:,2]
                    ax.scatter(x[0], y[0], z[0], linewidth=1, c='black', s=3, zorder=2)
                    ax.scatter(x[-1], y[-1], z[-1], linewidth=1, c='black', s=3, zorder=2)
                    ax.plot(x, y, z, c='red')

        #PLOTTING SINKS IF REQUIRED
        if arepo_voronoi_grid is None:
            pass
        else:
            if(self.scale != "parsec"):
                print("Can't plot sinks since grid is not on parsec scale")
            else:
                ids = arepo_voronoi_grid.sink_ids
                sink_mass = arepo_voronoi_grid.mass[ids]
                sink_pos = arepo_voronoi_grid.position[ids]
                #Only plot above a certain minimum sink mass (in solar masses)
                trunc_sink_mass = sink_mass[sink_mass>min_sink_mass]
                trunc_sink_pos = sink_pos[sink_mass>min_sink_mass]
                x = trunc_sink_pos[:,0]*uparsec
                y = trunc_sink_pos[:,1]*uparsec
                z = trunc_sink_pos[:,2]*uparsec
                s = 2.0 + 5.0*trunc_sink_mass/trunc_sink_mass.max()
                ax.scatter(x, y, z, s=3, c='red', zorder=2)

        #To save the plot!
        if(not save_png==None):
            fig.savefig(save_png,bbox_inches='tight',dpi=300)
        
    def plot_slice(self, x=None, y=None, z=None, ndensity=False, log=False, save_png=None):

        print("NOTE: Plotting 2D grid slice below...")

        #Density or number density
        if(ndensity):
            data = self.ndensity
        else:
            data = self.density

        #Creating figure
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)

        #Plotting options
        my_cmap = mpl.cm.get_cmap('magma')
        ax.tick_params(axis='both', which='major', labelsize='xx-large')
        #Log or linear
        if(log):
            my_cmap.set_bad((0,0,0))
            my_norm = mpl.colors.LogNorm()
        else:
            my_norm = mpl.colors.Normalize()
 
        #MAKING THE PLOTS       
        if(x==None and y==None and z==None):
            print("Please select an axis and an integer slice to plot!")
            return
        elif(not x==None):
            #ax.set_title("X-slice of grid at x = {}".format(x))
            plot_data = data[x,:,:].T #The transpose is CRUCIAL due to how imshow works!
            cb = ax.imshow(plot_data, cmap=my_cmap, norm=my_norm, origin='lower', extent=[self.ymin,self.ymax,self.zmin,self.zmax])
            ax.set_xlabel(r"$\mathrm{{Y~[{}]}}$".format(self.scale),fontsize='xx-large')
            ax.set_ylabel(r"$\mathrm{{Z~[{}]}}$".format(self.scale),fontsize='xx-large')
        elif(not y==None):
            #ax.set_title("Y-slice of grid at y = {}".format(y))
            plot_data = data[:,y,:].T #The transpose is CRUCIAL due to how imshow works!
            cb = ax.imshow(plot_data, cmap=my_cmap, norm=my_norm, origin='lower', extent=[self.xmin,self.xmax,self.zmin,self.zmax])
            ax.set_xlabel(r"$\mathrm{{X~[{}]}}$".format(self.scale),fontsize='xx-large')
            ax.set_ylabel(r"$\mathrm{{Z~[{}]}}$".format(self.scale),fontsize='xx-large')
        elif(not z==None):
            #ax.set_title("Z-slice of grid at z = {}".format(z))
            plot_data = data[:,:,z].T #The transpose is CRUCIAL due to how imshow works!
            cb = ax.imshow(plot_data, cmap=my_cmap, norm=my_norm, origin='lower', extent=[self.xmin,self.xmax,self.ymin,self.ymax])
            ax.set_xlabel(r"$\mathrm{{X~[{}]}}$".format(self.scale),fontsize='xx-large')
            ax.set_ylabel(r"$\mathrm{{Y~[{}]}}$".format(self.scale),fontsize='xx-large')

        #More plotting options
        cbar = fig.colorbar(cb, ax=ax,fraction=0.046, pad=0.02)
        if(ndensity):
            cbar.set_label(r'$\mathrm{Density~[cm}^{-3}]$',size='xx-large')
        else:
            cbar.set_label(r'$\mathrm{Density~[code~units]}$',size='xx-large')
        cbar.ax.tick_params(labelsize='x-large')
        cbar.ax.yaxis.get_offset_text().set_fontsize('large')

        #To save the plot!
        if(not save_png==None):
            fig.savefig(save_png,bbox_inches='tight',dpi=300) 

    def plot_projection(self, projection='z', ndensity=False, log=False, mask_below=None, network=None, bifurcations=False, arepo_voronoi_grid=None, min_sink_mass=0, save_png=None):

        print("NOTE: Plotting 2D projection of density with filaments")

        #Density or number density
        if(ndensity):
            data = self.ndensity
        else:
            data = self.density

        #Log or linear
        #if(log):
        #    data = np.log10(data, out=np.zeros_like(data), where=(data>0))

        #Masking below a certain threshold
        if(not mask_below==None):
            data = np.ma.masked_array(data,mask=data<mask_below)

        #The type of projection to make (axis here corresponds to numpy)
        if(projection=='x'):
            PLOT_proj_axis = 0 #x
            PLOT_x_axis = 1 #y
            PLOT_y_axis = 2 #z

            PLOT_x_min = self.ymin
            PLOT_x_max = self.ymax
            PLOT_y_min = self.zmin
            PLOT_y_max = self.zmax

            PLOT_x_label = "Y"
            PLOT_y_label = "Z"

        elif(projection=='y'):
            PLOT_proj_axis = 1 #y
            PLOT_x_axis = 0 #x
            PLOT_y_axis = 2 #z

            PLOT_x_min = self.xmin
            PLOT_x_max = self.xmax
            PLOT_y_min = self.zmin
            PLOT_y_max = self.zmax

            PLOT_x_label = "X"
            PLOT_y_label = "Z"

        elif(projection=='z'):
            PLOT_proj_axis = 2 #z
            PLOT_x_axis = 0 #x
            PLOT_y_axis = 1 #y

            PLOT_x_min = self.xmin
            PLOT_x_max = self.xmax
            PLOT_y_min = self.ymin
            PLOT_y_max = self.ymax
            
            PLOT_x_label = "X"
            PLOT_y_label = "Y"

        else:
            print("Projection not well defined!")
            return

        #Creating figure
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)

        #Plotting options
        #my_cmap = mpl.cm.get_cmap('gist_heat')
        my_cmap = mpl.cm.get_cmap('gist_gray')
        ax.tick_params(axis='both', which='major', labelsize='xx-large')
        ax.set_xlabel(r"$\mathrm{{{0}~[{1}]}}$".format(PLOT_x_label,self.scale),fontsize='xx-large')
        ax.set_ylabel(r"$\mathrm{{{0}~[{1}]}}$".format(PLOT_y_label,self.scale),fontsize='xx-large')
        if(log):
            if(not mask_below==None):
                data = np.log10(data, out=np.zeros_like(data), where=(data>0))
                my_norm = mpl.colors.Normalize()
            else:
                my_cmap.set_bad((0,0,0))
                my_norm = mpl.colors.LogNorm()
        else:
            my_norm = mpl.colors.Normalize()
        my_extent = [PLOT_x_min,PLOT_x_max,PLOT_y_min,PLOT_y_max]

        #DATA TO PLOT
        plot_data = np.sum(data,axis=PLOT_proj_axis)
        plot_data = plot_data.T #THIS IS IMPORTANT FOR HOW CONTOURF and IMSHOW WORKS!!!

        #MAKING THE PLOTS
        if(not mask_below==None):
            ax.set_facecolor('black')
            #ax.imshow(plot_data, extent=my_extent, cmap='gray', origin='lower', zorder=-1)
            cb = ax.contourf(plot_data, levels=10, extent=my_extent, cmap=my_cmap, norm=my_norm, origin='lower', zorder=0)
        else:
            cb = ax.imshow(plot_data, extent=my_extent, cmap=my_cmap, norm=my_norm, origin='lower', zorder=0)

        #PLOTTING FILAMENTS IF REQUIRED
        if network is None:
            pass
        else:
            if(network.scale != self.scale):
                print("Network and Arepo scales don't match!")
            else:   
                for fil in network.fils:
                    x = fil.samps[:,PLOT_x_axis]
                    y = fil.samps[:,PLOT_y_axis]
                    ax.plot(x, y, linewidth=1.5, zorder=1)
                    ax.scatter(x[0], y[0], linewidth=1, c='black', s=2, zorder=2)
                    ax.scatter(x[-1], y[-1], linewidth=1, c='black', s=2, zorder=2)
                    #if(annotate):
                    #    ax.text(pos_x[fil.nsamp//2], pos_y[fil.nsamp//2], fil.idx, color='orange', fontsize=12) 
                    
                if(bifurcations):
                    for bp in network.bifurcations:
                        x = bp.pos[PLOT_x_axis]
                        y = bp.pos[PLOT_y_axis]
                        ax.scatter(x, y, c='cyan', s=2, zorder=2)

        #PLOTTING SINKS IF REQUIRED
        if arepo_voronoi_grid is None:
            pass
        else:
            if(self.scale != "parsec"):
                print("Can't plot sinks since grid is not on parsec scale")
            else:
                ids = arepo_voronoi_grid.sink_ids
                sink_mass = arepo_voronoi_grid.mass[ids]
                sink_pos = arepo_voronoi_grid.position[ids]
                #Only plot above a certain minimum sink mass (in solar masses)
                trunc_sink_mass = sink_mass[sink_mass>min_sink_mass]
                trunc_sink_pos = sink_pos[sink_mass>min_sink_mass]
                x = trunc_sink_pos[:,PLOT_x_axis]*uparsec
                y = trunc_sink_pos[:,PLOT_y_axis]*uparsec
                s = 2.0 + 5.0*trunc_sink_mass/trunc_sink_mass.max()
                ax.scatter(x, y, s=s, c='red',zorder=2, alpha=0.5)
        
        #More plotting options
        cbar = fig.colorbar(cb, ax=ax,fraction=0.046, pad=0.02)
        cbar.set_label(r'$\Sigma~\log(n)\mathrm{~[cm}^{-3}]$',size='xx-large')
        cbar.ax.tick_params(labelsize='x-large')
        cbar.ax.yaxis.get_offset_text().set_fontsize('large')
        
        ax.set_xlim(28,30)
        ax.set_ylim(26,27.5)

        #To save the plot!
        if(not save_png==None):
            fig.savefig(save_png,bbox_inches='tight',dpi=300) 

        #ax.annotate(text, xy=(0.98, 0.95), xycoords='axes fraction', fontsize='xx-large',
        #        horizontalalignment='right', verticalalignment='bottom',
        #           bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="k", lw=2))            