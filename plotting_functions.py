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
from . import arepo_grids

######################################################################

######################################################################
#                      SINK BASED FUNCTIONS                          #
######################################################################

def plot_density_PDFs(arepo_voronoi_grids, colors=None, labels=None, title=None, fit_spline=False, only_spline=False, fit_gauss=False, save_png=None):

    print("NOTE: Plotting PDFs of the density distribution...")
    
    #PLOTTING OPTIONS
    mpl.rc('text', usetex = True)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize='xx-large')
    ax.set_xlabel(r"$s = \log(\rho/\rho_0)$",fontsize='xx-large')
    ax.set_ylabel(r"$\mathrm{p_s}$",fontsize='xx-large')
    if(not title==None):
        ax.set_title(title,fontsize='xx-large')        

    for AVG, color, label in zip(arepo_voronoi_grids,colors,labels):
        
        #Calculating mass-weighted PDF of the density distribution (see Burkhart, 2018)
        rho_avg = np.average(AVG.rho,weights=AVG.mass[AVG.gas_ids])
        s = np.log10(AVG.rho/rho_avg, out=np.zeros_like(AVG.rho), where=(AVG.rho>0))
        hist, bin_edges = np.histogram(s, bins=25, weights=AVG.mass[AVG.gas_ids], density=True)
        #Plotting histogram of PDF
        if(not only_spline):
            ax.bar(bin_edges[:-1], hist, width=(max(s)-min(s))/len(bin_edges[:-1]), color=color, alpha=0.1, label=label)

        #Fitting cubic spline to the whole PDF
        if(fit_spline):

            from scipy.interpolate import PchipInterpolator

            ndensity_spline = PchipInterpolator(bin_edges[:-1],hist)
            xval = np.linspace(min(bin_edges[:-1]),max(bin_edges[:-1]),1000)
            yval = ndensity_spline(xval)
            #Plotting cubic spline fit
            ax.plot(xval, yval, color=color, linewidth=2.5, label=label)

        #Fitting a Gaussian around the peak of the PDF
        if(fit_gauss):
            
            from scipy.interpolate import PchipInterpolator

            ndensity_spline = PchipInterpolator(bin_edges[:-1],hist)
            xval = np.linspace(min(bin_edges[:-1]),max(bin_edges[:-1]),1000)
            yval = ndensity_spline(xval)

            from scipy.optimize import curve_fit

            gauss=lambda x,mu,sig: (1/(sig*np.sqrt(2*np.pi)))*np.exp((-1/2)*((x-mu)/sig)**2)
            peak_start = xval[yval==max(yval)]-1
            peak_end = xval[yval==max(yval)]+2
            params, pcov = curve_fit(gauss,
                                     xval[(xval>peak_start)&(xval<peak_end)],
                                     yval[(xval>peak_start)&(xval<peak_end)],
                                     p0=[2,3])
            errors = np.sqrt(np.diag(pcov))
            #Plotting Gaussian fit
            xx = np.linspace(peak_start,max(xval)*2,1000)
            yy = gauss(xx,*params)
            ax.plot(xx,yy, color=color, linestyle='--', linewidth=2.5, alpha=0.8)
            #print(params,errors)
            
    ax.legend(loc='best',fontsize='xx-large')
    
    if(not save_png==None):
        fig.savefig(save_png,bbox_inches='tight',dpi=300)
        
def plot_ndensity_PDFs(arepo_voronoi_grids, colors=None, labels=None, title=None, fit_spline=False, only_spline=False, save_png=None):

    print("NOTE: Plotting PDFs of the density distribution...")
    
    #PLOTTING OPTIONS
    mpl.rc('text', usetex = True)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize='xx-large')
    ax.set_xlabel(r"$\mathrm{log}(n)~[\mathrm{cm}^{-3}]$",fontsize='xx-large')
    ax.set_ylabel(r"$\mathrm{p}$",fontsize='xx-large')
    #ax.set_xlim(-2,4)
    #ax.set_ylim(0.001,1)
    if(not title==None):
        ax.set_title(title,fontsize='xx-large')        

    for AVG, color, label in zip(arepo_voronoi_grids,colors,labels):
        
        ndens = calc_number_density(AVG.rho)
        log_ndens = np.log10(ndens, out=np.zeros_like(ndens), where=(ndens>0))
        hist, bin_edges = np.histogram(log_ndens, bins=25, weights=AVG.mass[AVG.gas_ids], density=True)
        #hist = np.cumsum(hist*np.diff(bin_edges))
        #Plotting histogram of PDF
        if(not only_spline):
            ax.bar(bin_edges[:-1], hist, width=(max(log_ndens)-min(log_ndens))/len(bin_edges[:-1]), color=color, alpha=0.1, label=label)
        
        #Fitting cubic spline to the whole PDF
        if(fit_spline):

            from scipy.interpolate import PchipInterpolator

            ndensity_spline = PchipInterpolator(bin_edges[:-1],hist)
            xval = np.linspace(min(bin_edges[:-1]),max(bin_edges[:-1]),1000)
            yval = ndensity_spline(xval)
            #Plotting cubic spline fit
            ax.plot(xval, yval, color=color, linewidth=2.5, label=label)
            
    ax.axvline(x=2.0,color='k',linestyle='--')
            
    ax.legend(loc='best',fontsize='medium')
    #ax.legend()
    
    if(not save_png==None):
        fig.savefig(save_png,bbox_inches='tight',dpi=300)
        
######################################################################
#                      SINK BASED FUNCTIONS                          #
######################################################################

def plot_sink_mass_evolutions(base_file_paths, min_nums, max_nums, colors=None, labels=None, title=None, save_png=None):
    
    from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

    #PLOTTING OPTIONS
    mpl.rc('text', usetex = True)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', which='major', labelsize='xx-large')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel(r"$\mathrm{Time~[Myrs]}$",fontsize='xx-large')
    ax.set_ylabel(r"$\mathrm{Mass~~[M}_\odot]$",fontsize='xx-large')
    if(not title==None):
        ax.set_title(title,fontsize='xx-large')  
    
    for base_file_path, min_num, max_num, color, label in zip(base_file_paths,min_nums,max_nums,colors,labels):
        nums = ["{0:03}".format(i) for i in range(min_num,max_num)]
        counter = 0
        time_array = []
        totsinkmass_array = []
        print("Started reading {}".format(base_file_path))
        for num in nums:
            file_path = base_file_path + num
            try:
                AVG = arepo_grids.ArepoVoronoiGrid(file_path)
            except FileNotFoundError:
                break
            time = AVG.time*utime/(31536000.0*1.e6) #Myrs
            print(counter,end=",")
            counter+=1
            time_array.append(time)
            totsinkmass = np.sum(AVG.mass[AVG.sink_ids])
            totsinkmass_array.append(totsinkmass)
        print("Finished reading {}".format(base_file_path))
        ax.scatter(time_array,totsinkmass_array,color=color,s=3,zorder=1)
        ax.plot(time_array,totsinkmass_array,color=color,label=label,zorder=0)
        
    ax.legend(fontsize='xx-large')
    
    if(not save_png==None):
        fig.savefig(save_png,bbox_inches='tight',dpi=300)
        
def plot_sink_mass_histograms(arepo_voronoi_grids, bins='auto', colors=None, labels=None, title=None, save_png=None):
    
    from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

    print("NOTE: Plotting histograms of sink masses...")
    
    #PLOTTING OPTIONS
    mpl.rc('text', usetex = True)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize='xx-large')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel(r"$\mathrm{Mass~[M}_\odot]$",fontsize='xx-large')
    ax.set_ylabel(r"$\mathrm{Number~of~sinks}$",fontsize='xx-large')
    #ax.set_xlim(0,50)
    if(not title==None):
        ax.set_title(title,fontsize='xx-large')  
    
    for AVG, color, label in zip(arepo_voronoi_grids,colors,labels):
        ax.hist(AVG.mass[AVG.sink_ids],bins=bins,edgecolor=color,label=label,histtype='step',stacked=True)
        
    ax.legend(fontsize='xx-large')
    
    if(not save_png==None):
        fig.savefig(save_png,bbox_inches='tight',dpi=300)
        
def plot_sink_mass_statistics(arepo_voronoi_grids, colors=None, labels=None, title=None, save_png=None):
    
    #'import'ant imports ;)
    from scipy.spatial import KDTree
    from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
    
    #PLOTTING OPTIONS
    mpl.rc('text', usetex = True)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize='xx-large')
    ax.set_xlabel(r"$\mathrm{Simulation~type}$",fontsize='xx-large')
    ax.set_ylabel(r"$\mathrm{Sink~mass~[M}_\odot]$",fontsize='xx-large')
    
    if(not title==None):
        ax.set_title(title,fontsize='xx-large')
        
    sink_masses = []
    for AVG in arepo_voronoi_grids:
        sink_mass = AVG.mass[AVG.sink_ids]
        sink_masses.append(sink_mass)
    bp = ax.boxplot(sink_masses,
                    patch_artist = True, showfliers=True,
                    showmeans = True, meanline = True,
                    whis=1e18, labels = labels)
    #(setting whiskers to unreasonably high value to force min/max to be within whiskers range)
        
    if colors is None:
        pass
    else:
        for patch, color in zip(bp['boxes'],colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
    
    n = np.arange(len(arepo_voronoi_grids))
    
    for (i,median) in zip(n,bp['medians']):
        median.set(color ='black',linewidth = 1,linestyle='-')
        y = median.get_ydata()[0]
        yround = np.round(y,2)
        ax.annotate(r'${}$'.format(yround), (i+1+0.3, y+0.01), fontsize='x-small',  ha='left', va='center')

    for (i,mean) in zip(n,bp['means']):
        mean.set(color ='black',linewidth = 1,linestyle='--')
        y = mean.get_ydata()[0]
        yround = np.round(y,2)
        ax.annotate(r'${}$'.format(yround), (i+1+0.3, y+0.01), fontsize='x-small',  ha='left', va='center')
        
    for whisker in bp['whiskers']:
        whisker.set(color ='black',linewidth = 1,linestyle='-')

    for i in n:
        y = bp['caps'][(2*i)].get_ydata()[0]
        yround = np.round(y,3)
        ax.annotate(r'${}$'.format(yround), (i+1+0.2, y), fontsize='x-small',  ha='left', va='center')
        y = bp['caps'][(2*i)+1].get_ydata()[0]
        yround = np.round(y,3)
        ax.annotate(r'${}$'.format(yround), (i+1+0.2, y), fontsize='x-small',  ha='left', va='center')


    #ax.set_xticklabels(ax.get_xticks(), rotation = 45)
    
    if(not save_png==None):
        fig.savefig(save_png,bbox_inches='tight',dpi=300)  
        
def plot_sink_bifurcation_boxwhiskers(arepo_voronoi_grid, network, nboxes=8, color=None, title=None, save_png=None):
    
    #'import'ant imports ;)
    from scipy.spatial import KDTree
    from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
    
    #PLOTTING OPTIONS
    mpl.rc('text', usetex = True)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    #ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize='xx-large')
    ax.set_xlabel(r"$\mathrm{log(sink~mass)~[M}_\odot]$",fontsize='xx-large')
    ax.set_ylabel(r"$\mathrm{log(distance~to~nearest~hub)~[parsec]}$",fontsize='xx-large')
    
    if(not title==None):
        ax.set_title(title,fontsize='xx-large')
        
    bftree = KDTree([bf.pos for bf in network.bifurcations]) #Creating a k-d tree of bifurcation points
    #Sink stuff
    sink_mass = arepo_voronoi_grid.mass[arepo_voronoi_grid.sink_ids]
    sink_pos = arepo_voronoi_grid.position[arepo_voronoi_grid.sink_ids]*uparsec
    #Sorting
    sink_pos = sink_pos[np.argsort(sink_mass)]
    sink_mass = sink_mass[np.argsort(sink_mass)]
    #Distance of sinks to closest bifurcations
    distances, _ = bftree.query(sink_pos, p=2)
    #Taking the log
    log_sink_mass = np.log10(sink_mass)
    log_distances = np.log10(distances)
    #Split into several boxes
    log_sink_mass_split = np.array_split(log_sink_mass, nboxes)
    log_distances_split = np.array_split(log_distances, nboxes)
    positions = [np.round(sublist[0]+(sublist[-1]-sublist[0])/2,2) for sublist in log_sink_mass_split]
    #print(positions)
    widths = [0.95*abs(np.round(sublist[-1],2)-np.round(sublist[0],2)) for sublist in log_sink_mass_split]
    #print(widths)
    bp = ax.boxplot(log_distances_split, positions=positions, widths=widths, 
                    patch_artist = True, showfliers=True,
                    whis=1e18)
    #(setting whiskers to unreasonably high value to force min/max to be within whiskers range)
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    for median in bp['medians']:
        median.set(color ='black',linewidth = 1,linestyle='--')

    t = np.mean([len(arr) for arr in log_sink_mass_split],dtype=int)
    ax.text(0.01, 0.98,r'$\mathrm{{Number~of~sinks~per~box~\approx~{}}}$'.format(t),transform = ax.transAxes)
    
    ax.set_xticklabels(ax.get_xticks(), rotation = 90)

    if(not save_png==None):
        fig.savefig(save_png,bbox_inches='tight',dpi=300)  


######################################################################
#                   FILAMENT BASED FUNCTIONS                         #
######################################################################   


def plot_filament_length_histogram(networks, bins='auto', colors=None, labels=None, title=None, save_png=None):
    
    from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

    print("NOTE: Plotting histogram of filament lengths below")
    
    #PLOTTING OPTIONS
    mpl.rc('text', usetex = True)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    #ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(axis='both', which='major', labelsize='xx-large')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_ylabel(r"$\mathrm{Counts}$",fontsize='xx-large')
    ax.set_xlabel(r"$\mathrm{Length~[parsec]}$",fontsize='xx-large')
    #ax.set_xlim(0,50)
    if(not title==None):
        ax.set_title(title,fontsize='xx-large')  
    
    for network, color, label in zip(networks,colors,labels):
        lengths = []
        for fil in network.fils:
            lengths.append(fil.length)
        #hist, bin_edges = np.histogram(lengths, bins=bins)
        #ax.bar(bin_edges[:-1], hist, edgecolor=color, label=label, linewidth=3)
        ax.hist(lengths,bins=bins,edgecolor=color,linewidth=3,
                label=label+r'$,~n{}$'.format(len(lengths)),
                histtype='step',stacked=True)
        
    ax.legend(fontsize='xx-large')
    
    if(not save_png==None):
        fig.savefig(save_png,bbox_inches='tight',dpi=300)

def plot_filament_length_statistics(networks, colors=None, labels=None, title=None, save_png=None):
    
    #'import'ant imports ;)
    from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
    
    #PLOTTING OPTIONS
    mpl.rc('text', usetex = True)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize='xx-large')
    ax.set_xlabel(r"$\mathrm{Simulation~type}$",fontsize='xx-large')
    ax.set_ylabel(r"$\mathrm{Filament~length~[parsec]}$",fontsize='xx-large')
    
    if(not title==None):
        ax.set_title(title,fontsize='xx-large')
        
    all_lengths = []
    for network in networks:
        lengths = []
        for fil in network.fils:
            lengths.append(fil.length)
        all_lengths.append(lengths)
    bp = ax.boxplot(all_lengths,
                    patch_artist = True, showfliers=True,
                    showmeans = True, meanline = True,
                    whis=1e18, labels = labels)
    #(setting whiskers to unreasonably high value to force min/max to be within whiskers range)
        
    if colors is None:
        pass
    else:
        for patch, color in zip(bp['boxes'],colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
    
    n = np.arange(len(networks))
    
    for (i,median) in zip(n,bp['medians']):
        median.set(color ='black',linewidth = 1,linestyle='-')
        y = median.get_ydata()[0]
        yround = np.round(y,2)
        ax.annotate(r'${}$'.format(yround), (i+1+0.3, y+0.01), fontsize='x-small',  ha='left', va='center')

    for (i,mean) in zip(n,bp['means']):
        mean.set(color ='black',linewidth = 1,linestyle='--')
        y = mean.get_ydata()[0]
        yround = np.round(y,2)
        ax.annotate(r'${}$'.format(yround), (i+1+0.3, y+0.01), fontsize='x-small',  ha='left', va='center')
        
    for whisker in bp['whiskers']:
        whisker.set(color ='black',linewidth = 1,linestyle='-')

    for i in n:
        y = bp['caps'][(2*i)].get_ydata()[0]
        yround = np.round(y,3)
        ax.annotate(r'${}$'.format(yround), (i+1+0.2, y), fontsize='x-small',  ha='left', va='center')
        y = bp['caps'][(2*i)+1].get_ydata()[0]
        yround = np.round(y,3)
        ax.annotate(r'${}$'.format(yround), (i+1+0.2, y), fontsize='x-small',  ha='left', va='center')

    #ax.set_xticklabels(ax.get_xticks(), rotation = 45)
    
    if(not save_png==None):
        fig.savefig(save_png,bbox_inches='tight',dpi=300)  

######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################

def plot_filaments(self, fil_idx):

        #Convenient variables
        g = self.grid
        s = self.structures

        #Making the plot
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        for fil in s.fils[fil_idx]:
            ax.scatter(fil.samps[:,0]*g.grid_length, fil.samps[:,1]*g.grid_length, fil.samps[:,2]*g.grid_length, s=7)
            n = fil.nsamp
            u_sampling = np.linspace(0,1,n*100)
            points = fil.get_filfunc(u_sampling)
            ax.plot(points[:,0]*g.grid_length, points[:,1]*g.grid_length, points[:,2]*g.grid_length)

        #ax.set_xlim(1,6)
        ax.set_ylim(6.7,7.8)
        #ax.set_zlim(10,12)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])