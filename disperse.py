######################################################################
#                            LIBRARIES                               #
######################################################################

#Standard libs
import numpy as np

#Own libs
from .units import *
from .physical_functions import *

######################################################################
#                         CRITICAL POINT                             #
######################################################################

class CriticalPoint:
    def __init__(self):
        self.idx = int
        self.pos = []
        self.nfil = int
        
    def __repr__(self):
        return "Critical point with index {} and position {} and nfil {}".format(self.idx,self.pos,self.nfil)
    
    def __str__(self):
        return "Critical point with index {} and position {} and nfil {}".format(self.idx,self.pos,self.nfil)
    
######################################################################
#                              FILAMENT                              #
######################################################################

class Filament:
    def __init__(self):
        
        #Core variables
        self.idx = int
        self.cps = []
        self.nsamp = int
        self.samps = [] #NOTE: The samps contain the critical points already (as first and last element)
        
        self.filfunc = None
        self.cylinders_AREPO_ids = []
        
        #Calculated properties
        self.length = float #parsec
        self.mass = float #solar masses
        
    def __repr__(self):
        return "Filament with index {} and nsamp {}".format(self.idx,self.nsamp)
    
    def __str__(self):
        return "Filament with index {} and nsamp {}".format(self.idx,self.nsamp)
    
    ######################################################################
    #                               FILFUNC                              #
    ######################################################################
    
    #The filfunc is a function that parametrizes the 1D curve in 3D space.
    #The parameter 'u' takes value from 0 to 1 and fully parametrizes the curve (distance along curve).
    #So, for u=0, its the first critical point and for u=1, its the last critical point.
    
    def set_filfunc(self):
            
        from scipy.interpolate import splprep
        #Since you can always fit a N-1 degree polynomial to N points
        if(self.nsamp>3):
            degree = 3
        else:
            degree = self.nsamp-1
        #Note the smoothing s=0 so that the fit curve passes through all sampling points
        tck, _ = splprep([self.samps[:,0],self.samps[:,1],self.samps[:,2]],s=0,k=degree)
        #Setting the filfunc!
        self.filfunc = tck
    
    def get_filfunc(self,u):
        
        from scipy.interpolate import splev
        points = np.array(splev(u,self.filfunc)).T
        return points
    
    ######################################################################
    #            CREATING CYLINDERS BETWEEN SAMPLING POINTS              #
    ######################################################################
            
    def make_cylinders(self, AVG, plot=False):
        # number of sample points
        #print('Number of sample points: ', len(self.samps))

        def normalize(v):
            v_unit = v/np.sqrt(v[0]**2+v[1]**2+v[2]**2)
            return v_unit

        def rotate_axes(normal):
            n = normalize(normal)
            theta = np.arccos(n[2])
            u = normalize(np.cross(n,[0,0,1]))
            rot = np.array([[np.cos(theta)+u[0]**2*(1-np.cos(theta)), u[0]*u[1]*(1-np.cos(theta))-u[2]*np.sin(theta), u[0]*u[2]*(1-np.cos(theta))+u[1]*np.sin(theta)],
                           [u[1]*u[0]*(1-np.cos(theta))+u[2]*np.sin(theta), np.cos(theta)+u[1]**2*(1-np.cos(theta)), u[1]*u[2]*(1-np.cos(theta))-u[0]*np.sin(theta)],
                           [u[2]*u[0]*(1-np.cos(theta))-u[1]*np.sin(theta), u[2]*u[1]*(1-np.cos(theta))+u[0]*np.sin(theta), np.cos(theta)+u[2]**2*(1-np.cos(theta))]])
            return rot
        
        # create storage arrays
        normals = np.zeros((len(self.samps),3))
        cylinders = []
        id_nums = []
        
        # get normal vectors for selfaments points
        normals[0] = self.samps[0]-self.samps[1]
        for i in range(1,len(self.samps)-1):
            normals[i] = self.samps[i-1]-self.samps[i+1]
        normals[len(self.samps)-1] = self.samps[len(self.samps)-2]-self.samps[len(self.samps)-1]
        
        # for each point in a selfament
        for i in range(len(self.samps)):
            cylinder = []
            id_num = []
            # compute rotation matrix
            rot = rotate_axes(normals[i])        
            # define bounds of region of interest [[x_min, y_min, z_min], [x_max, y_max, z_max]] (self.samps are in pc)
            bounds = [[self.samps[i][0]-0.2, self.samps[i][1]-0.2, self.samps[i][2]-0.2], [self.samps[i][0]+0.2, self.samps[i][1]+0.2, self.samps[i][2]+0.2]]
            # convert to AREPO code units for search
            bounds = np.array(bounds)/uparsec
            # find all points in the region
            pts_nan = np.where((AVG.position[0:AVG.ngas].transpose()[0] > bounds[0][0]) & (AVG.position[0:AVG.ngas].transpose()[1] > bounds[0][1]) & (AVG.position[0:AVG.ngas].transpose()[2] > bounds[0][2]) & (AVG.position[0:AVG.ngas].transpose()[0] < bounds[1][0]) & (AVG.position[0:AVG.ngas].transpose()[1] < bounds[1][1]) & (AVG.position[0:AVG.ngas].transpose()[2] < bounds[1][2]), AVG.position[0:AVG.ngas].transpose(), np.nan)
            ids_nan = np.where((AVG.position[0:AVG.ngas].transpose()[0] > bounds[0][0]) & (AVG.position[0:AVG.ngas].transpose()[1] > bounds[0][1]) & (AVG.position[0:AVG.ngas].transpose()[2] > bounds[0][2]) & (AVG.position[0:AVG.ngas].transpose()[0] < bounds[1][0]) & (AVG.position[0:AVG.ngas].transpose()[1] < bounds[1][1]) & (AVG.position[0:AVG.ngas].transpose()[2] < bounds[1][2]), np.array([[AVG.gas_ids]]*3), np.nan)
            pts = np.array([row[~np.isnan(row)] for row in pts_nan])
            ids = np.array([row[~np.isnan(row)] for row in ids_nan])
            pts = pts.transpose()
            ids = ids.transpose()
            # work in parsecs
            pts = pts*uparsec

            self_rotated = np.zeros((len(self.samps),3))
            # translate such that sample point i is at origin
            self_at_origin = self.samps-self.samps[i]
            for j in range(len(self.samps)):
                # rotate everything
                self_rotated[j] = np.dot(rot, self_at_origin[j])
            # translate back
            self_translated_back = self_rotated + self.samps[i]
            
            pts_rotated = np.zeros((len(pts),3))
            # translate such that sample point i is at origin
            pts_at_origin = pts-self.samps[i]
            #print(self_translated_back[1][2], self_translated_back[0][2])
            for k in range(len(pts)):
                # rotate everything
                pts_rotated[k] = np.dot(rot, pts_at_origin[k])
                if i < len(self.samps)-1:
                    if self_rotated[i+1][2] < self_rotated[i][2]: # does selfament have negative gradient?
                        # define cylinder
                        if ((pts_rotated[k][0]**2 + pts_rotated[k][1]**2) <= (0.1)**2) & (pts_rotated[k][2] >= self_rotated[i+1][2]) & (pts_rotated[k][2] <= self_rotated[i][2]):
                            cylinder.append(pts_rotated[k])
                            id_num.append(ids[k])
                    else:
                        # define cylinder
                        if ((pts_rotated[k][0]**2 + pts_rotated[k][1]**2) <= (0.1)**2) & (pts_rotated[k][2] <= self_rotated[i+1][2]) & (pts_rotated[k][2] >= self_rotated[i][2]):
                            cylinder.append(pts_rotated[k])
                            id_num.append(ids[k])
                elif i == len(self.samps)-1:
                    if self_rotated[i][2] < self_rotated[i-1][2]:
                        # define cylinder
                        if ((pts_rotated[k][0]**2 + pts_rotated[k][1]**2) <= (0.1)**2) & (pts_rotated[k][2] >= self_rotated[i][2]) & (pts_rotated[k][2] <= self_rotated[i-1][2]):
                            cylinder.append(pts_rotated[k])
                            id_num.append(ids[k])
                    else:
                        # define cylinder
                        if ((pts_rotated[k][0]**2 + pts_rotated[k][1]**2) <= (0.1)**2) & (pts_rotated[k][2] <= self_rotated[i][2]) & (pts_rotated[k][2] >= self_rotated[i-1][2]):
                            cylinder.append(pts_rotated[k])
                            id_num.append(ids[k])
                else:
                    print('i does not have an acceptable value')
                    
            cylinder = np.array(cylinder)
            id_num = np.array(id_num)
                        
            if cylinder.size == 0:
                #print('No points found for sample point '+ str(i) + ' in ' + str(self))
                pass
            else:
                id_num = id_num.transpose()[0]
            
                # find inverse rotation matrix (inverse of a rotation matrix is its transpose)
                invRot = rot.transpose()
                # invert rotation on points to confirm alignement with selfament
                for k in range(len(cylinder)):
                    cylinder[k] = np.dot(invRot, cylinder[k])

                # translate back
                cylinder = np.array(cylinder + self.samps[i])
                
            cylinders.append(cylinder)
            id_nums.append(id_num)
            
        
    # visual confirmation
        if plot == True:
            mpl.rc('text', usetex = True)
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.set_title(r"$\mathrm{Particles~adjacent~to~a~filament~spine}$", fontsize='xx-large')
            ax.set_xlabel(r"$\mathrm{x~[pc]}$",fontsize='xx-large')
            ax.set_ylabel(r"$\mathrm{y~[pc]}$",fontsize='xx-large')
            ax.set_zlabel(r"$\mathrm{z~[pc]}$",fontsize='xx-large')
            ax.plot3D(fil.samps.transpose()[0],fil.samps.transpose()[1],fil.samps.transpose()[2],color='k')
            #for c in range(len(cylinders)):
            #    if cylinders[c].size == 0:
            #        pass
            #    else:
            #        ax.scatter(cylinders[c].transpose()[0],cylinders[c].transpose()[1],cylinders[c].transpose()[2],color='midnightblue', alpha=0.05)
            #ax.plot3D([fil.samps.transpose()[0][0],fil.samps.transpose()[0][1]], [fil.samps.transpose()[1][0],fil.samps.transpose()[1][1]], [fil.samps.transpose()[2][0],fil.samps.transpose()[2][1]], color='darkseagreen')
            #ax.plot3D([fil.samps.transpose()[0][1],fil.samps.transpose()[0][2]], [fil.samps.transpose()[1][1],fil.samps.transpose()[1][2]], [fil.samps.transpose()[2][1],fil.samps.transpose()[2][2]], color='mediumseagreen')
            #ax.plot3D([fil.samps.transpose()[0][2],fil.samps.transpose()[0][3]], [fil.samps.transpose()[1][2],fil.samps.transpose()[1][3]], [fil.samps.transpose()[2][2],fil.samps.transpose()[2][3]], color='seagreen')
            #ax.plot3D([fil.samps.transpose()[0][3],fil.samps.transpose()[0][4]], [fil.samps.transpose()[1][3],fil.samps.transpose()[1][4]], [fil.samps.transpose()[2][3],fil.samps.transpose()[2][4]], color='midnightblue')
            #ax.plot3D([fil.samps.transpose()[0][4],fil.samps.transpose()[0][5]], [fil.samps.transpose()[1][4],fil.samps.transpose()[1][5]], [fil.samps.transpose()[2][4],fil.samps.transpose()[2][5]], color='slateblue')
            #ax.plot3D([fil.samps.transpose()[0][5],fil.samps.transpose()[0][6]], [fil.samps.transpose()[1][5],fil.samps.transpose()[1][6]], [fil.samps.transpose()[2][5],fil.samps.transpose()[2][6]], color='lightsteelblue')
            for i in range(len(cylinders)):
                ax.scatter(cylinders[i].transpose()[0], cylinders[i].transpose()[1], cylinders[i].transpose()[2], alpha=0.05)
            
        for c in range(len(cylinders)):
            cylinders[c] = cylinders[c]/uparsec

        self.cylinders_AREPO_ids = id_nums
        
    def calc_length(self):
        
        from scipy.spatial.distance import euclidean
        
        #First creating sampling points along the filament
        n = self.nsamp
        u_sampling = np.linspace(0,1,n*500)
        points = self.get_filfunc(u_sampling)
    
        #CALCULATING LENGTH
        length = 0
        for i in range(1,len(points)):
            length += euclidean(points[i-1],points[i])
        self.length = length #parsec
        
    def calc_mass(self, AVG):
        
        if(self.cylinders_AREPO_ids == []):
            self.make_cylinders(AVG)
        
        unique_ids = np.unique([item for sublist in self.cylinders_AREPO_ids for item in sublist])
        self.mass = np.sum(np.array(AVG.mass)[unique_ids.astype(int)]) #solar masses

######################################################################
#                           BIFURCATIONS                             #
######################################################################
#This is an extra functionality
class BifurcationPoint:
    def __init__(self, pos=[]):
        self.pos = pos
        
######################################################################
#                             STRUCTURES                             #
######################################################################
        
class Network:
    
    ######################################################################
    #                           CONSTRUCTORS                             #
    ######################################################################
    
    def __init__(self, file_path=None):
        self.file_path = str
        self.ntotcps = int
        self.ntotfils = int
        self.cps = []
        self.fils = []
        self.scale = str #either "pixel" or "parsec"

        #These are extra properties calculated if required
        self.bifurcations = []
        self.nbifurcation = int
        
        if(not file_path==None):
            self.read_ASCII_file(file_path)
        
    ######################################################################
    #                       READ AND WRITE FUNCTIONS                     #
    ######################################################################
        
    def read_ASCII_file(self, file_path):
        
        #see http://www2.iap.fr/users/sousbie/web/html/indexbea5.html?post/NDskl_ascii-format
    
        print("NOTE: Started reading ASCII file for critical points and filaments data...")
        
        self.file_path = file_path

        #FIRST STORING THE DIFFERENT SECTIONS OF THE ASCII OUTPUT
        with open(file_path) as f:
            text = f.read()
        stop1 = text.rfind("[CRITICAL POINTS]")
        stop2 = text.rfind("[FILAMENTS]")
        stop3 = text.rfind("[CRITICAL POINTS DATA]")
        stop4 = text.rfind("[FILAMENTS DATA]")
        critical_points_text = text[stop1:stop2]
        filaments_text = text[stop2:stop3]
        critical_data_text = text[stop3:stop4]
        filaments_data_text = text[stop4:]

        #READING CRITICAL POINT DATA
        critical_points = []
        critical_points_text_array = critical_points_text.splitlines()[2:]
        for i in range(len(critical_points_text_array)):
            line = critical_points_text_array[i]
            if(not line[0]==' '):
                l = line.split(' ')
                cp = CriticalPoint()
                cp.idx = len(critical_points) #Setting 'idx' here
                cp.pos = np.array([float(l[3]),float(l[2]),float(l[1])]) #Setting 'pos' here
                #IMPORTANT NOTE: Notice the order of l[1], l[2] and l[3].
                #This is the discrepancy in Disperse coordinates and AREPO.
                cp.nfil = int(critical_points_text_array[i+1]) #Setting 'nfil' here
                critical_points.append(cp)
        critical_points = np.array(critical_points)

        #READING FILAMENT DATA
        filaments = []
        filaments_text_array = filaments_text.splitlines()[2:]
        for i in range(len(filaments_text_array)):
            line = filaments_text_array[i]
            if(not line[0]==' '):
                l = line.split(' ')
                fil = Filament()
                fil.idx = len(filaments) #Setting 'idx' here
                fil.nsamp = int(l[2]) #Setting 'nsamp' here
                fil.cps = np.array([critical_points[int(l[0])],critical_points[int(l[1])]]) #Setting 'cps' here
                while(True):
                    if(i==len(filaments_text_array)-1):
                        break
                    else:
                        i+=1
                        line = filaments_text_array[i]
                    if(not line[0]==' '):
                        i-=1
                        fil.samps = np.array(fil.samps)
                        fil.set_filfunc() #Setting 'filfunc' here
                        filaments.append(fil)
                        break
                    else:
                        l = line.split(' ')
                        pos = np.array([float(l[3]),float(l[2]),float(l[1])])
                        #IMPORTANT NOTE: Notice the order of l[1], l[2] and l[3].
                        #This is the discrepancy in Disperse coordinates and AREPO.
                        fil.samps.append(pos)
        filaments = np.array(filaments)
                    
        print("NOTE: Completed reading ASCII file for critical points and filaments data.")

        self.cps = critical_points
        self.ntotcps = len(critical_points)
        self.fils = filaments
        self.ntotfils = len(filaments)
        self.scale = "pixel"
        
    ######################################################################
    #                      RESCALE TO ORIGINAL AREPO                     #
    ######################################################################
    
    def set_scale(self, arepocubicgrid):
        
        acg = arepocubicgrid
        self.scale = acg.scale
        
        #Scale critical points
        for cp in self.cps:
            cp.pos[0] = acg.xmin + cp.pos[0] * float(acg.xmax - acg.xmin)/float(acg.nx)
            cp.pos[1] = acg.ymin + cp.pos[1] * float(acg.ymax - acg.ymin)/float(acg.ny)
            cp.pos[2] = acg.zmin + cp.pos[2] * float(acg.zmax - acg.zmin)/float(acg.nz)
            
        #Scale filaments
        for fil in self.fils:
            for cp in fil.cps:
                cp.pos[0] = acg.xmin + cp.pos[0] * float(acg.xmax - acg.xmin)/float(acg.nx)
                cp.pos[1] = acg.ymin + cp.pos[1] * float(acg.ymax - acg.ymin)/float(acg.ny)
                cp.pos[2] = acg.zmin + cp.pos[2] * float(acg.zmax - acg.zmin)/float(acg.nz)
            for samp in fil.samps:
                samp[0] = acg.xmin + samp[0] * float(acg.xmax - acg.xmin)/float(acg.nx)
                samp[1] = acg.ymin + samp[1] * float(acg.ymax - acg.ymin)/float(acg.ny)
                samp[2] = acg.zmin + samp[2] * float(acg.zmax - acg.zmin)/float(acg.nz)
        
            #Resetting the fil function
            fil.set_filfunc()
            
        #Scale bifurcations (if any)
        for bf in self.bifurcations:
            bf.pos[0] = acg.xmin + bf.pos[0] * float(acg.xmax - acg.xmin)/float(acg.nx)
            bf.pos[1] = acg.ymin + bf.pos[1] * float(acg.ymax - acg.ymin)/float(acg.ny)
            bf.pos[2] = acg.zmin + bf.pos[2] * float(acg.zmax - acg.zmin)/float(acg.nz)
            
    ######################################################################
    #                    PROPERTIES OF THE NETWORK                       #
    ######################################################################
    
    def remove_short_filaments(self):
        
        #TODO: Also remove critical points corresponding to these filaments
        
        #Removing all filaments with only 2 sampling points
        long_fil_idxs = [i for i in range(self.ntotfils) if self.fils[i].nsamp!=2]
        
        self.fils = self.fils[long_fil_idxs]
        self.ntotfils = len(self.fils)    
       
    ######################################################################
    #                    PROPERTIES OF THE NETWORK                       #
    ######################################################################
    
    def calc_filament_lengths(self):
        print("NOTE: Started calculating length of filaments...")
        for fil in self.fils:
            fil.calc_length()
        print("NOTE: Completed calculating length of filaments.")
        
    def calc_filament_masses(self, AVG):
        print("NOTE: Started calculating mass of filaments...")
        for fil in self.fils:
            print("Filaments done:"+str(fil.idx)+"/"+str(len(self.fils)))
            fil.calc_mass(AVG)
        print("NOTE: Completed calculating mass of filaments...")

    ######################################################################
    #                   BIFURCATIONS OF THE NETWORK                      #
    ######################################################################

    # Finds bifurcation points for a set of filaments, with a variable separation threshold in pc - 
    # this parameter has an impact on
    # the evaluated number of bifurcations.
    # The chosen value of 0.0001 pc gives the correct number of intersections without excluding any bifurcation regions for the
    # test case. An improvement would be to have a range of thresholds for which the bifurcation number is evaluated and find the 
    # separation threshold value for which the bifurcation number remains stable for the most number of threshold values.
    # This would lend some robustness to the chosen separation threshold value, without having to 
    # require a human to manually count
    # the number of bifurcations for every new case.

    def find_bifurcation_points(self, separation_threshold=0.0001):
        
        #import required functions
        from scipy.spatial.distance import euclidean
        from scipy.spatial import cKDTree
        
        # initialise arrays
        points = []
        ids = []
        candidates = []
        bifurcations = []
        id1 = []

        # iterate through points in each filament and assign an ID corresponding to the filament they belong to
        for j in range(self.ntotfils):
            for i in self.fils[j].samps:
                points.append(i)
                ids.append(j)
        points = np.array(points)
        ids = np.array(ids)
        # make a KDtree from all the points
        tree = cKDTree(points)

        # get all the points in the KDTree which are within a distance of 0.0001 pc (by default) of a point for each sample point
        for k in points:
            candidate = tree.data[tree.query_ball_point(k, separation_threshold/uparsec)]
            candidates.append(candidate)

        # for each set of candidate points corresponding to a ball around a sample point
        for cs in range(len(candidates)):
            # reset arrays
            id1 = []
            # for each of the individual candidates
            for c in candidates[cs]:
                # for each sample point
                for k in range(len(points)):
                    # compare candidate point to sample point and assign IDs to matching points
                    if c[0] == points[k][0] and c[1] == points[k][1] and c[2] == points[k][2]:
                        id1.append(ids[k])
            # compare all ids in the list of candidates to the first
            first = id1[0]
            if all(i == first for i in id1):
                pass
            # if there are different ids in the set of candidate points around a sample point, it is a bifurcation point
            else:
                bifurcations.append(BifurcationPoint(points[cs]))
        bifurcations = np.array(bifurcations)

        # the number of bifurcation points is half of the points classified as bifurcation points (pairs) for the correct separation threshold
        bifurcation_number = len(bifurcations)/2
        
        self.bifurcations = bifurcations
        self.nbifurcation = bifurcation_number