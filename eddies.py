# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 15:03:30 2019

@author: jasaa
v0 of eddy detection functions

eddy_detection:
    inputs: 
        - filename: name of the netCDF file with the data
        - R2_criterion: Confidence level, usually 90%
        - OW_start: OW value at which to begin the evaluation of R2
        - max_evaluation_points: Number of local minima to evaluate using R2 method.
        Set low (like 20) to see a few R2 eddies quickly.
        Set high (like 1e5) to find all eddies in domain.
        - min_eddie_cells: Minimum number of cells required to be identified as an eddie.
        
    returns a tuple with:
        - lon: longitude vector (ª)
        - lat: latitude vector (ª)
        - uvel: zonal velocity (m/s)
        - vvel: meridional velocity (m/s)
        - vorticity (m/s)
        - nEddies: number of eddies found
        - eddy_census: characteristics of the detected eddies --> minOW, circ(m^2/s), lon(º), lat(º), cells, diameter(km)
        - OW: non-dimensional Okubo-Weiss parameter
        - OW_eddies: OW<-0.2 --> cells that could containt the center of an eddy
        - cyclonic_mask: mask of cyclonic (+1) and anti-cyclonic (-1) eddies
"""

# Eddy detection algorithm

def eddy_detection(filename,R2_criterion,OW_start,max_evaluation_points,min_eddie_cells):
    import math
    import numpy as np
    import netCDF4 as nc4

    
    f = nc4.Dataset(filename,'r', format='NETCDF4') #'r' stands for read
    
    # Load longitude and latitude, and depth of grid
    lon = f.variables['longitude'][:]
    lat = f.variables['latitude'][:]
    depth = f.variables['depth'][:]
    # Load zonal and meridional velocity, in m/s
    uvel = f.variables['uo'][:]
    vvel = f.variables['vo'][:]
    
    f.close() 
    
    ########################################################################
    
    # Initialize variables
        
    ########################################################################
    
    # We transpose the data to fit with the algorithm provided
    uvel = uvel[0,:,:,:].transpose(2,1,0)
    vvel = vvel[0,:,:,:].transpose(2,1,0)
    
    # Since they are masked arrays (in the mask, True = NaN value), we can fill the masked values with 0.0 to describe land
    uvel.set_fill_value(0.0)
    vvel.set_fill_value(0.0)

    # Create an ocean mask which has value True at ocean cells.
    ocean_mask = ~uvel.mask
    n_ocean_cells = uvel.count()
    
    nx,ny,nz = uvel.shape
    
    # Compute cartesian distances for derivatives, in m
    R = 6378e3
    
    x = np.zeros((nx,ny))
    y = np.zeros((nx,ny))
                 
    for i in range(0,nx):
        for j in range(0,ny):
            x[i,j] = 2.*math.pi*R*math.cos(lat[j]*math.pi/180.)*lon[i]/360.
            y[i,j] = 2.*math.pi*R*lat[j]/360.
            
    # Gridcell area
    dx,dy,grid_area = grid_cell_area(x,y)
    
    # Calculate the thickness of each depth level, we do a mean between the level above and below => dz[i] = (depth[i+1] - depth[i-1]) / 2.0;
    # except for the first depth wich is 2*depth[0]
    
    # if the data has only one depth, we choose dz=1, a value chosen arbitrarily 
    # to well work with the volume calculations (in this case we would work formally with areas)
    if nz==1:
        dz = np.array([1])
    else:
        dz = np.zeros(nz)
        # Thickness of each layer
        dz[0] = 2.0*depth[0]
        for i in range(1,nz-1):
            dz[i] = (depth[i+1] - depth[i-1]) / 2.0
        dz[nz-1] = depth[nz-1] - depth[nz-2]
    
    ########################################################################

    #  Compute Okubo-Weiss
    
    ########################################################################
    
    uvel = uvel.filled(0.0)
    vvel = vvel.filled(0.0)
    
    # velocity derivatives
    du_dx,du_dy = deriv1_central_diff_3D(uvel,x,y)
    dv_dx,dv_dy = deriv1_central_diff_3D(vvel,x,y)
    # strain and vorticity
    normal_strain = du_dx - dv_dy
    shear_strain = du_dy + dv_dx
    vorticity = dv_dx - du_dy
    
    # Compute OW, straight and then normalized with its standart deviation
    OW_raw = normal_strain ** 2 + shear_strain ** 2 - vorticity ** 2
    OW_mean = OW_raw.sum() / n_ocean_cells
    OW_std = np.sqrt(np.sum((np.multiply(ocean_mask,(OW_raw - OW_mean)) ** 2)) / n_ocean_cells)
    OW = OW_raw / OW_std
    
    OW_eddies = np.zeros(OW.shape,dtype=int)
    OW_eddies[np.where(OW < - 0.2)] = 1
        
    
    ########################################################################
    
    #  Find local minimums in Okubo-Weiss field
        
    ########################################################################
    # Efficiency note: Search for local minima can be merged with R2
    # algorithm below.
        
    print('\nNote: max_evaluation_points set to '+ repr(max_evaluation_points) ,'\nTo identify eddies over the full domain, set max_evaluation_points to a high number like 1e4.')
    local_mins = local_minima3D (OW,OW_start,max_evaluation_points)
    num_mins = local_mins.shape[1]
    
    # Compute OW, straight and then normalized with its standart deviation
    OW_raw = normal_strain ** 2 + shear_strain ** 2 - vorticity ** 2
    OW_mean = OW_raw.sum() / n_ocean_cells
    OW_std = np.sqrt(np.sum((np.multiply(ocean_mask,(OW_raw - OW_mean)) ** 2)) / n_ocean_cells)
    OW = OW_raw / OW_std
    
    OW_eddies = np.zeros(OW.shape,dtype=int)
    OW_eddies[np.where(OW < - 0.2)] = 1
    
    
    ########################################################################
    
    #  R2 algorithm
    
    ########################################################################
    print('Beginning R2 algorithm\n')
    # Set a maximum number of cells to search through, for initializing
    # arrays.
    max_eddy_cells_search = 10000
    
    # Initialize variables for eddy census
    iEddie = 0
    eddie_census = np.zeros((6, num_mins))
    all_eddies_mask = np.zeros(uvel.shape,dtype=int)
    cyclonic_mask = np.zeros(uvel.shape,dtype=int)
    
    print('Evaluating eddy at local OW minimuma.  Number of minimums = %g \n' %num_mins)
    
    # loop over local OW minima
    for imin in range(0, num_mins):
        # initialize variables for this local minimum in OW
        ie = local_mins[0, imin]
        je = local_mins[1, imin]
        ke = local_mins[2, imin]
        
        # Efficiency note: Eddie and neigbor masks are logical arrays the
        # size of the full 3D domain.  A more efficient implementation is
        # to create a list that records the indices of all eddy and
        # neighbor cells.
        eddie_mask = np.zeros(uvel.shape,dtype=int)
        neighbor_mask = np.zeros(uvel.shape,dtype=int)
    
        eddie_mask[ie, je, ke] = 1
        minOW = np.zeros((max_eddy_cells_search, 1))
        volume = np.zeros((max_eddy_cells_search, 1))
        R2 = np.zeros((max_eddy_cells_search, 1))
        
        minOW[0] = OW[ie, je, ke]
        volume[0] = grid_area[ie, je]* dz[ke]
        start_checking = 0
        max_k = 0
        min_k = nz
        print('imin=' + repr(imin) ,'lon='+repr(lon[ie]) ,'lon='+repr(lat[je]) ,'lon='+repr(lon[ie]) ,'k='+repr(ke),end=' ')
        
        # Loop to accumulate cells neighboring local min, in order of min OW.
        for ind in range(1, max_eddy_cells_search):
            # Identify six neighbors to the newest cell.
            # Subtract eddy mask so cells already in eddy are not candidates.
            neighbor_mask[np.max((ie - 1, 0)), je, ke] = 1 - eddie_mask[np.max((ie - 1, 0)), je, ke]
            neighbor_mask[np.min((ie + 1, nx-1)), je, ke] = 1 - eddie_mask[np.min((ie + 1, nx-1)), je, ke]
            neighbor_mask[ie, np.max((je - 1, 0)), ke] = 1 - eddie_mask[ie, np.max((je - 1, 0)), ke]
            neighbor_mask[ie, np.min((je + 1, ny-1)), ke] = 1 - eddie_mask[ie, np.min((je + 1, ny-1)), ke]
            neighbor_mask[ie, je, np.max((ke - 1, 0))] = 1 - eddie_mask[ie, je, np.max((ke - 1, 0))]
            neighbor_mask[ie, je, np.min((ke + 1, nz-1))] = 1 - eddie_mask[ie, je, np.min((ke + 1, nz-1))]
            # neighboring the current eddy cells.
            neighbor_indices = np.where(neighbor_mask)
            minOW[ind] = np.min(OW[neighbor_indices])
            minInd = np.where(OW[neighbor_indices] == minOW[ind])[0][0]
            ie,je,ke =np.asarray(neighbor_indices)[:,minInd]
    
            # (ie,je,ke) is the newest cell added to the eddy.  Reset masks
            # at that location.
            eddie_mask[ie, je, ke] = 1
            neighbor_mask[ie, je, ke] = 0
            min_k = np.min((min_k, ke+1))
            max_k = np.max((max_k, ke+1))
                
            # We are building a data set of minimum OW versus volume
            # accumulated so far in this search.  If the new eddy cell has
            # lower OW, record the previous value of OW.  This is so OW
            # values are always increasing.
            minOW[ind] = np.max((minOW[ind], minOW[ind - 1]))
            volume[ind] = volume[ind - 1] + grid_area[ie, je]*dz[ke]
                
            # Reject eddies identified over duplicate cells. Don't check every time for efficiency.
            # Note: This illustrative algorithm uses the first accepted
            # eddy, and all later eddies in identical cells are duplicates.
            # A better method is to find the bounds of all accepted eddies,
            # and then choose among duplicates with another criteria, for
            # example largest volume.
            if np.mod(ind, 20) == 0:
                if np.max(eddie_mask + all_eddies_mask) == 2:
                    print('No, duplicate\n')
                    break
            if start_checking == 0:
            # When OW value greater than OW_start, check if R2 criterion
            # is met.
                if minOW[ind] > OW_start:
                # Compute R2 value of linear fit of volume versus min OW.
                    temp = np.corrcoef(minOW[0:ind+1],volume[0:ind+1],rowvar=False)
                    R2[ind] = temp[0, 1]
                    if R2[ind] < R2_criterion:
                        print('No, R2 criterion not met\n')
                        break
                    else:
                        # After this iteration, check R2 every time.
                        start_checking = 1
            else:
                # Compute R2 value of linear fit of volume versus min OW.
                temp = np.corrcoef(minOW[0:ind+1],volume[0:ind+1],rowvar=False)
                R2[ind] = temp[0, 1]
                # When the R2 value falls below the critical level, we may have an eddie.
                if R2[ind] < R2_criterion:
    
                    # Reject eddies identified over duplicate cells.
                    if np.max(eddie_mask + all_eddies_mask) == 2:
                        print('No, duplicate eddie\n')
                        break
                    # Reject eddies that are too small.
                    if ind <= min_eddie_cells:
                        print('No, too small.  Number of cells ='+repr(ind),'\n')
                        break
    
                    iEddie += 1
                    print('Yes, eddie confirmed.  iEddie='+repr(iEddie),'\n')
    
                    # find minimum OW value and location with this eddie
                    eddie_indices = np.where(eddie_mask)
                    minOW_eddie = np.min(OW[eddie_indices])
                    tempInd = np.where(OW[eddie_indices] == minOW_eddie)[0][0]
                    iE, jE, kE = np.asarray(eddie_indices)[:,tempInd]
                    
                    # Find diameter of this eddie, using area at depth of max OW
                    # value, in cm^2.  Diameter is in km.
                    area = np.sum(grid_area[np.where(eddie_mask[:,:, kE])])
                    diameter = 2*np.sqrt(area/np.pi)/1e3
                    
                    # Circulation aroung the eddie
                    # Calculated on a square line around the center of the eddy, positive in the clockwise direction
                    
                    circ_sides = -vvel[np.min((iE+1,nx-1)), jE, kE]*dy[np.min((iE+1,nx-1)),jE] - uvel[iE, np.max((jE-1,0)), kE]*dx[iE,np.max((0,jE-1))] + vvel[np.max((0,iE-1)), jE, kE]*dy[np.max((iE-1,0)),jE] + uvel[iE, np.min((jE+1,ny-1)), kE]*dx[iE,np.min((jE+1,ny-1))]   
                    circ_corner1 = -vvel[np.min((iE+1,nx-1)), np.max((jE-1,0)), kE]*0.5*dy[np.min((iE+1,nx-1)),np.max((jE-1,0))] - uvel[np.min((iE+1,nx-1)), np.max((jE-1,0)), kE]*0.5*dx[np.min((iE+1,nx-1)),np.max((jE-1,0))]
                    circ_corner2 = -uvel[np.max((0,iE-1)), np.max((jE-1,0)), kE]*0.5*dx[np.max((0,iE-1)),np.max((jE-1,0))] + vvel[np.max((0,iE-1)), np.max((jE-1,0)), kE]*0.5*dy[np.max((0,iE-1)),np.max((jE-1,0))]
                    circ_corner3 =  vvel[np.max((0,iE-1)), np.min((jE+1,ny-1)), kE]*0.5*dy[np.max((0,iE-1)),np.min((jE+1,ny-1))] + uvel[np.max((0,iE-1)), np.min((jE+1,ny-1)), kE]*0.5*dx[np.max((0,iE-1)),np.min((jE+1,ny-1))]
                    circ_corner4 =  uvel[np.min((iE+1,nx-1)), np.min((jE+1,ny-1)), kE]*0.5*dx[np.min((iE+1,nx-1)),np.min((jE+1,ny-1))] - vvel[np.min((iE+1,nx-1)), np.min((jE+1,ny-1)), kE]*0.5*dy[np.min((iE+1,nx-1)),np.min((jE+1,ny-1))]
                
                    circ = circ_sides + circ_corner1 + circ_corner2 + circ_corner3 + circ_corner4  
                    
                    
                    # add this eddie to the full eddie mask
                    all_eddies_mask = all_eddies_mask + eddie_mask 
                    
                    if circ>0.0:
                        cyclonic_mask = cyclonic_mask + eddie_mask
                    else:
                        cyclonic_mask = cyclonic_mask - eddie_mask
      
                    
                    # record eddie data
                    eddie_census[:, iEddie-1] = (minOW[0], circ, lon[iE], lat[jE], ind, diameter)
                
                    break
    
    nEddies = iEddie
    
    return (lon,lat,uvel,vvel,vorticity,OW,OW_eddies,eddie_census,nEddies,cyclonic_mask)
    

## Creates grid #####################################################
def grid_cell_area(x,y):
    import numpy as np
# Given 2D arrays x and y with grid cell locations, compute the
# area of each cell.
    
    nx,ny = x.shape
    dx = np.zeros((nx,ny))
    dy = np.zeros((nx,ny))
    
    for j in range(0,ny):
        dx[0,j] = x[1,j] - x[0,j]
        for i in range(1,nx-1):
            dx[i,j] = (x[i+1,j] - x[i-1,j]) / 2.0
        dx[nx-1,j] = x[nx-1,j] - x[nx-2,j]
    
    for i in range(0,nx):
        dy[i,0] = y[i,1] - y[i,0]
        for j in range(1,ny - 1):
            dy[i,j] = (y[i,j+1] - y[i,j-1]) / 2.0
        dy[i,ny-1] = y[i,ny-1] - y[i,ny-2]
    
    A = np.multiply(dx,dy)
    return (dx,dy,A)


## Calculate strains, vorticity and Okubo-Weiss ######################################

def deriv1_central_diff_3D(a,x,y):
# Take the first derivative of a with respect to x and y using
# centered central differences. The variable a is a 3D field.
    import numpy as np
    
    nx,ny,nz = a.shape
    dadx = np.zeros((nx,ny,nz))                            
    dady = np.zeros((nx,ny,nz))
    
    for k in range(0,nz):
        for j in range(0,ny):
            dadx[0,j,k] = (a[1,j,k] - a[0,j,k]) / (x[1,j] - x[0,j])
            for i in range(1,nx-1):
                dadx[i,j,k] = (a[i+1,j,k] - a[i-1,j,k]) / (x[i+1,j] - x[i-1,j])
            dadx[nx-1,j,k] = (a[nx-1,j,k] - a[nx-2,j,k]) / (x[nx-1,j] - x[nx-2,j])
        
        for i in range(0,nx):
            dady[i,0,k]=(a[i,1,k] - a[i,0,k]) / (y[i,1] - y[i,0])
            for j in range(1,ny-1):
                dady[i,j,k]=(a[i,j+1,k] - a[i,j-1,k]) / (y[i,j+1] - y[i,j-1])
            dady[i,ny-1,k]=(a[i,ny-1,k] - a[i,ny-2,k]) / (y[i,ny-1] - y[i,ny-2])
    
    return dadx,dady 


## Find local minima ###################################################################

def find_local_mins(A,A_start,max_evaluation_points):
# Find local minimums of the 3D array A that are less than
# A_start.  The output, local_mins, is a 3xm array of the m
# minimums found, containing the three A indices of each minimum.
# The search evaluates every k level, but not the horizontal edges.
    
    import numpy as np
    
    nx,ny, nz = A.shape
    local_mins = np.zeros((3,max_evaluation_points),dtype=int)
    
    imin = -1
    for k in range(0,nz):
        for j in range(1,ny-1):
            for i in range(1,nx-1):
               # if np.max((0,k-1)) == np.min((nz-1,k+1)):
                   # A_min_neighbors =  np.min(A[i-1:i+1, j-1:j+1,np.max((0,k-1))])
                #else:
                A_min_neighbors =  np.min(A[i-1:i+2, j-1:j+2, np.max((0,k-1)): 1+np.min((nz-1,k+1))])
                if (A[i,j,k] < A_start) and (A[i,j,k] == A_min_neighbors):
                    imin += 1
                    local_mins[:,imin] = (i,j,k)
                    if imin == max_evaluation_points-1:
                        return local_mins
    
    local_mins = local_mins[:,0:imin]
    return local_mins


def local_minima3D(A,A_start,max_evaluation_points):
# Alternative method of finding minima of the 3D array A that are less than
# A_start.  The output, local_mins, is a 3xm array of the positions of the minima found.
# minimums found, containing the three A indices of each minimum. 
# The compares each point with its neighbors and creates a boolean mask with the positions of the minima.
# minima_mask is an boolean array where True == minima in that point.
# Doesn´t treat values at the boundaries correctly; finds the neighbor at the other side
    
    import numpy as np
    
    mask_minima = ((A<A_start)      &
                   (np.abs(A)>0.0) &
            (A <= np.roll(A,  1, axis = 0)) &
            (A <= np.roll(A, -1, axis = 0)) &
            (A <= np.roll(A,  1, axis = 1)) &
            (A <= np.roll(A, -1, axis = 1)) &
            (A <= np.roll(A,  1, axis = 2)) &
            (A <= np.roll(A, -1, axis = 2)))
   
    n_minima = np.count_nonzero(mask_minima)
    local_min = np.asarray(np.where(mask_minima))
    sample = np.random.randint(0,local_min.shape[1],size = np.min((max_evaluation_points,n_minima)))

    return (mask_minima,local_min[:,sample])

## Print the eddy census ##################################################################
def print_eddies(eddie_census,nEddies):
    #prints the characteristics of the eddies from eddie_census
    import pandas as pd   
    import numpy as np
    
    print('\nEddie census data\n')
    
    name_list = ['minOW','circ(m^2/s)','lon(º)','lat(º)','cells','diameter(km)']
    data = eddie_census[:,0:nEddies].T    
    
    df = pd.DataFrame(data,index= np.arange(1,nEddies+1),columns=name_list)
    print(df)
    
## Plot velocities and eddies #############################################################
    
def plot_eddies(lon,lat,uvel,vvel,vorticity,OW,OW_eddies,eddie_census,nEddies,cyclonic_mask,k_plot):
    #k_plot: z-level to plot.  Usually set to 0 for the surface.

    import matplotlib.pyplot as plt
    
    fig,axes = plt.subplots(nrows=3, ncols=2,figsize=(10,10))

    pos1 = axes[0,0].imshow(uvel[:,:,k_plot].T, extent=[lon[0],lon[-1],lat[0],lat[-1]],aspect='auto',origin="lower",cmap='jet')
    axes[0,0].set_title('Zonal velocity (m/s) ->')
    
    pos2 =axes[0,1].imshow(vvel[:,:,k_plot].T, extent=[lon[0],lon[-1],lat[0],lat[-1]],aspect='auto',origin="lower",cmap='jet')
    axes[0,1].set_title('Meridional velocity (m/s) ^')
    
    pos3 = axes[1,0].imshow(1e5*vorticity[:,:,k_plot].T, extent=[lon[0],lon[-1],lat[0],lat[-1]],aspect='auto',origin="lower",cmap='jet')
    axes[1,0].set_title('1e5·Vorticity (1/s)')
    
    pos4 = axes[1,1].imshow(OW[:,:,k_plot].T, extent=[lon[0],lon[-1],lat[0],lat[-1]],aspect='auto',origin="lower",cmap='jet')
    axes[1,1].set_title('OW')
    
    pos5 = axes[2,0].imshow(OW_eddies[:,:,k_plot].T, extent=[lon[0],lon[-1],lat[0],lat[-1]], aspect='auto',origin="lower")
    axes[2,0].set_title('OW<-0.2')
    
    pos6 = axes[2,1].imshow(cyclonic_mask[:,:,k_plot].T, extent=[lon[0],lon[-1],lat[0],lat[-1]],aspect='auto',origin="lower")
    axes[2,1].set_title('Eddies (cyclonic=+1, anticyclonic=-1)')
    for i in range(0,nEddies):
        text = axes[2,1].annotate(i+1, eddie_census[2:4,i])
        text.set_color('r')
    
    # add the colorbar using the figure's method,telling it which mappable we're talking about and which axes object it should be near
    fig.colorbar(pos1, ax=axes[0,0])
    fig.colorbar(pos2, ax=axes[0,1])
    fig.colorbar(pos3, ax=axes[1,0])
    fig.colorbar(pos4, ax=axes[1,1])
    fig.colorbar(pos5, ax=axes[2,0])
    fig.colorbar(pos6, ax=axes[2,1])
        
    plt.tight_layout()
    plt.show()