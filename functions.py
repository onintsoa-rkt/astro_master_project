import numpy as np
import matplotlib.pyplot as plt
import unyt
from scipy.integrate import trapz

# calculate the distance between central galaxy and satellite galaxies
def get_distances(
    central_pos: np.ndarray,
    satellite_pos: np.ndarray,
    box_size: float,
    projected: bool = True,
    projection_axis: int = 2,
) -> np.ndarray:
    """
    Calculate the distance between central galaxy and satellite galaxies.
    Parameters
    ----------
    central_pos: np.ndarray
        The position of the central galaxy.
    satellite_pos: np.ndarray
        The position of the satellite galaxies.
    box_size : float
        The size of the simulation box.
    projected: bool
        If True, calculate the projected distance.
    projection_axis: int
        The axis along which to project.
        x: 0, y: 1, z: 2
    Returns
    -------
    distances: np.ndarray
        The distances between central galaxy and satellite galaxies
    """
    delta = satellite_pos - central_pos

    # set the periodic boundary conditions
    delta[delta > (box_size/2)] -= box_size
    delta[delta <= -(box_size/2)] += box_size

    if projected:
        # select the axes that are not the projection axis
        axes = [i for i in range(3) if i != projection_axis]
        # calculate the projected distance
        distances = np.sqrt(np.sum(delta[:, axes] ** 2, axis=1))
    else:
        distances = np.sqrt(np.sum(delta ** 2, axis=1))
    return distances



# calculate the central-satellite distances of all the satellite galaxies
def get_distances_all_satellites(
    the_centrals: list,
    the_satellites: list,
    isbox_size: unyt.array.unyt_array,
    unit_distance: str = 'Mpc',
    isprojected: bool = True,
    isprojection_axis: int = 2,
) -> list:
    """
    Calculate the central-satellite distances of all the satellite galaxies
    Parameters
    ----------
    the_centrals: list
        The list of the central galaxies
    the_satellites: list
        The list of the corresponding satellite galaxies
    box_size : unyt.array.unyt_array
        The size of the simulation box
    unit_distance: str
        The desired distance unit
    isprojected: bool
        If True, calculate the projected distance in the get_distances function
    isprojection_axis: int
        The axis along which to project in the get_distances function
        x: 0, y: 1, z: 2
    Returns
    -------
    all_distances: list
        List of the central-satellite distances of all the satellite galaxies
    """
    converted_box_size = float(isbox_size.to(unit_distance))

    all_distances = []
    for i in range(len(the_centrals)):
        central_in_halo = the_centrals[i]
        satellites_in_halo = the_satellites[i]

        # get the positions and convert to the given unit distance
        sat_gal_pos = np.array([i.pos.to(unit_distance)
            for i in satellites_in_halo])
        cen_gal_pos = np.array(central_in_halo.pos.to(unit_distance))

        # get the distances
        the_distances = get_distances(
            central_pos = cen_gal_pos,
            satellite_pos = sat_gal_pos,
            box_size = converted_box_size,
            projected = isprojected,
            projection_axis = isprojection_axis,
        )

        all_distances.append(the_distances)
    return all_distances



#calculate the pairwise distances of galaxies
def find_distances_galaxy_pairs(
    positions_gal1: np.ndarray,
    box_size: unyt.array.unyt_array,
    positions_gal2: np.ndarray = None,
    unit_distance: str = 'Mpc/h',
    projected: bool = True,
    projection_axis: int = 2,
    pbc: bool = True
) -> np.ndarray:
    """
    Calculate the pairwise distances of galaxies
    Parameters
    ----------
    positions_gal1: np.ndarray
        The positions of the galaxies
    box_size : unyt.array.unyt_array
        The size of the simulation box
    positions_gal2: np.ndarray
        The random positions
    unit_distance: str
        The desired distance unit
    projected: bool
        If True, calculate the projected distance
    projection_axis: int
        The axis along which to project
        x: 0, y: 1, z: 2
    pbc: bool
        If True, use the peridoc boundary conditions
    Returns
    -------
    dist_pairs: np.ndarray
        The pairwise distances of the galaxies
    """

    # select the projection axis
    if projected:
        if projection_axis == 0:
            positions_gal1 = positions_gal1[::,1:]
        elif projection_axis == 1:
            positions_gal1 = positions_gal1[::,::2]
        elif projection_axis == 2:
            positions_gal1 = positions_gal1[::,:2]
    
    if positions_gal2 is not None:
        if projected:
            if projection_axis == 0:
                positions_gal2 = positions_gal2[::,1:]
            elif projection_axis == 1:
                positions_gal2 = positions_gal2[::,::2] 
            elif projection_axis == 2:
                positions_gal2 = positions_gal2[::,:2]

    the_box_size = np.float64(box_size.to(unit_distance))

#    if pbc == True:
#        positions_gal1[positions_gal1 < -(the_box_size/2)] += the_box_size
#       positions_gal1[positions_gal1 >= (the_box_size/2)] -= the_box_size 
#
#        if positions_gal2 is not None:
#            positions_gal2[positions_gal2 < -(the_box_size/2)] += the_box_size
#            positions_gal2[positions_gal2 >= (the_box_size/2)] -= the_box_size 

    # galaxy positions pairwise difference
    if ((positions_gal2 is not None) and
        ((positions_gal1.shape != positions_gal2.shape) or
            np.all(positions_gal1 != positions_gal2))):
        all_pairs = positions_gal1[:, np.newaxis] - positions_gal2
        all_pairs = all_pairs.reshape(-1, all_pairs.shape[-1])

    else:
        i, j = np.triu_indices(len(positions_gal1), k=1)
        all_pairs = positions_gal1[j] - positions_gal1[i]
    
    if pbc == True:
        # set the periodic boundary conditions
        all_pairs[all_pairs > (the_box_size/2)] -= the_box_size
        all_pairs[all_pairs <= -(the_box_size/2)] += the_box_size

    dist_pairs = np.linalg.norm(all_pairs, axis=1)

    return dist_pairs




# Normalize the histogram
def get_normalized_histogram(
    data: np.ndarray,
    box_size: unyt.array.unyt_array,
    bin_number: int = 50,
    unit_distance: str = 'Mpc/h',
    pbc: bool = True
) -> tuple:
    """
    Create a normalized histogram
    Parameters
    ----------
    data: np.ndarray
        The data
    box_size : unyt.array.unyt_array
        The size of the simulation box
    bin_number:
        The desired number of bins
    unit_distance: str
        The desired distance unit
    pbc: bool
        If True, use the peridoc boundary conditions
    Returns
    -------
    (hist,edges): tuple
        The normalized values of the histogram
    """
    box_size = np.array(box_size.to(unit_distance))

    if pbc == True:
        #bins = np.linspace(0,box_size,bin_number+1) #linear
        bins = np.logspace(0, np.log10(box_size), bin_number+1)
    else:
        #bins = np.linspace(0,np.max(data),bin_number+1) #linear
        bins = np.logspace(0,np.log10(np.max(data)),bin_number+1)

    hist, edges = np.histogram(data, bins, density=True)

    return (hist,edges)



# calculate the 2-point correlation function
def calculate_2pcf(
    positions1: np.ndarray,
    boxsize: unyt.array.unyt_array,
    positions2: np.ndarray = None,
    unitdistance: str = 'Mpc/h',
    isprojected: bool = True,
    isprojection_axis: int = 2,
    binnumber: int = 50
) -> tuple:
    """
    Calculate the 2-point correlation function
    Parameters
    ----------
    positions1: np.ndarray
        The positions of the galaxies
    boxsize : unyt.array.unyt_array
        The size of the simulation box
    positions2: np.ndarray
        The random positions if given
    unitdistance: str
        The desired distance unit
    isprojected: bool
        If True, calculate the projected distance
    isprojection_axis: int
        The axis along which to project
        x: 0, y: 1, z: 2
    binnumber:
        The desired number of bins
    Returns
    -------
    (r,tpcf): tuple 
        The mid-point of the bin and the 2-point correlation function
    """

    # get random positions the same size as the data
    if positions2 is None:
        shape_positions = np.array(positions1).shape
        np.random.seed(0)
        positions2 = np.random.uniform(0,
            boxsize.to(unitdistance), shape_positions)

    # get the parwise distances dd,dr,rr
    dist_dd = find_distances_galaxy_pairs(
        positions_gal1= positions1,
        box_size= boxsize,
        unit_distance= unitdistance,
        projected= isprojected,
        projection_axis= isprojection_axis,
    )
    dist_rr = find_distances_galaxy_pairs(
        positions_gal1= positions2,
        box_size= boxsize,
        unit_distance= unitdistance,
        projected= isprojected,
        projection_axis= isprojection_axis,
    )
    dist_dr = find_distances_galaxy_pairs(
        positions_gal1= positions1,
        box_size= boxsize,
        positions_gal2= positions2,
        unit_distance= unitdistance,
        projected= isprojected,
        projection_axis= isprojection_axis,
    )
    # normalize the distribution dd,rr,dr
    normal_dd = get_normalized_histogram(
        data= dist_dd,
        box_size= boxsize,
        bin_number= binnumber,
        unit_distance= unitdistance,
    ) 
    normal_rr = get_normalized_histogram(
        data= dist_rr,
        box_size= boxsize,
        bin_number= binnumber,
        unit_distance= unitdistance,
    ) 
    normal_dr = get_normalized_histogram(
        data= dist_dr,
        box_size= boxsize,
        bin_number= binnumber,
        unit_distance= unitdistance,
    )

    # get mid-point of each bin
    bin_start = np.divide(normal_dd[1][1]-normal_dd[1][0], 2)
    #bin_step = bin_start * 2 #linear
    #r = np.arange(bin_start,normal_dd[1][-1], bin_step) #linear
    r = (normal_dd[1] + bin_start)[:-1]

    counts = np.array([np.array(normal_dd[0]),
                    -2*np.array(normal_dr[0]),
                    np.array(normal_rr[0])])
    
    numerator = np.sum(counts, axis=0)
    tpcf = np.divide(numerator, np.array(normal_rr[0]))

    return (r,tpcf)



# calculate uncertainty real-space 2pcf using k-fold
def find_uncertainty_kfold(
    data: np.ndarray,
    size_box: unyt.array.unyt_array,
    pos_random: np.ndarray = None,
    k: int = 5,
    dist_unit: str = 'Mpc/h',
    proj: bool = True,
    proj_ax: int = 2,
    number_bin: int = 50
 ) -> tuple:
    """
    Calculate the uncertainty of the real-space 2pcf using k-fold
    Parameters
    ----------
    data: np.ndarray
        The positions of the galaxies
    size_box : unyt.array.unyt_array
        The size of the simulation box
    pos_random: np.ndarray
        The random positions if given
    k: int
        The number of folds or samples 
    dist_unit: str
        The desired distance unit
    proj: bool
        If True, calculate the projected distance
    proj_ax: int
        The axis along which to project
        x: 0, y: 1, z: 2
    number_bin:
        The desired number of bins
    Returns
    -------
    (x,uncertainty): tuple 
        The mid-point of the bin and the uncertainty
    """
    
    # split the data into k samples
    np.random.seed(0)
    np.random.shuffle(data)
    data_split = np.array_split(data, k)
    
    if pos_random is None:
    # create random points same size as the data
        np.random.seed(0)
        pos_random = np.random.uniform(0,
            size_box.to(dist_unit), np.array(data).shape)
        random_data_split = np.array_split(pos_random, k)

    else:
        # split the given random positions
        random_data_split = np.array_split(pos_random, k)

    # calculate 2pcf for each sample
    tpcf_sample = []
    for i in range(k):
        (x,y) = calculate_2pcf(
                positions1 = data_split[i],
                boxsize= size_box,
                positions2= random_data_split[i],
                unitdistance= dist_unit,
                isprojected= proj,
                isprojection_axis= proj_ax,
                binnumber= number_bin
                )
        tpcf_sample.append(y)

    tpcf_sample = np.array(tpcf_sample)
    uncertainty = np.std(tpcf_sample, axis=0)

    return (x,uncertainty)




# calculate uncertainty real-space 2pcf using bootstrap
def find_uncertainty_bootstrap(
    data: np.ndarray,
    size_box: unyt.array.unyt_array,
    pos_random: np.ndarray = None,
    k: int = 5,
    dist_unit: str = 'Mpc/h',
    proj: bool = True,
    proj_ax: int = 2,
    number_bin: int = 50
 ) -> tuple:
    """
    Calculate the uncertainty of the real-space 2pcf using bootstrap
    Parameters
    ----------
    data: np.ndarray
        The positions of the galaxies
    size_box : unyt.array.unyt_array
        The size of the simulation box
    pos_random: np.ndarray
        The random positions if given
    k: int
        The number of folds or samples 
    dist_unit: str
        The desired distance unit
    proj: bool
        If True, calculate the projected distance
    proj_ax: int
        The axis along which to project
        x: 0, y: 1, z: 2
    number_bin:
        The desired number of bins
    Returns
    -------
    (x,uncertainty): tuple 
        The mid-point of the bin and the uncertainty
    """

    # create bootstrap samples
    np.random.seed(0)
    samples = []
    for i in range(k):
        one_sample = data[np.random.choice(data.shape[0],
                size=len(data), replace=True)]
        samples.append(one_sample)       

    if pos_random is None:
        # create random points same size as the data
        np.random.seed(0)
        pos_random = np.random.uniform(0,
            size_box.to(dist_unit), np.array(data).shape)

    # create random samples
    np.random.seed(0)
    all_pos_random = []
    for i in range(k):
        one_pos_random = pos_random[np.random.choice(
                pos_random.shape[0],size=len(data),replace=True)]
        all_pos_random.append(one_pos_random)
    all_pos_random = np.array(all_pos_random)    

    # calculate 2pcf for each sample
    tpcf_sample = []
    for i in range(k):
        (x,y) = calculate_2pcf(
                positions1 = samples[i],
                boxsize= size_box,
                positions2= all_pos_random[i],
                unitdistance= dist_unit,
                isprojected= proj,
                isprojection_axis= proj_ax,
                binnumber= number_bin
                )
        tpcf_sample.append(y)

    tpcf_sample = np.array(tpcf_sample)
    uncertainty = np.std(tpcf_sample, axis=0)

    return (x,uncertainty)



# calculate uncertainty real-space 2pcf using cosmic variance
def find_uncertainty_cosmic_var(
    data: np.ndarray,
    size_box: unyt.array.unyt_array,
    pos_random: np.ndarray = None,
    dist_unit: str = 'Mpc/h',
    proj: bool = True,
    proj_ax: int = 2,
    number_bin: int = 50
 ) -> tuple:
    """
    Calculate the uncertainty of the real-space 2pcf using cosmic variance
    Parameters
    ----------
    data: np.ndarray
        The positions of the galaxies
    size_box : unyt.array.unyt_array
        The size of the simulation box
    pos_random: np.ndarray
        The random positions if given
    dist_unit: str
        The desired distance unit
    proj: bool
        If True, calculate the projected distance
    proj_ax: int
        The axis along which to project
        x: 0, y: 1, z: 2
    number_bin:
        The desired number of bins
    Returns
    -------
    (x,uncertainty): tuple 
        The mid-point of the bin and the uncertainty
    """
    l = size_box.to(dist_unit)/2

    # split the data into 8 octants 
    oct = [
        (data[:, 0] < l) & (data[:, 1] < l) & (data[:, 2] < l),
        (data[:, 0] >= l) & (data[:, 1] < l) & (data[:, 2] < l),
        (data[:, 0] < l) & (data[:, 1] >= l) & (data[:, 2] < l),
        (data[:, 0] >= l) & (data[:, 1] >= l) & (data[:, 2] < l),
        (data[:, 0] < l) & (data[:, 1] < l) & (data[:, 2] >= l),
        (data[:, 0] >= l) & (data[:, 1] < l) & (data[:, 2] >= l),
        (data[:, 0] < l) & (data[:, 1] >= l) & (data[:, 2] >= l),
        (data[:, 0] >= l) & (data[:, 1] >= l) & (data[:, 2] >= l)
    ]

    octants = [data[i] for i in oct]

    if pos_random is None:
    # create random points same size as the data
        np.random.seed(0)
        pos_random = np.random.uniform(0,
            size_box.to(dist_unit), np.array(data).shape)
    
    random_oct = [
        (pos_random[:,0]<l)&(pos_random[:,1]<l)&(pos_random[:,2]<l),
        (pos_random[:,0]>=l)&(pos_random[:,1]<l)&(pos_random[:,2]<l),
        (pos_random[:,0]<l)&(pos_random[:,1]>=l)&(pos_random[:,2]<l),
        (pos_random[:,0]>=l)&(pos_random[:,1]>=l)&(pos_random[:,2]<l),
        (pos_random[:,0]<l)&(pos_random[:,1]<l)&(pos_random[:,2]>=l),
        (pos_random[:,0]>=l)&(pos_random[:,1]<l)&(pos_random[:,2]>=l),
        (pos_random[:,0]<l)&(pos_random[:,1]>=l)&(pos_random[:,2]>=l),
        (pos_random[:,0]>=l)&(pos_random[:,1]>=l)&(pos_random[:,2]>=l)
    ]

    random_octants = [pos_random[i] for i in random_oct]

    # calculate 2pcf for each octant
    tpcf_sample = []
    for i in range(8):
        (x,y) = calculate_2pcf(
                positions1 = octants[i],
                boxsize= size_box/2,
                positions2= random_octants[i],
                unitdistance= dist_unit,
                isprojected= proj,
                isprojection_axis= proj_ax,
                binnumber= number_bin//2
                )
        tpcf_sample.append(y)

    tpcf_sample = np.array(tpcf_sample)
    other_half = np.zeros((8,number_bin//2))
    all_tpcf = np.concatenate([tpcf_sample,other_half],axis=1)
    uncertainty = np.std(all_tpcf, axis=0)
    
    return (x,uncertainty)



# calculate pairwise distances uncertainty using bootstrap
def find_pairdist_uncertainty_bootstrap(
    data: np.ndarray,
    size_box: unyt.array.unyt_array,
    pos_random: np.ndarray = None,
    k: int = 5,
    dist_unit: str = 'Mpc/h',
    proj: bool = True,
    proj_ax: int = 2,
    periodic_bound: bool = True,
    number_bin: int = 50
 ) -> tuple:
    """
    Calculate the uncertainty of pairwise distances using bootstrap
    Parameters
    ----------
    data: np.ndarray
        The positions of the galaxies
    size_box : unyt.array.unyt_array
        The size of the simulation box
    pos_random: np.ndarray
        The random positions if given
    k: int
        The number of folds or samples 
    dist_unit: str
        The desired distance unit
    proj: bool
        If True, calculate the projected distance
    proj_ax: int
        The axis along which to project
        x: 0, y: 1, z: 2
    periodic_bound: bool
        If True, use the peridoc boundary conditions
    Returns
    -------
    (x,uncertainty): tuple 
        The bin and the uncertainty
    """

    # split data into samples
    np.random.seed(0)
    data_split = np.array_split(data, k)
    number = len(data_split[0])

    samples = []
    for i in range(k):
        one_sample = data[np.random.choice(data.shape[0],
                        size=number, replace=True)]
        samples.append(one_sample)       
    samples = np.array(samples)

    if pos_random is None:
        # create random points same size as the data
        np.random.seed(0)
        pos_random = np.random.uniform(0,
            size_box.to(dist_unit), np.array(data).shape)
        random_data_split = np.array_split(pos_random, k)
        num = len(random_data_split[0])

    else:
        # split the given random positions
        random_data_split = np.array_split(pos_random, k)
        num = len(random_data_split[0])

    # create random samples
    np.random.seed(0)
    all_pos_random = []
    for i in range(k):
        one_pos_random = pos_random[np.random.choice(
                pos_random.shape[0],size=num,replace=True)]
        all_pos_random.append(one_pos_random)
    all_pos_random = np.array(all_pos_random)

    # calculate pairwise distances for each sample
    pairwise_dist = []
    for i in range(k):
        one_pair_dist = find_distances_galaxy_pairs(
            positions_gal1= samples[i],
            box_size= size_box,
            positions_gal2= all_pos_random[i],
            unit_distance= dist_unit,
            projected= proj,
            projection_axis= proj_ax,
            pbc= periodic_bound
        )
        pairwise_dist.append(one_pair_dist)

    norm_pairwise_dist = []
    for i in range(k):
        one_norm_pairwise_dist = get_normalized_histogram(
            data= np.array(pairwise_dist),
            box_size= size_box,
            bin_number= number_bin,
            unit_distance= dist_unit,
            pbc= periodic_bound
        )
        norm_pairwise_dist.append(one_norm_pairwise_dist)

    x = norm_pairwise_dist[0][1]
    uncertainty = np.std(
        np.array([i[0] for i in norm_pairwise_dist]), axis=0)

    return (x,uncertainty)



# calculate perpendicular and parallel separations of 2pcf
def compute_separation(
    positions1: np.ndarray,
    boxsize: unyt.array.unyt_array,
    positions2: np.ndarray = None,
    unitdistance: str = 'Mpc/h',
    proj: bool = True,
    proj_axis: int = 2,
    periodic_bound: bool = True
) -> tuple:
    """
    Calculate perpendicular and parallel separations of 2pcf
    Parameters
    ----------
    positions1: np.ndarray
        The positions of the galaxies
    boxsize : unyt.array.unyt_array
        The size of the simulation box
    positions2: np.ndarray
        The positions of all the galaxies if given
    unitdistance: str
        The desired distance unit
    proj: bool
        If True, calculate the projected distance
    proj_axis: int
        The axis along which to project
        x: 0, y: 1, z: 2
    Returns
    -------
    rp,pi : tuple
        The perpendicular and parallel separations
    """

    #get pairwise distances of positions on the projection axis
    pos_pi1 = positions1[:,proj_axis]
    if positions2 is None:
        u,v = np.triu_indices(len(pos_pi1), k=1)
        pi = pos_pi1[v] - pos_pi1[u]

    else:
        pos_pi2 = positions2[:, proj_axis]
        pi = pos_pi2[:, np.newaxis] - pos_pi1
        #pi = pi.reshape(-1, pi.shape[-1])
        pi = pi.flatten()

    the_box_size = np.float64(boxsize.to(unitdistance))

    # set periodic boundary conditions
    pi[pi > (the_box_size/2)] -= the_box_size
    pi[pi <= -(the_box_size/2)] += the_box_size

    # get pairwise distances of positions other than the projection axis
    rp = find_distances_galaxy_pairs(
        positions_gal1= positions1,
        box_size= boxsize,
        positions_gal2= positions2,
        unit_distance= unitdistance,
        projected= proj,
        projection_axis= proj_axis,
    )

    return rp, np.abs(pi)



# Compute ξ(rp, pi) using a 2D histogram
def get_2d_histogram(
    xdata: np.ndarray,
    ydata: np.ndarray,
    bin_number: int = 50,
) -> tuple:
    """
    Create a 2 dimensional histogram
    Parameters
    ----------
    data: np.ndarray
        The data
    box_size : unyt.array.unyt_array
        The size of the simulation box
    bin_number:
        The desired number of bins
    unit_distance: str
        The desired distance unit
    pbc: bool
        If True, use the peridoc boundary conditions
    Returns
    -------
    (hist,xedges,yedges): tuple
        The values of the histogram
    """

    bins1 = np.logspace(0,np.log10(np.max(xdata)),bin_number+1)
    bins2 = np.logspace(0,np.log10(np.max(ydata)),bin_number+1)

    hist, xedges, yedges = np.histogram2d(
        xdata, ydata, bins=[bins1,bins2])

    return (hist,xedges,yedges)



# calculate projected correlation function 
def calculate_projected_2pcf(
    data1: np.ndarray,
    box_size: unyt.array.unyt_array,
    data2: np.ndarray = None,
    unit_dist: str = 'Mpc/h',
    isprojected: bool = True,
    isprojection_axis: int = 2,
    binnumber: int = 30,
) -> tuple:
    """
    Calculate the projected 2-point correlation function
    Parameters
    ----------
    data: np.ndarray
        The positions of the galaxies
    box_size : unyt.array.unyt_array
        The size of the simulation box
    data2: np.ndarray
        The positions of all the galaxies
    unit dist: str
        The desired distance unit
    isprojected: bool
        If True, calculate the projected distance
    isprojection_axis: int
        The axis along which to project
        x: 0, y: 1, z: 2
    binnumber:
        The desired number of bins
    Returns
    -------
    (rp,wp): tuple 
        The mid-point of the bin and the projected 2pcf
    """

    # get perpendicular and parallel separations 
    rp_data, pi_data = compute_separation(
        positions1= data1,
        boxsize= box_size,
        positions2= data2,
        unitdistance= unit_dist,
        proj= isprojected,
        proj_axis= isprojection_axis,
    )

    # Compute ξ(rp, pi) using a 2D histogram
    hist1,rp_edges1,pi_edges1 = get_2d_histogram(
        xdata= rp_data,
        ydata= pi_data,
        bin_number= binnumber
        )

    # normalize by a random distribution
    shape1 = data1.shape
    np.random.seed(0)
    pos_random1 = np.random.uniform(0, box_size.to(unit_dist), shape1)

    if data2 is not None:
        shape2 = data2.shape
        np.random.seed(0)
        pos_random2 = np.random.uniform(0, box_size.to(unit_dist), shape2)
    else:
        pos_random2 = None

    rp_ran, pi_ran = compute_separation(
        positions1= pos_random1,
        boxsize= box_size,
        positions2= pos_random2,
        unitdistance= unit_dist,
        proj= isprojected,
        proj_axis= isprojection_axis,
    )

    hist2,rp_edges2,pi_edges2 = get_2d_histogram(
        xdata= rp_ran,
        ydata= pi_ran,
        bin_number= binnumber
        )

    xi = np.divide(hist1,hist2)

    # Integrate for each rp bin
    wp_values = []
    for i in range(len(rp_edges1) - 1):
        wp_values.append(2 * trapz(xi[i, :], pi_edges1[:-1]))

    # get mid-point of each bin
    bin_start = np.divide(rp_edges1[1]-rp_edges1[0], 2)
    rp = (rp_edges1 + bin_start)[:-1]
    
    return rp, np.array(wp_values)



# calculate projected 2pcf uncertainty using k-fold
def find_ptpcf_uncertainty_kfold(
    positions1: np.ndarray,
    size_box: unyt.array.unyt_array,
    positions2: np.ndarray = None,
    k: int = 5,
    dist_unit: str = 'Mpc/h',
    proj: bool = True,
    proj_ax: int = 2,
    number_bin: int = 50
 ) -> tuple:
    """
    Calculate the uncertainty of projected 2pcf using k-fold
    Parameters
    ----------
    data: np.ndarray
        The positions of the galaxies
    size_box : unyt.array.unyt_array
        The size of the simulation box
    positions2: np.ndarray
        The positions of all the galaxies
    k: int
        The number of folds or samples 
    dist_unit: str
        The desired distance unit
    proj: bool
        If True, calculate the projected distance
    proj_ax: int
        The axis along which to project
        x: 0, y: 1, z: 2
    number_bin:
        The desired number of bins
    Returns
    -------
    (x,uncertainty): tuple 
        The mid-point of each bin and the uncertainty
    """
    
    # split the data into k samples
    np.random.seed(0)
    np.random.shuffle(positions1)
    data_split1 = np.array_split(positions1, k)

    if positions2 is not None:
        np.random.seed(0)
        np.random.shuffle(positions2)
        data_split2 = np.array_split(positions2, k)
    else:
        data_split2 = None


    # calculate projected 2pcf for each sample
    ptpcf_sample = []
    if data_split2 is not None:
        for i in range(k):
            (x,y) = calculate_projected_2pcf(
                    data1 = data_split1[i],
                    box_size= size_box,
                    data2 = data_split2[i],
                    unit_dist= dist_unit,
                    isprojected= proj,
                    isprojection_axis= proj_ax,
                    binnumber= number_bin
                    )
            ptpcf_sample.append(y)

    else:
        for i in range(k):
            (x,y) = calculate_projected_2pcf(
                    data1 = data_split1[i],
                    box_size= size_box,
                    data2 = data_split2,
                    unit_dist= dist_unit,
                    isprojected= proj,
                    isprojection_axis= proj_ax,
                    binnumber= number_bin
                    )
            ptpcf_sample.append(y)

    ptpcf_sample = np.array(ptpcf_sample)
    uncertainty = np.std(ptpcf_sample, axis=0)

    return (x,uncertainty)



# calculate projected 2pcf uncertainty using bootstrap
def find_ptpcf_uncertainty_bootstrap(
    positions1: np.ndarray,
    size_box: unyt.array.unyt_array,
    positions2: np.ndarray = None,
    k: int = 5,
    dist_unit: str = 'Mpc/h',
    proj: bool = True,
    proj_ax: int = 2,
    number_bin: int = 50
 ) -> tuple:
    """
    Calculate the uncertainty of projected 2pcf using bootstrap
    Parameters
    ----------
    data: np.ndarray
        The positions of the galaxies
    size_box : unyt.array.unyt_array
        The size of the simulation box
    positions2: np.ndarray
        The positions of all the galaxies
    k: int
        The number of folds or samples 
    dist_unit: str
        The desired distance unit
    proj: bool
        If True, calculate the projected distance
    proj_ax: int
        The axis along which to project
        x: 0, y: 1, z: 2
    number_bin:
        The desired number of bins
    Returns
    -------
    (x,uncertainty): tuple 
        The mid-point of each bin and the uncertainty
    """
    
    # create bootstrap samples
    np.random.seed(0)
    samples1 = []
    for i in range(k):
        one_sample1 = positions1[np.random.choice(positions1.shape[0],
                size=len(positions1), replace=True)]
        samples1.append(one_sample1)

    if positions2 is not None:
        np.random.seed(0)
        samples2 = []
        for i in range(k):
            one_sample2 = positions2[np.random.choice(positions2.shape[0],
                    size=len(positions2), replace=True)]
            samples2.append(one_sample2)
    else:
        samples2 = None


    # calculate projected 2pcf for each sample
    ptpcf_sample = []
    if samples2 is not None:
        for i in range(k):
            (x,y) = calculate_projected_2pcf(
                    data1 = samples1[i],
                    box_size= size_box,
                    data2 = samples2[i],
                    unit_dist= dist_unit,
                    isprojected= proj,
                    isprojection_axis= proj_ax,
                    binnumber= number_bin
                )
            ptpcf_sample.append(y)

    else:
        for i in range(k):
            (x,y) = calculate_projected_2pcf(
                    data1 = samples1[i],
                    box_size= size_box,
                    data2 = samples2,
                    unit_dist= dist_unit,
                    isprojected= proj,
                    isprojection_axis= proj_ax,
                    binnumber= number_bin
                )
            ptpcf_sample.append(y)

    ptpcf_sample = np.array(ptpcf_sample)
    uncertainty = np.std(ptpcf_sample, axis=0)

    return (x,uncertainty)



# calculate projected 2pcf uncertainty using cosmic variance
def find_ptpcf_uncertainty_cosmic_var(
    positions1: np.ndarray,
    size_box: unyt.array.unyt_array,
    positions2: np.ndarray = None,
    dist_unit: str = 'Mpc/h',
    proj: bool = True,
    proj_ax: int = 2,
    number_bin: int = 50
 ) -> tuple:
    """
    Calculate the uncertainty of projected 2pcf using cosmic variance
    Parameters
    ----------
    positions1: np.ndarray
        The positions of the galaxies
    size_box : unyt.array.unyt_array
        The size of the simulation box
    positions2: np.ndarray
        The positions of all the galaxies
    dist_unit: str
        The desired distance unit
    proj: bool
        If True, calculate the projected distance
    proj_ax: int
        The axis along which to project
        x: 0, y: 1, z: 2
    number_bin:
        The desired number of bins
    Returns
    -------
    (x,uncertainty): tuple 
        The mid-point of each bin and the uncertainty
    """
    
    l = size_box.to(dist_unit)/2

    # split the box into 8 octants 
    oct1 = [
        (positions1[:, 0] < l) & (positions1[:, 1] < l) & (positions1[:, 2] < l),
        (positions1[:, 0] < l) & (positions1[:, 1] < l) & (positions1[:, 2] >= l),
        (positions1[:, 0] < l) & (positions1[:, 1] >= l) & (positions1[:, 2] < l),
        (positions1[:, 0] < l) & (positions1[:, 1] >=l) & (positions1[:, 2] >= l),
        (positions1[:, 0] >= l) & (positions1[:, 1] >= l) & (positions1[:, 2] >= l),
        (positions1[:, 0] >= l) & (positions1[:, 1] >= l) & (positions1[:, 2] < l),
        (positions1[:, 0] >= l) & (positions1[:, 1] < l) & (positions1[:, 2] >= l),
        (positions1[:, 0] >= l) & (positions1[:, 1] < l) & (positions1[:, 2] < l)
    ]
    octants1 = [positions1[i] for i in oct1]

    if positions2 is not None:
        oct2 = [
            (positions2[:, 0] < l) & (positions2[:, 1] < l) & (positions2[:, 2] < l),
            (positions2[:, 0] < l) & (positions2[:, 1] < l) & (positions2[:, 2] >= l),
            (positions2[:, 0] < l) & (positions2[:, 1] >= l) & (positions2[:, 2] < l),
            (positions2[:, 0] < l) & (positions2[:, 1] >=l) & (positions2[:, 2] >= l),
            (positions2[:, 0] >= l) & (positions2[:, 1] >= l) & (positions2[:, 2] >= l),
            (positions2[:, 0] >= l) & (positions2[:, 1] >= l) & (positions2[:, 2] < l),
            (positions2[:, 0] >= l) & (positions2[:, 1] < l) & (positions2[:, 2] >= l),
            (positions2[:, 0] >= l) & (positions2[:, 1] < l) & (positions2[:, 2] < l)
        ]
        octants2 = [positions2[i] for i in oct2]
    else:
        octants2 = octants1

    # calculate projected 2pcf for each octant
    ptpcf_sample = []
    for i in range(8):
        (x,y) = calculate_projected_2pcf(
                data1 = octants1[i],
                box_size= size_box,
                data2 = octants2[i],
                unit_dist= dist_unit,
                isprojected= proj,
                isprojection_axis= proj_ax,
                binnumber= number_bin
                )
        ptpcf_sample.append(y)

    ptpcf_sample = np.array(ptpcf_sample)
    uncertainty = np.std(ptpcf_sample, axis=0)

    return (x,uncertainty)




# calculate projected correlation function with bootstrap uncertainties of a large data
def calculate_projected_2pcf_split(
    large_positions: np.ndarray,
    boxsize: unyt.array.unyt_array,
    unitdist: str = 'Mpc/h',
    projected: bool = True,
    projection_axis: int = 2,
    binnum: int = 10,
    N: int = 10,
    K: int = 5
) -> tuple:
    """
    Calculate the projected 2-point correlation function of a large data
        with errors estimated using the bootstrap resampling
    Parameters
    ----------
    positions1: np.ndarray
        The positions of the galaxies
    boxsize : unyt.array.unyt_array
        The size of the simulation box
    positions2: np.ndarray
        The positions of all the galaxies    
    unitdist: str
        The desired distance unit
    projected: bool
        If True, calculate the projected distance
    projection_axis: int
        The axis along which to project
        x: 0, y: 1, z: 2
    binnum:
        The desired number of bins
    N: int
        The number of the division of the original large data
    K: int
        The number of folds or samples
    Returns
    -------
    mean_rp, proj_tpcf, error_proj_tpcf: tuple 
        The mid-point of each bin, the projected 2pcf and the uncertainties
    """

    # Split the positions into N
    positions_split = np.array_split(large_positions, N, axis=0)

    # Calculate 2pcf for each of the k
    all_rp = []
    all_wp = []
    for i in range(N):
        one_rp, one_wp = calculate_projected_2pcf(
            data1 = positions_split[i],
            box_size= boxsize,
            data2 = large_positions,
            unit_dist= unitdist,
            isprojected= projected,
            isprojection_axis= projection_axis,
            binnumber= binnum
        )
        all_rp.append(one_rp)
        all_wp.append(one_wp)

    # Commbine the outputs
    mean_rp = np.mean(all_rp, axis=0)
    proj_tpcf = np.sum(all_wp, axis=0)


    # Uncertainty using bootstrap
    error_rp = []
    error_wp = []
    for i in range(N):
        one_error_rp, one_error_wp = find_ptpcf_uncertainty_bootstrap(
            positions1 = positions_split[i],
            size_box= boxsize,
            positions2= large_positions,
            k= K,
            dist_unit= unitdist,
            proj= projected,
            proj_ax= projection_axis,
            number_bin= binnum
        )
        error_rp.append(one_error_rp)
        error_wp.append(one_error_wp)

    # Commbine the outputs
    mean_error = np.mean(error_rp, axis=0)
    error_proj_tpcf = np.sum(error_wp, axis=0)

    return mean_rp, proj_tpcf, error_proj_tpcf




# calculate projected correlation function with bootstrap uncertainties of a large data
def calculate_projected_2pcf_split(
    large_positions: np.ndarray,
    boxsize: unyt.array.unyt_array,
    unitdist: str = 'Mpc/h',
    projected: bool = True,
    projection_axis: int = 2,
    binnum: int = 10,
    N: int = 10,
    K: int = 5
) -> tuple:
    """
    Calculate the projected 2-point correlation function of a large data
        with errors estimated using the bootstrap resampling
    Parameters
    ----------
    positions1: np.ndarray
        The positions of the galaxies
    boxsize : unyt.array.unyt_array
        The size of the simulation box
    positions2: np.ndarray
        The positions of all the galaxies    
    unitdist: str
        The desired distance unit
    projected: bool
        If True, calculate the projected distance
    projection_axis: int
        The axis along which to project
        x: 0, y: 1, z: 2
    binnum:
        The desired number of bins
    N: int
        The number of the division of the original large data
    K: int
        The number of folds or samples
    Returns
    -------
    mean_rp, proj_tpcf, error_proj_tpcf: tuple 
        The mid-point of each bin, the projected 2pcf and the uncertainties
    """

    # get perpendicular and parallel separations 
    rp_large_data, pi_large_data = compute_separation(
        positions1= large_positions,
        boxsize= boxsize,
        unitdistance= unitdist,
        proj= projected,
        proj_axis= projection_axis,
    )

    # Compute ξ(rp, pi) using a 2D histogram
    hist1,rp_edges1,pi_edges1 = get_2d_histogram(
        xdata= rp_large_data,
        ydata= pi_large_data,
        bin_number= binnum
        )

    # get mid-point of each bin
    bin_start = np.divide(rp_edges1[1]-rp_edges1[0], 2)
    rp = (rp_edges1 + bin_start)[:-1]    


    # Split the positions into N chunks
    positions_split = np.array_split(large_positions, N, axis=0)

    # Calculate 2pcf wp for each of the k
    all_wp = []
    for i in range(N):
        one_rp, one_wp = calculate_projected_2pcf(
            data1 = positions_split[i],
            box_size= boxsize,
            data2 = large_positions,
            unit_dist= unitdist,
            isprojected= projected,
            isprojection_axis= projection_axis,
            binnumber= binnum
        )
        all_wp.append(one_wp)

    # Commbine the outputs
    proj_tpcf = np.sum(all_wp, axis=0)


    # Uncertainty using bootstrap
    error_wp = []
    for i in range(N):
        one_error_rp, one_error_wp = find_ptpcf_uncertainty_bootstrap(
            positions1 = positions_split[i],
            size_box= boxsize,
            positions2= large_positions,
            k= K,
            dist_unit= unitdist,
            proj= projected,
            proj_ax= projection_axis,
            number_bin= binnum
        )
        error_wp.append(one_error_wp)

    # Commbine the outputs
    error_proj_tpcf = np.sum(error_wp, axis=0)

    return rp, proj_tpcf, error_proj_tpcf








