import time
import numpy as np
from pynx import scattering
from pynx.scattering import fthomson
from pandas import read_csv
from misc import rotate_coords, file_nb

def RSM(f_name, cation, anion, h, k, l, acryst, rot_angles, outpath, sublattices=False):
    # Read data file
    nb = file_nb(f_name)  # fluence = (nb+1)*0.0875
    coord = read_csv(f_name, delim_whitespace=True, header=None, names=["x", "y", "z", "Name"], skiprows=2)

    cat = cation[:-2]
    an = anion[:-2]
    print(cat,an)
    # Extract coordinates and convert to fractional
    xU = coord[coord.Name == cat].x.values / acryst
    yU = coord[coord.Name == cat].y.values / acryst
    zU = coord[coord.Name == cat].z.values / acryst
    xO = coord[coord.Name == an].x.values / acryst
    yO = coord[coord.Name == an].y.values / acryst
    zO = coord[coord.Name == an].z.values / acryst

    xU, yU, zU = rotate_coords(xU, yU, zU, rot_angles)
    xO, yO, zO = rotate_coords(xO, yO, zO, rot_angles)

    # Compute scattering for each sub-lattice
    print("Computing scattering for the cationic sub-lattice...")
    fhklU, dt = scattering.Fhkl_thread(h, k, l, xU, yU, zU, verbose=False, gpu_name="P5000", language="opencl")
    print("Computing time:", dt)
    print("Computing scattering for the anionic sub-lattice...")
    fhklO, dt = scattering.Fhkl_thread(h, k, l, xO, yO, zO, verbose=False, gpu_name="P5000", language="opencl")
    print("Computing time:", dt)

    # Compute atomic scattering factor for CuKa1
    print("Computing full scattering...")
    t0 = time.time()
    s = np.sqrt(h ** 2 + k ** 2 + l ** 2) / acryst
    fU = fthomson.f_thomson(s, cation)# - 4.173 + 13.418j
    fO = fthomson.f_thomson(s, anion) #+ 0.050 + 0.032j

    # Full scattering
    if sublattices:
        FU = fhklU * fU
        iU = (abs(FU) ** 2).sum(axis=1)
        FO = fhklO * fO
        iO = (abs(FO) ** 2).sum(axis=1)
        FUO2 = FU + FO 
        iUO2 = (abs(FUO2) ** 2).sum(axis=1)
    else:
        FUO2 = fhklU * fU + fhklO * fO
        iUO2 = (abs(FUO2) ** 2).sum(axis=1)
        iU = 0
        iO = 0
    print("Computing time: %5.4f" %(time.time()-t0))
    # Save and plot
    print("Saving RSM data...")
    rotx, roty, rotz = rot_angles[0], rot_angles[1], rot_angles[2]
    np.savetxt(outpath + "RSM_UO2-"+str(rotx)+"-"+ str(roty)+"-"+str(rotz)+"."+str(nb), iUO2, fmt="%10.8f")
    if sublattices:
        np.savetxt(outpath + "RSM_U-"+str(rotx)+"-"+ str(roty)+"-"+str(rotz)+"."+str(nb), iU, fmt="%10.8f")
        np.savetxt(outpath + "RSM_O-"+str(rotx)+"-"+ str(roty)+"-"+str(rotz)+"."+str(nb), iO, fmt="%10.8f")

    return iUO2, iU, iO
