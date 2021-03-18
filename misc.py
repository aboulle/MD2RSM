"""
DxTools: Processing XRD data files recorded with the Bruker D8 diffractometer
Copyright 2016, Alexandre  Boulle
alexandre.boulle@unilim.fr
"""
import numpy as np
from scipy.optimize import  leastsq

def pVoigt(x, p):
	maximum = p[0]
	pos = p[1]
	FWHM = p[2]
	eta = p[3]
	a = p[4]
	b = p[5]
	gauss = maximum * np.exp(-np.log(2.) * ((x-pos)/(0.5*FWHM))**2)
	lorentz = maximum / (1. + ((x - pos)/(0.5*FWHM))**2)
	return eta*lorentz + (1-eta)*gauss + a*x + b

def splitpVoigt(x, p):
	maximum = p[0]
	pos = p[1]
	FWHM1 = p[2]
	FWHM2 = p[3]
	eta = p[4]
	if eta > 1:
		eta = 1
	elif eta < 0:
		eta = 0

	gauss1 = maximum * np.exp(-np.log(2.) * ((x-pos)/(0.5*FWHM1))**2)
	lorentz1 = maximum / (1. + ((x - pos)/(0.5*FWHM1))**2)
	pV1 = eta*lorentz1 + (1-eta)*gauss1
	gauss2 = maximum * np.exp(-np.log(2.) * ((x-pos)/(0.5*FWHM2))**2)
	lorentz2 = maximum / (1. + ((x - pos)/(0.5*FWHM2))**2)
	pV2 = eta*lorentz2 + (1-eta)*gauss2
	pV1[x>pos]=pV2[x>pos]
	return pV1

def gauss(x,p):
	maximum = p[0]
	pos = p[1]
	FWHM = p[2]
	gauss = maximum * np.exp(-np.log(2.) * ((x-pos)/(0.5*FWHM))**2)
	return gauss #+ a*x + b

def splitgauss(x,p):
	maximum = p[0]
	pos = p[1]
	FWHM1 = p[2]
	FWHM2 = p[3]
	gauss1 = maximum * np.exp(-np.log(2.) * ((x-pos)/(0.5*FWHM1))**2)
	gauss2 = maximum * np.exp(-np.log(2.) * ((x-pos)/(0.5*FWHM2))**2)
	gauss1[x>pos]=gauss2[x>pos]
	return gauss1 #+ a*x + b

def guess_pV(x,y):
	a = 0
	b = y.min()
	maximum = y.max()
	pos = x[y==maximum][0]
	d=y-(maximum/2.) - b/2.
	indexes = np.where(d > 0)[0]
	FWHM = np.abs(x[indexes[-1]] - x[indexes[0]])
	eta = 0.5
	return np.array([maximum-b, pos, FWHM, eta, a, b])

def guess_splitpV(x,y):
	a = 0
	b = y.min()
	maximum = y.max()
	pos = x[y==maximum][0]
	d=y-(maximum/2.) - b/2.
	indexes = np.where(d > 0)[0]
	FWHM = np.abs(x[indexes[-1]] - x[indexes[0]])
	eta = 0.5
	return np.array([maximum-b, pos, FWHM, FWHM, eta])

def guess_gauss(x,y):
	a = 0
	b = y.min()
	maximum = y.max()
	pos = x[y==maximum][0]
	d=y-(maximum/2.) - b/2.
	indexes = np.where(d > 0)[0]
	FWHM = np.abs(x[indexes[-1]] - x[indexes[0]])

	return np.array([maximum-b, pos, FWHM])

def guess_splitgauss(x,y):
	a = 0
	b = y.min()
	maximum = y.max()
	pos = x[y==maximum][0]
	d=y-(maximum/2.) - b/2.
	indexes = np.where(d > 0)[0]
	FWHM = np.abs(x[indexes[-1]] - x[indexes[0]])

	return np.array([maximum-b, pos, FWHM, FWHM])

def pVoigt_area(p):
	maximum = p[0]
	FWHM = p[2]
	eta = p[3]
	beta = (eta*np.pi*FWHM/2.) + (1-eta)*(FWHM/2.)*np.sqrt(np.pi/np.log(2))
	return beta*maximum

def pVoigt_area_err(p, perr):
	maximum = p[0]
	dmax = perr[0]
	FWHM = p[2]
	dFWHM = perr[2]
	eta = p[3]
	deta = perr[3]

	dbeta2 = ((eta*np.pi*dFWHM/2.)**2) + ((deta*np.pi*FWHM/2.)**2) + (((1-eta)*(dFWHM/2.)*np.sqrt(np.pi/np.log(2)))**2) + ((deta*(FWHM/2.)*np.sqrt(np.pi/np.log(2)))**2)

	beta = (eta*np.pi*FWHM/2.) + (1-eta)*(FWHM/2.)*np.sqrt(np.pi/np.log(2))
	area = beta*maximum
	err_area = area*np.sqrt((dmax/maximum)**2 + (dbeta2/(beta**2)))

	return err_area

def pV_fit(x,y):
	errfunc = lambda p, x, y: (pVoigt(x, p) - y)#/(y**0.5)
	p0 = guess_pV(x,y)
	p1 = leastsq(errfunc, p0[:], args=(x, y))[0]
	return p1

def splitpV_fit(x,y):
	errfunc = lambda p, x, y: (splitpVoigt(x, p) - y)#/(y**0.5)
	p0 = guess_splitpV(x,y)
	p1 = leastsq(errfunc, p0[:], args=(x, y))[0]
	return p1

def gauss_fit(x,y):
	errfunc = lambda p, x, y: (splitgauss(x, p) - y)#/(y**0.5)
	p0 = guess_splitgauss(x,y)
	p1 = leastsq(errfunc, p0[:], args=(x, y))[0]
	return p1

def splitgauss_fit(x,y,p0):
	errfunc = lambda p, x, y: (splitgauss(x, p) - y)#/(y**0.5)
	#p0 = guess_splitgauss(x,y)
	p1 = leastsq(errfunc, p0[:], args=(x, y))[0]
	return p1

def pVfit_param_err(x,y):
	errfunc = lambda p, x, y: (pVoigt(x, p) - y)#/(y**0.5)
	p0 = guess_param(x,y)
	p1, pcov, infodict, errmsg, success = leastsq(errfunc, p0[:], args=(x, y), full_output=1)

	#compute esd from fit
	if (len(y) > len(p0)) and pcov is not None:
		s_sq = (errfunc(p1, x, y)**2).sum()/(len(y)-len(p0))
		pcov = pcov * s_sq
	else:
		pcov = np.inf

	error = []
	for i in range(len(p1)):
		try:
			error.append(np.abs(pcov[i,i])**0.5)
		except:
			error.append( 0.00 )

	return p1, error

def file_nb(path):
    number = int(path.split(".")[-1])
    return number

def rotate_x(y,z,angle):
	yr = y*np.cos(angle*np.pi/180) - z*np.sin(angle*np.pi/180)
	zr = y*np.sin(angle*np.pi/180) + z*np.cos(angle*np.pi/180)
	return yr, zr

def rotate_y(x,z,angle):
	xr = x*np.cos(angle*np.pi/180) + z*np.sin(angle*np.pi/180)
	zr = -x*np.sin(angle*np.pi/180) + z*np.cos(angle*np.pi/180)
	return xr, zr

def rotate_z(x,y,angle):
	xr = x*np.cos(angle*np.pi/180) - y*np.sin(angle*np.pi/180)
	yr = x*np.sin(angle*np.pi/180) + y*np.cos(angle*np.pi/180)
	return xr, yr

def rotate_coords(xA, yA, zA, rot_angles):
	rotx, roty, rotz = rot_angles[0], rot_angles[1], rot_angles[2]
	# 1 rotation angle
	if (rotx != 0) and (roty == 0) and (rotz == 0):
	    yA, zA = rotate_x(yA, zA, rotx)

	if (rotx == 0) and (roty != 0) and (rotz == 0):
	    xA, zA = rotate_y(xA, zA, roty)

	if (rotx == 0) and (roty == 0) and (rotz != 0):
	    xA, yA = rotate_z(xA, yA, rotz)

	# 2 rotation angles: order X / Y / Z
	if (rotx != 0) and (roty != 0) and (rotz == 0):
		# rotate around x
		yA, zA = rotate_x(yA, zA, rotx)
		# rotate around y
		xA, zA = rotate_y(xA, zA, roty)

	if (rotx != 0) and (roty == 0) and (rotz != 0):
		# rotate around x
		yA, zA = rotate_x(yA, zA, rotx)
		# rotate around z
		xA, yA = rotate_z(xA, yA, rotz)
	if (rotx == 0) and (roty != 0) and (rotz != 0):
		# rotate around y
		xA, zA = rotate_y(xA, zA, roty)
		# rotate around z
		xA, yA = rotate_z(xA, yA, rotz)
    # 3 rotation angles: order X / Y / Z
	if (rotx != 0) and (roty != 0) and (rotz != 0):
		# rotate around x
		yA, zA = rotate_x(yA, zA, rotx)
		# rotate around y
		xA, zA = rotate_y(xA, zA, roty)
		# rotate around z
		xA, yA = rotate_z(xA, yA, rotz)

	return xA, yA, zA

def rsm_4fold(rsm):
	jc = int((np.shape(rsm)[1]-1)/2)
	ic = int((np.shape(rsm)[0]-1)/2)

	bg = np.flip(rsm[:ic+1, :jc+1], axis=(0,1))
	bd = np.flip(rsm[:ic+1, jc:], axis=0)
	hg = np.flip(rsm[ic:, :jc+1], axis=1)
	hd = rsm[ic:, jc:]
	return bg+bd+hg+hd

def arb_scan(data, x0, y0, x1, y1, step, width):
	length = int(np.hypot(x1-x0, y1-y0))
	angle = np.arccos((y1 - y0)/length)
	scan = np.zeros(length)
	w_range = np.arange(-width, width +1)

	for w in w_range:
		dy = int(np.round(w * np.sin(angle)))
		dx = int(np.round(w * np.cos(angle)))
		xs, ys = np.linspace(x0+dx, x1+dx, length), np.linspace(y0-dy, y1-dy, length)
		# if w == w_range.min():
		# 	min_coords = np.array[[][]]
	
	xs, ys = np.linspace(x0, x1, length), np.linspace(y0, y1, length)
	scan = data[xs.astype(np.int), ys.astype(np.int)]

	# xmin = x0 - int(np.round(width * np.cos(angle)))
	# xmax = x0 + int(np.round(width * np.cos(angle)))
	# ymin = y0 + int(np.round(width * np.sin(angle)))
	# ymax = y0 - int(np.round(width * np.sin(angle)))

	return scan#, [xmin, xmax], [ymin,ymax]

def new_func(dy, dx):
     print(dy, dx)