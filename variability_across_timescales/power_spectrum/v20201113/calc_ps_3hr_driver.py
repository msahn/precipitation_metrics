import cdms2 as cdms
import MV2 as MV
import glob
import cdutil
import cdtime
import genutil
import numpy as np
from regrid2 import Regridder
from scipy import signal
from scipy.stats import chi2
import scipy.interpolate as interp
import pandas as pd
import math
import os
from pcmdi_metrics.pcmdi.pmp_parser import PMPParser

P = PMPParser()

P.add_argument("--mip",
               type=str,
               dest='mip',
               default=None,
               help="cmip5, cmip6 or other mip")

P.add_argument("--var",
               type=str,
               dest='var',
               default=None,
               help="pr or other variable")

P.add_argument("--dir",
               type=str,
               dest='dir',
               default=None,
               help="data directory")

P.add_argument("--prd",
               type=str,
               dest='prd',
               default=None,
               help="list of start- and end-year for analysis")

P.add_argument("--fac",
               type=str,
               dest='fac',
               default=None,
               help="factor to make unit of [mm/day]")

P.add_argument("--nperseg",
               type=str,
               dest='nperseg',
               default=None,
               help="length of segment in power spectra")


args = P.get_parameter()
mip = args.mip
var = args.var
dir = args.dir
prd = args.prd
fac = args.fac
nperseg = args.nperseg

print(dir)
print(prd)
print(nperseg)

#==================================================================================
def Regrid2deg(d):
    """
    Regrid to 2deg (180lon*90lat) horizontal resolution
    Input
    - d: cdms variable
    Output
    - drg: cdms variable with 2deg horizontal resolution
    """
    # Regridding
    tgrid = cdms.createUniformGrid(-89, 90, 2.0, 0, 180, 2.0, order='yx')
    orig_grid = d.getGrid()
    regridFunc = Regridder(orig_grid,tgrid)
    drg=MV.zeros((d.shape[0], tgrid.shape[0], tgrid.shape[1]), MV.float)
    for it in range(d.shape[0]):
        drg[it] = regridFunc(d[it])
        
    # Dimension information
    time = d.getAxis(0)
    lat = tgrid.getAxis(0)
    lon = tgrid.getAxis(1)
    drg.setAxisList((time,lat,lon))
    
    # Missing value (In case, missing value is changed after regridding)
    drg[drg>=d.missing_value]=d.missing_value
    mask=np.array(drg==d.missing_value)
    drg.mask = mask
    
    print('Complete regridding from', d.shape, 'to', drg.shape)
    return drg
#==================================================================================

#==================================================================================
def ClimAnom3hr(d,syr,eyr):
    """
    Calculate climatoloty and anomaly with 3-hr data
    Input
    - d: cdms variable with 3-hr frequency
    - syr: analysis start year
    - eyr: analysis end year
    Output
    - clim: climatology with 3-hr frequency (climatological diurnal and annual cycles)
    - anom: anomaly departure from the climatological diurnal and annual cycles
    """
    # Year segment
    cal=d.getTime().calendar
    nyr=eyr-syr+1
    if 'gregorian' in cal:
        ndy=366
        ldy=31
        dseg = MV.zeros((nyr, ndy, 8, d.shape[1], d.shape[2]), MV.float)
        for iyr, year in enumerate(range(syr, eyr+1)):
            yrtmp=d(time=(str(year)+'-1-1 0:0:0', str(year)+'-12-'+str(ldy)+' 23:59:59'))
            yrtmpseg=MV.reshape(yrtmp,(int(yrtmp.shape[0]/8), 8, yrtmp.shape[1], yrtmp.shape[2]))
            if (yrtmpseg.shape[0]==365):
                dseg[iyr,0:59] = yrtmpseg[0:59]
                dseg[iyr,60:366] = yrtmpseg[59:365]
                dseg[iyr,59] = d.missing_value
            else:               
                dseg[iyr] = yrtmpseg
    else:
        if '360' in cal:
            ndy=360
            ldy=30
        else: #365-canlendar
            ndy=365
            ldy=31
        dseg = MV.zeros((nyr, ndy, 8, d.shape[1], d.shape[2]), MV.float)
        for iyr, year in enumerate(range(syr, eyr+1)):
            yrtmp=d(time=(str(year)+'-1-1 0:0:0', str(year)+'-12-'+str(ldy)+' 23:59:59'))
            yrtmpseg=MV.reshape(yrtmp,(int(yrtmp.shape[0]/8), 8, yrtmp.shape[1], yrtmp.shape[2]))           
            dseg[iyr] = yrtmpseg                                                           
    mask=np.array(dseg==d.missing_value)
    dseg.mask = mask
    
    # Climatology
    clim=cdutil.averager(dseg, axis=0, weights='unweighted')

    # Anomaly
    anom=MV.array([])
    if 'gregorian' in cal:
        for iyr, year in enumerate(range(syr, eyr+1)):    
            yrtmp=d(time=(str(year)+'-1-1 0:0:0', str(year)+'-12-'+str(ldy)+' 23:59:59'))
            if (yrtmp.shape[0]==365*8):
                anom=np.append(anom,(np.delete(dseg[iyr],59,axis=0)-np.delete(clim,59,axis=0)))
            else:
                anom=np.append(anom,(dseg[iyr]-clim))
    else:
        for iyr, year in enumerate(range(syr, eyr+1)):
            anom=np.append(anom,(dseg[iyr]-clim))                                
    mask=np.array(anom==d.missing_value)
    anom.mask = mask
  
    # Reahape and Dimension information
    clim=MV.reshape(clim,(ndy*8, d.shape[1], d.shape[2]))
    anom=MV.reshape(anom,(d.shape[0], d.shape[1], d.shape[2]))        
    time = d.getAxis(0)
    lat = d.getAxis(1)
    lon = d.getAxis(2)
    clim.setAxis(1,lat)
    clim.setAxis(2,lon)
    anom.setAxisList((time,lat,lon))
    
    print('Complete calculating climatology and anomaly for calendar of', cal)
    return clim, anom 
#==================================================================================

#==================================================================================
def Powerspectrum(d,nperseg):
    """
    Power spectrum (scipy.signal.welch)
    Input
    - d: cdms variable
    - nperseg: Length of each segment
    Output
    - freqs: Sample frequencies
    - psd: Power spectra
    - rn: Rednoise
    - sig95: 95% rednoise confidence level
    """
    # Fill missing date using interpolation
    dnp=np.array(d)
    # Missing value (In case, missing value is changed after np.array)
    dnp[dnp>=d.missing_value]=d.missing_value
    dnp[dnp==d.missing_value]=np.nan
    dfm=np.zeros((d.shape[0], d.shape[1], d.shape[2]), np.float)
    for iy in range(d.shape[1]):
        for ix in range(d.shape[2]):
            yp = pd.Series(dnp[:,iy,ix])
            ypfm = yp.interpolate(method='linear')
            dfm[:,iy,ix]=np.array(ypfm)

    # Calculate power spectrum
    freqs, psd = signal.welch(dfm,scaling='spectrum',nperseg=nperseg,axis=0)
    
    # Signigicance test of power spectra (from J. Lee's MOV code)
    nps=max(np.floor((dfm.shape[0]/(nperseg/2))-1),1) # Number of power spectra
    rn=np.zeros((psd.shape[0], psd.shape[1], psd.shape[2]), np.float)
    sig95=np.zeros((psd.shape[0], psd.shape[1], psd.shape[2]), np.float)
    for iy in range(psd.shape[1]):
        for ix in range(psd.shape[2]):
            r1 = np.array(lag1_autocorrelation(dfm[:,iy,ix]))
            rn[:,iy,ix] = rednoise(psd[:,iy,ix], len(freqs), r1)
            nu = 2*nps
            sig95[:,iy,ix] = RedNoiseSignificanceLevel(nu, rn[:,iy,ix])

    #print('Complete power spectra with segment of', nperseg)
    print('Complete power spectra ( nps=', nps, ')')
    return freqs, psd, rn, sig95
#==================================================================================

#==================================================================================
def lag1_autocorrelation(x):
    result = float(genutil.statistics.autocorrelation(x, lag=1)[-1])
    return result
#==================================================================================

#==================================================================================
def rednoise(VAR,NUMHAR,R1):
    """
    NOTE: THIS PROGRAM IS USED TO CALCULATE THE RED NOISE SPECTRUM OF 
          AN INPUT SPECTRUM. Modified from K. Sperber's FORTRAN code.
    Input
    - VAR    : array of spectral estimates (input)
    - NUMHAR : number of harmonics (input)
    - R1     : lag correlation coefficient (input)
    Output
    - RN     : array of null rednoise estimates (output)
    """
    WHITENOISE = sum(VAR)/float(NUMHAR)
    # CALCULATE "NULL" RED NOISE
    R1X2 = 2.*R1
    R12 = R1*R1
    TOP = 1. - R12
    BOT = 1. + R12
    RN = []
    for K in range(NUMHAR):
        RN.append(WHITENOISE*(TOP/(BOT - R1X2*float(math.cos(math.pi*K/NUMHAR)))))
    return RN
#==================================================================================

#==================================================================================
def RedNoiseSignificanceLevel(nu, rn, p=0.050):
    """
    nu is the number of degrees of freedom (2 in case of an fft). 
    Note: As per Wilks (1995, p. 351, Section 8.5.4) when calculating  
    the mean of M spectra and rednoise estimates nu will be nu*M (input)
    
    factor is the scale factor by which the rednoise must be multiplied by to 
    obtain the 95% rednoise confidence level (output)
    Note: the returned value is the reduced chi-square value
    
    95% Confidence CHI-SQUARED FOR NU DEGREES OF FREEDOM 
    """    
    factor = chi2.isf(p, nu)/nu
    siglevel = MV.multiply(rn, factor)
    return siglevel
 #==================================================================================


# Read data
file_list = sorted(glob.glob(dir+'*'))
f=[]
data=[]
for ifl in range(len(file_list)):
    f.append(cdms.open(file_list[ifl]))   
    file=file_list[ifl]
    if(mip=='obs'):
        model = file.split('/')[-1].split('.')[2]
        data.append(model)
    else:
        model = file.split('/')[-1].split('.')[2]
        ens = file.split('/')[-1].split('.')[3]
        data.append(model+'.'+ens)
print('# of data:', len(data))
print(data)

## To remove incomplete data
#idx=data.index('EC-Earth3.r13i1p1f1')
#idx=data.index('FGOALS-g3.r1i1p1f1')
#idx=data.index('KACE-1-0-G.r2i1p1f1')
#idx=data.index('KACE-1-0-G.r3i1p1f1')
#idx=data.index('MIROC6.r2i1p1f1')
idx=data.index('NESM3.r4i1p1f1')
data=data[idx+1:]
f=f[idx+1:]
print('# of file:', len(f))
print('# of data:', len(data))
print(data)
#------------------------------------------------

# Make output directory
outdir='/work/ahn6/pr/variability_across_timescales/power_spectrum/v20201113/data/'+mip
if not(os.path.isdir(outdir)):
    os.makedirs(outdir)     

# Regridding -> Anomaly -> Power spectra -> Write
syr=prd[0]
eyr=prd[1]
for id, dat in enumerate(data):
    cal = f[id][var].getTime().calendar
    if '360' in cal:
        ldy=30
    else:
        ldy=31
    do = f[id](var,time=(str(syr)+'-1-1 0:0:0',str(eyr)+'-12-'+str(ldy)+' 23:59:59'))*float(fac)
    print(dat, do.shape, cal)

    # Regridding
    drg = Regrid2deg(do)

    # Anomaly
    clim, anom = ClimAnom3hr(drg,syr,eyr)
    
    # Power spectum of total
    freqs, ps, rn, sig95 = Powerspectrum(drg,nperseg)
    # Write data (nc file)
    freqs=MV.array(freqs)
    ps=MV.array(ps)
    rn=MV.array(rn)
    sig95=MV.array(sig95)
    frq = cdms.createAxis(range(len(freqs)),id='frequency')
    lat = drg.getAxis(1)
    lon = drg.getAxis(2)
    freqs.setAxis(0,frq)
    ps.setAxisList((frq,lat,lon))
    rn.setAxisList((frq,lat,lon))
    sig95.setAxisList((frq,lat,lon))
    out = cdms.open(outdir+'/PS_'+var+'.3hr_regrid.180x90_'+dat+'.nc','w')
    out.write(freqs, id='freqs')
    out.write(ps, id='power')
    out.write(rn, id='rednoise')
    out.write(sig95, id='sig95')
    out.close()
    
    # Power spectum of anomaly
    freqs, ps, rn, sig95 = Powerspectrum(anom,nperseg)
    # Write data (nc file)
    freqs=MV.array(freqs)
    ps=MV.array(ps)
    rn=MV.array(rn)
    sig95=MV.array(sig95)
    frq = cdms.createAxis(range(len(freqs)),id='frequency')
    lat = drg.getAxis(1)
    lon = drg.getAxis(2)
    freqs.setAxis(0,frq)
    ps.setAxisList((frq,lat,lon))
    rn.setAxisList((frq,lat,lon))
    sig95.setAxisList((frq,lat,lon))
    out = cdms.open(outdir+'/PS_'+var+'.3hr_regrid.180x90_'+dat+'_unforced.nc','w')
    out.write(freqs, id='freqs')
    out.write(ps, id='power')
    out.write(rn, id='rednoise')
    out.write(sig95, id='sig95')
    out.close()
    
