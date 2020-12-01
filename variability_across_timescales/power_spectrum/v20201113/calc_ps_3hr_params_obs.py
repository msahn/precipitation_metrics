import os

mip = 'obs'
dat = 'TRMM-3B43v-7'
var = 'pr'
frq = '3hr'
ver = 'v20200707'

xmldir = './xml_obs/'
if not(os.path.isdir(xmldir)):
    os.makedirs(xmldir)

path = '/p/user_pub/PCMDIobs/PCMDIobs2/atmos/'+frq+'/'+var+'/'+dat+'/gn/'+ver+'/'
nc = var+'_'+frq+'_'+dat+'_BE_gn_'+ver+'_*.nc'
os.system('cdscan -x '+xmldir+var+'.'+frq+'.'+dat+'.xml '+path+nc)

dir = xmldir
prd = [1998, 2013] # analysis period
fac = 24 # factor to make unit of [mm/day]
nperseg = 10*365*8 # length of segment in power spectra (~10yrs)

