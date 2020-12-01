mip = 'cmip5'
exp = 'historical'
var = 'pr'
frq = '3hr'
#ver = 'v20201024'
ver = 'v20201107'
dir = '/p/user_pub/pmp/pmp_results/pmp_v1.1.2/additional_xmls/latest/'+ver+'/'+mip+'/'+exp+'/atmos/'+frq+'/'+var+'/'
prd = [1985, 2004] # analysis period
fac = 86400 # factor to make unit of [mm/day]
nperseg = 10*365*8 # length of segment in power spectra (~10yrs)

