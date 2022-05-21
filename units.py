######################################################################
#                            AREPO units                             #
######################################################################

#Base units
ulength = 1.0000000e+17
umass = 1.991e33
uvel = 36447.268

#Derived units
utime = ulength/uvel
udensity = umass/ulength/ulength/ulength
uenergy= umass*uvel*uvel
ucolumn = umass/ulength/ulength

#More derived units
uparsec=ulength/3.0856e18

######################################################################
#                      Fundamental constants                         #
######################################################################

mp = 1.6726231e-24
kb = 1.3806485e-16