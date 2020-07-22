import floris.tools as wfct
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
import numpy as np

def florisEval(XY, WS, DIR):

   # Initialize the FLORIS interface fi
   fi = wfct.floris_interface.FlorisInterface("./example_input.json")

   # Set to 2x2 farm
   fi.reinitialize_flow_field(layout_array=XY,
                           wind_speed=WS, wind_direction=DIR)

   # Calculate wake
   fi.calculate_wake()

   return(fi.get_farm_power())

plt.close('all')
speeds = np.linspace(5, 15, 5)
pows = []
XY = np.array([[0,0,600,600],[0,300,0,300]])
for speed in speeds:
   pows.append(florisEval(XY, speed, 0))

plt.plot(speeds, pows)
plt.savefig('hey')
