import floris.tools as wfct
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('tkagg')
import numpy as np

def florisEval(yaw_angle, WS, DIR, lf=False):

   # Initialize the FLORIS interface fi
   if lf:
      fi = wfct.floris_interface.FlorisInterface("./twoexample_inputLF.json")
   else:
      fi = wfct.floris_interface.FlorisInterface("./twoexample_input.json")

   # Set to 2x2 farm
   fi.reinitialize_flow_field(#layout_array=XY,
                           wind_speed=WS, wind_direction=DIR)

   # Calculate wake
   fi.calculate_wake(yaw_angles=np.array([0, yaw_angle]))

   return(-1 * fi.get_farm_power() / 1e6)

def f(x, lf=False):
   return florisEval(x, WS=[7], DIR=0, lf=lf)

if __name__ == '__main__':
   plt.close('all')
   speeds = np.linspace(5, 15, 5)
   pows = []
   XY = np.array([[0,0,600,600],[0,300,0,300]])
   for speed in speeds:
      pows.append(florisEval(XY, speed, 0))

   #plt.plot(speeds, pows)
   #plt.savefig('hey')