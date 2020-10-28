import floris.tools as wfct
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('tkagg')
import numpy as np

def florisEval(yaw_angle, WS, DIR, lf=False, MD=False):

   #if lf: return -10 - np.sum(np.deg2rad(np.abs(yaw_angle)) - np.deg2rad(yaw_angle) ** 2)
   #if lf: return -10 - np.sum(np.abs(np.sin(np.deg2rad(yaw_angle)))) ** 1.5
   #if lf: return -10 - np.sum(np.abs(np.sin(np.deg2rad(yaw_angle))))
   #if lf: return np.array([-10 - np.sum(np.sin(np.deg2rad(yaw_angle)) ** 2)])

   # Initialize the FLORIS interface fi
   if not MD: 
      if lf:
         fi = wfct.floris_interface.FlorisInterface("./twoexample_inputLF.json")
      else:
         fi = wfct.floris_interface.FlorisInterface("./twoexample_input.json")
   else:
      if lf:
         fi = wfct.floris_interface.FlorisInterface("./twoexample_inputLFMD.json")
      else:
         fi = wfct.floris_interface.FlorisInterface("./twoexample_inputMD.json")

   # Set to 2x2 farm
   fi.reinitialize_flow_field(#layout_array=XY,
                           wind_speed=WS, wind_direction=DIR)

   # Calculate wake
   if MD:
      fi.calculate_wake(yaw_angles=np.array([0] + list(yaw_angle[0])))
   else:
      fi.calculate_wake(yaw_angles=np.array([0, yaw_angle]))

   return(-1 * fi.get_farm_power() / 1e6)

def f(x, lf=False, MD=False):
   #if lf == True:
   #   return np.sin(np.pi + x / 10) - 3
   #else:
   #   return np.sin(np.pi + x / 10 - 1) - 3  + 0.2 * np.cos(x)
   return florisEval(x, WS=[7], DIR=0, lf=lf, MD=MD)

def g(x, lf=False): 
   if lf:
      #return 0
      #return np.sum(np.sin(np.deg2rad(x - 10)) ** 2 + 0.5 * np.sin(np.deg2rad(x + 12)) ** 4) - 10
      return np.sum(np.sin(np.deg2rad(x))) ** 2 - 10
   else:
      return np.sum(np.sin(np.deg2rad(x)) ** 2 + 0.5 * np.sin(np.deg2rad(x)) ** 4) - 10

if __name__ == '__main__':
   plt.close('all')
   speeds = np.linspace(5, 15, 5)
   pows = []
   XY = np.array([[0,0,600,600],[0,300,0,300]])
   for speed in speeds:
      pows.append(florisEval(XY, speed, 0))

   #plt.plot(speeds, pows)
   #plt.savefig('hey')
