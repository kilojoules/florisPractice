import numpy as np
def patternSearch(f, x0, bounds, deltaX=1, writeEvals=False, args={}):
   bnds = np.array(bounds).T
   NEvals = 0
   evals = []
   for _ in range(x0.size * 100):
      f0 = f(x0, **args)
      evals.append(f0)
      NEvals += 1
      changed = False
      print(_, x0, deltaX)
      for ii in range(x0.size):
         dx = np.zeros(x0.size)
         dx[ii] = deltaX
         xl = np.max([x0 - dx, bnds[0, :]], 0)
         xh = np.min([x0 + dx, bnds[1, :]], 0)
         fl = f(xl, **args)
         fh = f(xh, **args)
         NEvals += 2
         if fl < f0:
            x0 = xl
            f0 = fl
            changed = True
         evals.append(f0)
         if fh < f0:
            x0 = xh
            f0 = fh
            changed = True
         evals.append(f0)

      if changed == False:
         deltaX /= 2

      if deltaX < 1e-2: break

   if writeEvals: np.save('theseEvals', evals)
   
   print('number of evals ', NEvals)
   return {'x': x0, 'f': f0}        
