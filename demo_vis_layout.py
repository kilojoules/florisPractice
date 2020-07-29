# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

import numpy as np
import matplotlib.pyplot as plt

import floris.tools as wfct


plt.style.use('dark_background')


# Initialize the FLORIS interface fi
fi = wfct.floris_interface.FlorisInterface("./twoexample_input.json")

# Make a random 9-turbine layout
DIR = [0]
WS = 7
fi.reinitialize_flow_field(wind_speed=WS, wind_direction=DIR)
fi.calculate_wake(yaw_angles=np.array([0, 30]))

# Show layout visualizations
fig, axarr = plt.subplots()

ax = axarr
hor_plane = fi.get_hor_plane()
c = wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
plt.colorbar(c, label='Wind Speed (m/s)')
plt.xlim(-200, 200)
plt.ylim(-250, 700)
plt.xlabel('Position (m)')
plt.ylabel('Position (m)')

plt.savefig('twoTurbs')
plt.clf()
