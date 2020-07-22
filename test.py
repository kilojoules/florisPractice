import matplotlib
matplotlib.use('tkagg')
import floris.tools as wfct
# Initialize the FLORIS interface fi
fi = wfct.floris_interface.FlorisInterface("./example_input.json")
# Set to 2x2 farm
fi.reinitialize_flow_field(layout_array=[[0,0,600,600],[0,300,0,300]])
# Calculate wake
fi.calculate_wake()

