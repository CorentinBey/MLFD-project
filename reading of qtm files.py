"""
This file serves as an exploration tool to find the segment of interest for the different drone flights.
It can also be used for comparing the data coming from the barometer with the one from Qualisys (motion capture system).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
import os


#%% Functions

def get_data(path):
    """
    Load TSV file data into a NumPy array.

    Args:
        path (str): Path to the TSV file.

    Returns:
        np.ndarray: Loaded data from the file.
    """
    df = pd.read_csv(path, sep='\t', skiprows = 12, header=None)
    data = df.to_numpy()
    return data


def plotdata(data,title,index):
    """
    Plot 3D trajectory data for multiple markers.

    Args:
        data (np.ndarray): Data array containing the markers' positions.
        title (str): Title of the plot.
        index (list[int]): List containing [start, end] indices for plotting.
    """
    plt.figure()
    ax = plt.axes(projection = '3d')
    ax.set_title(title)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    
    for i in range(20):
        ax.plot(data[index[0]:index[1],(i*3)+2],
                data[index[0]:index[1],(i*3)+3],
                data[index[0]:index[1],(i*3)+4], label = f"index: {i}")
        #print(f"({np.max(data[index[0]:index[1],(i*3)+2])},{np.min(data[index[0]:index[1],(i*3)+2])}), ({np.max(data[index[0]:index[1],(i*3)+3])},{np.min(data[index[0]:index[1],(i*3)+3])}), ({np.max(data[index[0]:index[1],(i*3)+4])},{np.min(data[index[0]:index[1],(i*3)+4])})")
    ax.legend()
    plt.show()
    
def plot_sigle_line(data,i):
    """
    Plot a single marker's 3D trajectory.

    Args:
        data (np.ndarray): Data array containing the markers' positions.
        i (int): Marker index to plot (0-based).
    """
    plt.figure()
    ax = plt.axes(projection = '3d')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.plot(data[:,(i*3)+2],data[:,(i*3)+3],data[:,(i*3)+4])
    
def plot_sigle_line_time_segment(data,i,time_seg):
    """
    Plot a single marker's 3D trajectory over a specific time segment.

    Args:
        data (np.ndarray): Data array containing the markers' positions.
        i (int): Marker index to plot (0-based).
        time_seg (list[int]): [start, end] indices for the time segment.
    """
    plt.figure()
    ax = plt.axes(projection = '3d')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.plot(data[time_seg[0]:time_seg[1], (i*3)+2], 
            data[time_seg[0]:time_seg[1], (i*3)+3], 
            data[time_seg[0]:time_seg[1], (i*3)+4])
    
def plot_alt_time(data,index):
    """
    Plot altitude (or any other scalar) over time.

    Args:
        data (np.ndarray): Data array containing the data.
        index (int): Column index corresponding to the scalar to plot.
    """
    plt.figure()
    plt.plot(data[:,1],data[:,index]/1000)
    #plt.xlabel("Time (s)")
    #plt.ylabel("Alt (m)")
    
#%% Hover Flights Data Loading and Plotting

# Define paths for hover flights
hov_path1 = r'D:\VKI\Project\Test flight\20250409\Flights\export-tsv\20250904-1543-01-hover.tsv'
hov_path2 = r'D:\VKI\Project\Test flight\20250409\Flights\export-tsv\20250904-1547-02-hover.tsv'
hov_path3 = r'D:\VKI\Project\Test flight\20250409\Flights\export-tsv\20250904-1556-03-hover.tsv'
hov_path4 = r'D:\VKI\Project\Test flight\20250409\Flights\export-tsv\20250904-1558-04-hover.tsv'
hov_path5 = r'D:\VKI\Project\Test flight\20250409\Flights\export-tsv\20250904-1559-05-hover.tsv'

# Load hover data
data_hov1 = get_data(hov_path1)
data_hov2 = get_data(hov_path2)
data_hov3 = get_data(hov_path3)
data_hov4 = get_data(hov_path4)
data_hov5 = get_data(hov_path5)

# Plot hover data
plotdata(data_hov1, "Hover 1", [0,len(data_hov1)])
plotdata(data_hov2, "Hover 2", [0,len(data_hov2)])
plotdata(data_hov3, "Hover 3", [0,len(data_hov3)])
plotdata(data_hov4, "Hover 4", [0,len(data_hov4)])
plotdata(data_hov5, "Hover 5", [0,len(data_hov5)])

# Focused segment for Hover 4
hov4_ascent_part = pd.DataFrame(data_hov4[1175:1500,[1,23,24,25]])

# Select representative points
n_points = 10
indices = np.linspace(0, len(hov4_ascent_part) - 1, n_points).astype(int)
df_selected = hov4_ascent_part.iloc[indices]


# =============================================================================
# plt.figure()
# plt.plot(hov4_ascent_part[0],hov4_ascent_part[3]/1000)
# =============================================================================
# Plot specific hover 4 marker
plot_sigle_line(data_hov4, 7)
plot_sigle_line_time_segment(data_hov4, 7, [1175,1500])
plot_alt_time(data_hov4, 25)
#%% Ascent-Descent Flights Data Loading and Plotting

# Define paths
path1 = r'D:\VKI\Project\Test flight\20250409\Flights\export-tsv\20250904-1604-01-ascdsc.tsv'
path2 = r'D:\VKI\Project\Test flight\20250409\Flights\export-tsv\20250904-1606-02-ascdsc.tsv'
path3 = r'D:\VKI\Project\Test flight\20250409\Flights\export-tsv\20250904-1608-03-ascdsc.tsv'
path4 = r'D:\VKI\Project\Test flight\20250409\Flights\export-tsv\20250904-1609-04-ascdsc.tsv'
path5 = r'D:\VKI\Project\Test flight\20250409\Flights\export-tsv\20250904-1611-05-ascdsc.tsv'

# Load ascent-descent data
ascdsc1 = get_data(path1)
ascdsc2 = get_data(path2)
ascdsc3 = get_data(path3)
ascdsc4 = get_data(path4)
ascdsc5 = get_data(path5)

# Plot ascent-descent data
plotdata(ascdsc1, "Ascent-Descent 1", [0,len(ascdsc1)])
plotdata(ascdsc2, "Ascent-Descent 2", [0,len(ascdsc2)])
plotdata(ascdsc3, "Ascent-Descent 3", [0,len(ascdsc3)])
plotdata(ascdsc4, "Ascent-Descent 4", [0,len(ascdsc4)])
plotdata(ascdsc5, "Ascent-Descent 5", [0,len(ascdsc5)])

#%%Barometer vs Qualisys Comparison

# Load barometric data
path = r"D:\VKI\Project\Test flight\20250409\Flights\hover\drone\20250409-1558-04-hover_BARO_Alt.csv"
df = pd.read_csv(path, names = ['time', 'barometer'])

shift = df['time'].iloc[0]-2
index = 25
data = data_hov4

# Thin and smooth barometer data
baro_thinned = df[::2].reset_index(drop=True)
baro_thinned['baro_smooth'] = savgol_filter(baro_thinned['barometer'], window_length=40, polyorder=3)


# Plot barometer vs Qualisys
plt.figure(figsize=(10, 5))
#plt.scatter(df['time'], df['barometer'], alpha=0.5, label='Raw')
plt.plot(baro_thinned['time'][78:106], baro_thinned['baro_smooth'][78:106], color='red', label='Smoothed')
#plt.scatter(df['time'], df['barometer'], alpha=0.4)
plt.plot(data[1175:1500,1]+shift,data[1175:1500,index]/1000, label = 'Qualisys')
plt.xlabel("Time (s)")
plt.ylabel("Barometric Altitude (m)")
plt.title("Barometer Altitude During Flight")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#%% Compute displacement from IMU


# Load the uploaded acceleration data
file_path = r"D:\VKI\Project\Test flight\20250409\Flights\hover\drone\20250409-1547-02-hover_IMU_AccX.csv"
acc_data = pd.read_csv(file_path)

# Rename columns for clarity
acc_data.columns = ['Time', 'Acceleration']

# Convert time to seconds relative to the first timestamp
acc_data['Time'] = acc_data['Time'] - acc_data['Time'].iloc[0]

# Integrate acceleration to get velocity using cumulative trapezoidal integration
velocity = (acc_data['Acceleration'].rolling(2).mean() * acc_data['Time'].diff()).fillna(0).cumsum()

# Integrate velocity to get displacement
displacement = (velocity.rolling(2).mean() * acc_data['Time'].diff()).fillna(0).cumsum()

# Combine into a single DataFrame for display
result_df = acc_data.copy()
result_df['Velocity'] = velocity
result_df['Displacement'] = displacement

plt.figure()
plt.plot(result_df['Displacement'])



#%% RPM Data Plotting


def load_rpm_data(paths):
    """
    Load RPM CSV files.

    Args:
        paths (list[str]): List of file paths to RPM CSV files.

    Returns:
        dict: Dictionary of DataFrames, key = motor label, value = data.
    """
    data = {}
    for path in paths:
        filename = os.path.basename(path)
        label = filename.split("-hover_")[-1].replace(".csv", "")
        
        df = pd.read_csv(path)
        data[label] = df
    return data

def plot_rpm_data(data_dict):
    """
    Plot motor RPM data over time.

    Args:
        data_dict (dict): Dictionary of DataFrames with 'Time' and 'RPM' columns.
    """
    plt.figure(figsize=(12, 8))

    for label, df in data_dict.items():
        if 'Time' in df.columns and 'RPM' in df.columns:
            plt.plot(df['Time'], df['RPM'], label=label)
        else:
            print(f"Warning: {label} missing 'Time' or 'RPM' columns.")

    plt.xlabel('Time [s]')
    plt.ylabel('RPM')
    plt.title('Motor RPM vs Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    folder = r"D:\VKI\Project\Test flight\20250409\Flights\hover\drone\20250409-1558-04-hover_"  # Change this path if needed
    patterns = np.array(["RPM_aft_left", "RPM_aft_right", "RPM_front_left", "RPM_front_right"])
    file_paths = [folder + pattern + ".csv" for pattern in patterns]
    data = load_rpm_data(file_paths)
    plot_rpm_data(data)
