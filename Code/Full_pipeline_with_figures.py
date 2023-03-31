# Full Pipeline ---- Without Figures
#%load_ext autoreload
#%autoreload 2
import time
startTime = time.time()

from src import Direct_Strapdown as ds
from src import IMU_load
import numpy as np 
import matplotlib.pyplot as plt
from dataclasses import dataclass
params = {'axes.labelsize': 'x-large', 'axes.titlesize':'xx-large','xtick.labelsize':'large', 'ytick.labelsize':'large', 'legend.fontsize': 'x-large','mathtext.fontset':'stix', 'font.family':'STIXGeneral'}
plt.rcParams.update(params)
from pathlib import Path
import pandas as pd
import os
import scipy.io as spio
# Set GNSS antenna -> IMU lever arm 
# lever_arm = np.array([1.570, 0.170, -1.470]).reshape(-1,1) #DK2022
lever_arm = np.array([-0.345, -0.323, -0.653]).reshape(-1,1) #Roskilde16
# Set filter time coefficient 
ftc = 120
stages = 3

def echo():
    for i in range(3): 
        print(".") 
    return

# ------------------ Data load -------------------------

file1 = Path("..", "data", "data_new", "DK2022", "285_rqh2_inat.txt")
file2 = Path("..", "data", "data_new", "DK2022", "285_rqh2_ppp_1Hz.txt")
file3 = Path("..", "data", "data_new", "DK2022", "285_inat.dat")
file4 = Path("..", "data", "AllFreeAir.dat")

nav = IMU_load.load_nav(file1)
gnss = IMU_load.load_gnss(file2)
imu = IMU_load.readIMAR(file3, "echo", "on")

file = 'Coastline_val.txt'
coast = pd.read_table(file, sep = ",")


# ------------------ Derive freq of GNSS epoch -----------
freq = ds.PPP_freq(file2)


# ------------------ Low pass filter IMU f^b -------------
filter_size = 600   # @ 2 sek, f = 300 Hz
imu_mov_mean = ds.movmean(imu.bacc3.values, filter_size)
imu_mov_mean_time = ds.movmean(imu.time.values, filter_size)

# ------------------ Derive acc from GNSS -----------------
print('Deriving accelerations from GNSS')

temp = {}
# Finite difference 
temp_gps_acc_raw, temp_time = ds.gnss_accelerations_v1(gnss.time.values, gnss.h,"difference")

temp_gps_acc_raw, index = ds.cutoff_bound(temp_gps_acc_raw)
temp_time = temp_time[:index]
temp["gps_acc"] = temp_gps_acc_raw
temp["time"]  = temp_time

# Filter signals 
temp_imu_acc = ds.but2_v2(imu_mov_mean, stages, ftc, 1/300)
temp_gps_acc = ds.but2_v2(temp["gps_acc"], stages, ftc, freq)

temp["imu_acc"] = temp_imu_acc
temp["gps_acc"] = temp_gps_acc

# Interpolate 
temp["imu_acc"] = ds.interpolate_DS(imu_mov_mean_time, temp["imu_acc"], temp["time"], "linear","extrapolate")

print("> Done")


# -------------- Translate GNSS position to IMU Location ------------------
print("Translate GNSS Position to IMU Location")
nav_key = ["roll", "pitch", "yaw"]
for i in range(0, len(nav_key)): 
    gnss[nav_key[i]] = ds.interpolate_DS(nav.time, nav[nav_key[i]], gnss.time, "linear", "extrapolate")

olat, olon, oh = ds.pos_translate_v1(gnss.lat, gnss.lon, gnss.h, gnss.roll, gnss.pitch, gnss.yaw, lever_arm)
gnss["imu_lat"] = olat
gnss["imu_lon"] = olon
gnss["imu_h"] = oh

print("> Done")


# ------------- Derive Accelerations from GNSS at IMU location ----------------
print("Deriving accelerations from IMU Location")

# Finite difference
temp_gps_acc2, _= ds.gnss_accelerations_v1(gnss.time.values,gnss.imu_h, "difference")
temp["gps_acc2"], index = ds.cutoff_bound(temp_gps_acc2)
temp["time_cut"] = temp["time"][:index]
temp["imu_acc_cut"] = temp["imu_acc"][:index]

# Filter signals  
temp_gps_acc2 = ds.but2_v2(temp["gps_acc2"], stages, ftc, freq)
temp["gps_acc2"] = temp_gps_acc2

print("> Done")

# ---------- Transform IMU acc into North-East-Down (NED) Frame ------------------
print("Interpolating IMU Accelerations")
for key, value in nav.items(): 
    imu[key] = ds.interpolate_DS(nav.time, value, 
                                        imu.time, "linear", False, "extrapolate")
print("> Done")

print("Rotating IMU Accelerations")
imu_bacc = np.vstack([imu.bacc1.values, imu.bacc2.values, imu.bacc3.values]).T
imu_att = np.vstack([imu.roll.values, imu.pitch.values, imu.yaw.values]).T

# Transform Accelerations 
imu_nacc = ds.b2n_v1(imu.time, imu_bacc, imu_att)
temp["nacc"] = imu_nacc.T  # Using temp["nacc"] instead of imu.nacc as the pd datafram of imu can not take the [N x 3]

# Filter signal 
temp["imu_acc2"] = ds.but2_v2(temp["nacc"][:,2], stages, ftc, 1/300)

# Interpolate 
temp["imu_acc2"] = ds.interpolate_DS(imu.time, temp["imu_acc2"], temp["time"], "linear", False, "extrapolate")
temp["imu_acc2"] = temp["imu_acc2"][:index]
print("> Done")


# --------------- Compute Transport Rate + Eötvos and Corilis effect ---------------- 
print("Computing Transport-Rate (Eotvos and Coriolis) Effect")

vel = np.vstack([imu.vn.values, imu.ve.values, imu.vd.values]).T
pos = np.vstack([imu.lat.values, imu.lon.values, imu.h.values]).T

# Compute Transport Rate 
imu_tacc = ds.transport_rate_v2(imu.time, vel, pos)
temp["imu_tacc"] = imu_tacc

# Filter signal 
temp_trans = ds.but2_v2(temp["imu_tacc"][:,2], 3, ftc, 1/300)
temp["trans"] = temp_trans

# Interpolate 
temp["trans"] = ds.interpolate_DS(imu.time, temp["trans"], 
                               temp["time"], "linear", False, "extrapolate")
temp["trans"] = temp["trans"][:index]
# Correct for transport rate 
temp["imu_acc3"] = temp["imu_acc2"] - temp["trans"]

print("> Done")

# ---------------------- Derive Gravity ---------------------------

print("Deriving Gravity Acceleration")
@dataclass
class solution: 
    time: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    h: np.ndarray
    imu_acc: np.ndarray
    gps_acc: np.ndarray 
    gamma: np.ndarray

nav_key2 = ["lat", "lon", "h"]
for i in range(0, len(nav_key)): 
    temp[nav_key2[i]] = ds.interpolate_DS(nav.time, nav[nav_key2[i]], 
                                          temp["time"], "linear", "extrapolate")


solution = solution(temp["time_cut"],temp["lat"][:index],temp["lon"][:index],temp["h"][:index],
                    temp["imu_acc3"],temp["gps_acc2"], [])

gamma, _, _ = ds.normal_gravity_precise_v1(solution.lat.reshape(-1,1),solution.lon.reshape(-1,1),solution.h.reshape(-1,1), 3)
#gamma, _, _ = normal_gravity_precise_v1(solution.lat.reshape(-1,1),solution.lon.reshape(-1,1),solution.h.reshape(-1,1), 3)

solution.gamma = gamma.down

# Derive Gravity 
solution.g = solution.gps_acc - solution.imu_acc
solution.dg = (solution.g.reshape(-1,1) - solution.gamma)*10**5

print("> Done")

# ------------------------- Bias and Drift Correction ---------------
print("Computing Bias and Drift correction")
vel_dic = {}
vn_gnss = ds.interpolate_DS(nav.time, nav.vn, temp["time"], "linear", "extrapolate")
nav_key = ["vn", "ve", "vd"]
for i in range(0, len(nav_key)): 
    vel_dic[nav_key[i]] = ds.interpolate_DS(nav.time, nav[nav_key[i]], temp["time"],
                                             "linear", "extrapolate")

vel_scalar = np.round(np.sqrt(vel_dic["vn"]**2 + vel_dic["ve"]**2 + vel_dic["vd"]**2))

solution.dg, index = ds.cutoff_bound(solution.dg.reshape(1,-1))
solution.time = solution.time[:index]
solution.dg = solution.dg.reshape(-1,1)[:index]
dg_corr = ds.bias_drift_corr(solution.dg, solution.time, vel_scalar[:index])

print("> Done")

# ----------------------- Write result txt file --------------------
print("Writing to result output file")
# Insert code. 
input_path = file3

# Extract the input filename and path
input_filename = os.path.basename(input_path)
input_dir = os.path.dirname(input_path)

# Extract the number from the input filename
number = input_filename.split('_')[0]

# Construct the output filename
output_filename = f"{number}_results.txt"
output_path = os.path.join(input_dir, output_filename)

FlightID = np.ones(len(solution.dg)) * int(number)
df = pd.DataFrame({"FlightID":FlightID.reshape(-1)[:index],
                    "Latitude":solution.lat.reshape(-1)[:index],
                      "Longitude":solution.lon.reshape(-1)[:index],
                        "H-Ell":solution.h.reshape(-1)[:index],
                          "dg":solution.dg.reshape(-1)[:index],
                            "dg_corr":dg_corr.reshape(-1)[:index],
                              "Time":solution.time.reshape(-1)[:index]})

df.to_csv(output_path, sep="\t", index=False)

print(f"> Result file: {number}_results.txt created at: ", output_path)

# --------------------- Print results to figures -------------------
print("Printing figures")

# --------- Extracting name of survey for figure naming
# Set path to output file, in parent folder.
filename = file3
path = Path("Figures", "catch_file.txt")

# Extract the input filename and path
input_filename = os.path.basename(filename)
input_dir = os.path.dirname(path)

# Extract the number from the input filename
number = input_filename.split('_')[0]
 
def output_name(name, number, input_dir): 
    output_filename = f"{number}_{name}.pdf"
    output_path = os.path.join(input_dir, output_filename)
    return output_path


N = 6
import matplotlib.gridspec as gridspec

## ------------ Survey overview 
fig = plt.figure(figsize=(N, N-1))
fig.suptitle("Survey Area Overview", fontsize= 22)
# create a grid with 3 rows and 2 columns
gs = gridspec.GridSpec(3, 2, figure=fig)

# add the first three subplots
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])

# add the fourth subplot
ax4 = fig.add_subplot(gs[0:, 1])

#fig = plt.figure(figsize=figsize)
ax4.plot(coast.X, coast.Y, color="black", linewidth=1)
ax4.plot(gnss.lon, gnss.lat, color="red", linewidth=1)
ax4.scatter(gnss.lon[0], gnss.lat[0], 50, label="Start/End", marker='^')
ax4.legend(loc = "upper right")
ax4.grid()
ax4.set_xlabel('Longitude', fontsize=16)
ax4.set_ylabel('Latitude', fontsize=16)
# ax4.set_title(r'Coastline')
ax4.set_xlim(7, 13);
ax4.set_ylim(54.5, 58.5);
ax4.set_title("d) Flight Path", fontsize=18)


ax1.plot(gnss.time, gnss.lat, color="red", linewidth=2)
ax1.grid()
# ax1.set_xlabel('GNSSS SOW [s]', fontsize=10)
ax1.set_ylabel(r'Lat [$^{\circ}$]', fontsize=16)
ax1.set_xlim(gnss.time.values[0], gnss.time.values[-1]);
ax1.set_title("a) Latitude", fontsize=18)
ax1.autoscale(tight=True)

ax2.plot(gnss.time, gnss.lon, color="red", linewidth=2)
ax2.grid()
# ax2.set_xlabel('GNSSS SOW [s]', fontsize=10)
ax2.set_ylabel('Lon [$^{\circ}$]', fontsize=16)
ax2.set_xlim(gnss.time.values[0], gnss.time.values[-1]);
ax2.set_title("b) Longitude", fontsize=18)

ax3.plot(gnss.time, gnss.h, color="red", linewidth=2)
ax3.grid()
ax3.set_xlabel('GNSSS SOW [s]', fontsize=16)
ax3.set_ylabel('Height [m]', fontsize=16)
ax3.set_xlim(gnss.time.values[0], gnss.time.values[-1]);
ax3.set_title("c) Height", fontsize=18)

plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=0.6, hspace=0.9)

plt.tight_layout()
plt.savefig(output_name("Survey_overview", number, input_dir))


# ------------- gps acc antenna pos 
fig, ax = plt.subplots(2,1, figsize = (N*2,N))
ax[0].plot(temp["time"], temp["imu_acc"]-np.mean(temp["imu_acc"]), label="imu (body)")
ax[0].plot(temp["time"], temp["gps_acc"], label="gps (antenna)")
ax[0].set_xlim(temp["time"][0], temp["time"][-1])
ax[0].set_ylim(-0.1, 0.1)
ax[0].set_ylabel(r"Accelerations [$m/s^{2}$]")
# ax[0].set_xlabel('Temp time SOW [s]')
ax[0].grid()
ax[0].legend(loc="lower left")

ax[1].plot(temp["time"], (temp["gps_acc"] - temp["imu_acc"])*10**5)
ax[1].grid()
ax[1].set_xlim(temp["time"][0], temp["time"][-1])
ax[1].set_ylabel("gps - imu [mGal]")
ax[1].set_xlabel('Time SOW [s]')
fig.suptitle(r"$\mathbf{\ddot{r}}^n_{GNSS}$ derived at GNSS receiver position", fontsize=20)
plt.savefig(output_name("GPS_acc_1", number, input_dir))


# --------------- gps acc IMU pos
fig, ax = plt.subplots(2,1, figsize = (N*2,N))
ax[0].plot(temp["time"], temp["imu_acc"]-np.mean(temp["imu_acc"]), label="imu (body)")
ax[0].plot(temp["time"], temp["gps_acc"], label="gps (antenna location)")
ax[0].plot(temp["time_cut"], temp["gps_acc2"], label="gps (imu location)")
ax[0].set_xlim(temp["time"][0], temp["time"][-1])
ax[0].set_ylim(-0.1, 0.1)
ax[0].set_ylabel(r"Accelerations [$m/s^{2}$]")
ax[0].grid()
ax[0].legend(loc = "lower left")

ax[1].plot(temp["time_cut"], (temp["gps_acc2"] - temp["imu_acc_cut"])*10**5)
ax[1].grid()
ax[1].set_xlim(temp["time_cut"][0], temp["time_cut"][-1])
ax[1].set_ylabel("gps - imu [mGal]")
ax[1].set_xlabel("Time SOW [s]")
fig.suptitle(r"$\mathbf{\ddot{r}}^n_{GNSS}$ derived at IMU position", fontsize=20)
plt.savefig(output_name("GPS_acc_2", number, input_dir))


# ----------- acc at IMU loc NED frame
fig, ax = plt.subplots(2,1, figsize = (N*2,N))
ax[0].plot(temp["time"], temp["imu_acc"]-np.mean(temp["imu_acc"]), label="imu (body)")
ax[0].plot(temp["time"], temp["gps_acc"], label="gps (antenna location)")
ax[0].plot(temp["time_cut"], temp["gps_acc2"], label="gps (imu location)")
ax[0].plot(temp["time_cut"], temp["imu_acc2"]-np.mean(temp["imu_acc2"]), label="imu (down)")
ax[0].set_xlim(temp["time"][0], temp["time"][-1])
ax[0].set_ylabel(r"Accelerations [$m/s^{2}$]")
ax[0].grid()
ax[0].legend(loc="lower left")
ax[0].set_ylim(-0.1, 0.1)


ax[1].plot(temp["time_cut"], (temp["gps_acc2"] - temp["imu_acc2"])*10**5)
ax[1].grid()
ax[1].set_xlim(temp["time_cut"][0], temp["time_cut"][-1])
ax[1].set_ylabel("gps - imu [mGal]")
ax[1].set_xlabel("Time SOW [s]")
fig.suptitle(r"$\mathbf{\ddot{r}}^n_{IMU, D}$, NED Frame", fontsize=20)
plt.savefig(output_name("imu_acc2", number, input_dir))


# -------------- Transport rate 
fig, ax = plt.subplots(2,1, figsize = (N*2,N))
ax[0].plot(temp["time"], temp["imu_acc"]-np.mean(temp["imu_acc"]), label="imu (body)")
ax[0].plot(temp["time"], temp["gps_acc"], label="gps (antenna location)")
ax[0].plot(temp["time_cut"], temp["gps_acc2"], label="gps (imu location)")
ax[0].plot(temp["time_cut"], temp["imu_acc2"]-np.mean(temp["imu_acc2"]), label="imu (down)")
ax[0].plot(temp["time_cut"], temp["trans"], label="Trans-rate")
ax[0].plot(temp["time_cut"], temp["imu_acc3"]-np.mean(temp["imu_acc3"]), label="imu-trans")
ax[0].set_xlim(temp["time"][0], temp["time"][-1])
ax[0].set_ylabel(r"Accelerations [$m/s^{2}$]")
ax[0].grid()
ax[0].set_ylim(-0.1, 0.05)
ax[0].legend(loc = "lower left")


ax[1].plot(temp["time_cut"], (temp["gps_acc2"] - temp["imu_acc3"])*10**5)
ax[1].grid()
ax[1].set_xlim(temp["time_cut"][0], temp["time_cut"][-1])
ax[1].set_ylabel("gps - imu [mGal]")
ax[1].set_xlabel("Time SOW [s]")
fig.suptitle(r"Transport Rate inclusion", fontsize=20)
plt.savefig(output_name("trans", number, input_dir))


# -------------- Correlated Gravity 
# Plot results...

fig = plt.figure(figsize=(N*2, N))
plt.plot(solution.time, solution.dg, label="Airborne", color="red")
plt.plot(solution.time, dg_corr, label="Bias and Drift correlated")
plt.ylabel(r"Gravity disturbance [mGal]")
plt.xlabel("Second of week [s]")
plt.grid()
plt.legend(loc="lower left")
# plt.xlim(285000, 302500)

plt.savefig(output_name("dg", number, input_dir))

print("> Done")

echo()

print("> Direct Strapdown method done")
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))


