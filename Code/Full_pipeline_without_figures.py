# Full Pipeline ---- Without Figures
#%load_ext autoreload
#%autoreload 2
from src import Direct_Strapdown as ds
from src import IMU_load
import numpy as np 
import matplotlib.pyplot as plt
from dataclasses import dataclass
params = {'axes.labelsize': 'x-large', 'axes.titlesize':'xx-large','xtick.labelsize':'large', 'ytick.labelsize':'large', 'legend.fontsize': 'x-large','mathtext.fontset':'stix', 'font.family':'STIXGeneral'}
plt.rcParams.update(params)
from pathlib import Path

# Set GNSS antenna -> IMU lever arm 
lever_arm = np.array([1.570, 0.170, -1.470]).reshape(-1,1)
# Set filter time coefficient 
ftc = 150

def echo(): 
    return print(".")

# ------------------ Data load -------------------------
file1 = Path("..", "data", "data_new", "DK2022", "285_rqh2_inat.txt")
file2 = Path("..", "data", "data_new", "DK2022", "285_rqh2_ppp_1Hz.txt")
file3 = Path("..", "data", "data_new", "DK2022", "285_inat.dat")
file4 = Path("..", "data", "AllFreeAir.dat")

nav = IMU_load.load_nav(file1)
gnss = IMU_load.load_gnss(file2)
imu = IMU_load.readIMAR(file3, "echo", "on")


# ------------------ Low pass filter IMU f^b -------------
filter_size = 600   # @ 2 sek, f = 300 Hz
imu_mov_mean = ds.movmean(imu.bacc3.values, filter_size)
imu_mov_mean_time = ds.movmean(imu.time.values, filter_size)

# ------------------ Derive acc from GNSS -----------------
print('Deriving accelerations from GNSS')

temp = {}
# Finite difference 
temp_gps_acc_raw, temp_time = ds.gnss_accelerations_v1(gnss.time.values, gnss.h,"difference")

temp["gps_acc"] = temp_gps_acc_raw
temp["time"]  = temp_time

# Filter signals 
temp_imu_acc = ds.but2_v2(imu_mov_mean, 3, ftc, 1/300)
temp_gps_acc = ds.but2_v2(temp["gps_acc"], 3, ftc, 1)

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
temp_gps_acc2 = ds.but2_v2(temp["gps_acc2"], 3, ftc, 1)
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
temp["imu_acc2"] = ds.but2_v2(temp["nacc"][:,2], 3, ftc, 1/300)

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

dg_corr = ds.bias_drift_corr(solution.dg, solution.time, vel_scalar[:index])

print("> Done")

# ----------------------- Write result txt file --------------------
print("Writing to result output file")





print("> Direct Strapdown method done")




