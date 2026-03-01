import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import weibull_min
import os

# =========================================================================
# USER SETTINGS
# =========================================================================
# Pas hier de bestandsnamen aan indien nodig
NASA_CSV_FILE = 'Windspeed_direction.csv'
# Let op: als je het originele Excel bestand hebt, gebruik dan pd.read_excel in de code hieronder.
# Hier gaan we uit van de CSV versie zoals geüpload.
LOAD_FILE = 'scaling2023.2.csv' 

# =========================================================================
# 1. TURBINE SPECIFICATIONS (Vestas V117-4.2 MW)
# =========================================================================
n_turbines = 2
P_rated_1 = 4.2  # Rated power per turbine [MW]
P_rated = n_turbines * P_rated_1  # Total: 8.4 MW

v_cutin = 3.0    # Cut-in wind speed [m/s]
v_rated = 13.0   # Rated wind speed [m/s]
v_cutout = 25.0  # Cut-out wind speed [m/s]
hub_height = 117 # Hub height [m]

# =========================================================================
# 2. HOUSEHOLD DEMAND PARAMETERS
# =========================================================================
n_households = 10869
kWh_per_hh = 3000  # [kWh/yr/household]
Annual_Load_MWh = n_households * kWh_per_hh / 1000  # 32,607 MWh
Avg_Load_MW = Annual_Load_MWh / 8760  # ~3.72 MW

# =========================================================================
# 3. LOAD DEMAND PROFILE
# =========================================================================
# =========================================================================
# 3. LOAD DEMAND PROFILE
# =========================================================================
print(f"Loading demand profile from {LOAD_FILE} ...")

try:
    # OPTIE 1: Probeer met komma als decimaal (Europees formaat) en punt-komma als separator
    df_load = pd.read_csv(LOAD_FILE, sep=';', decimal=',', usecols=["Time (s)", "Load Harvestehude (MWh)"])
except ValueError:
    # OPTIE 2: Probeer standaard (komma als separator, punt als decimaal)
    # Soms leest hij het wel in, maar ziet hij getallen als strings door de komma's.
    df_load = pd.read_csv(LOAD_FILE, usecols=["Time (s)", "Load Harvestehude (MWh)"])

# VEILIGHEIDSCHECK: Forceer de kolommen naar numerieke waarden
# Als er ergens '3,5' staat terwijl pandas een punt verwacht, fixen we dat hier door komma's te vervangen
if df_load["Load Harvestehude (MWh)"].dtype == object:
    df_load["Load Harvestehude (MWh)"] = df_load["Load Harvestehude (MWh)"].astype(str).str.replace(',', '.')

# Zet om naar nummers, foutieve waarden (tekst) worden NaN
df_load["Load Harvestehude (MWh)"] = pd.to_numeric(df_load["Load Harvestehude (MWh)"], errors='coerce')
df_load["Time (s)"] = pd.to_numeric(df_load["Time (s)"], errors='coerce')

# Verwijder rijen die nu NaN zijn geworden (door bijv. slechte data)
df_load = df_load.dropna()

# Data toewijzen
time_load_s = df_load["Time (s)"].values
load_MWh_step = df_load["Load Harvestehude (MWh)"].values

# Tijdstappen berekenen
dt_load_s = time_load_s[1] - time_load_s[0]
dt_load_h = dt_load_s / 3600
N_load = len(time_load_s)

# Omrekenen van Energie (MWh) naar Vermogen (MW)
# Power [MW] = Energy [MWh] / Time [h]
P_demand_MW = load_MWh_step / dt_load_h

time_load_h = time_load_s / 3600
time_load_d = time_load_s / 86400

print(f"  Timestep: {dt_load_s:.0f} s | Points: {N_load} | Duration: {N_load*dt_load_h/24:.1f} days")
print(f"  Annual demand (check): {np.sum(P_demand_MW)*dt_load_h:.0f} MWh")
# =========================================================================
# 4. LOAD WIND SPEED DATA
# =========================================================================
print(f"\nLoading NASA POWER wind data: {NASA_CSV_FILE} ...")

# NASA POWER CSV header overslaan (skiprows=10)
try:
    df_nasa = pd.read_csv(NASA_CSV_FILE, skiprows=10)
except Exception as e:
    print(f"Error reading wind file: {e}")
    raise

# Kolomnamen opschonen (spaties verwijderen)
df_nasa.columns = [c.strip() for c in df_nasa.columns]

if 'WS50M' not in df_nasa.columns:
    raise KeyError("WS50M column not found. Check CSV format.")

v_50m = df_nasa['WS50M'].values

# Replace fill values (-999) with NaN, then interpolate
v_50m = np.where(v_50m < 0, np.nan, v_50m)
# Pandas interpolate is handig voor NaN invullen
v_50m = pd.Series(v_50m).interpolate(method='linear').values

N_nasa = len(v_50m)
dt_nasa_h = 8760 / N_nasa if N_nasa > 0 else 1.0
time_nasa_h = np.arange(N_nasa) * dt_nasa_h

print(f"  NASA datapoints: {N_nasa} (dt = {dt_nasa_h:.1f} h)")
print(f"  Mean wind speed at 50 m: {np.nanmean(v_50m):.2f} m/s")

# --- Extrapolate from 50 m to hub height using power law ---
h_ref = 50       # Reference height [m]
alpha = 0.143    # Shear exponent

v_hub = v_50m * (hub_height / h_ref)**alpha

print(f"  Mean wind speed at {hub_height} m (hub): {np.nanmean(v_hub):.2f} m/s")

# --- Interpolate to load timestep ---
# Create interpolation function
f_wind = interp1d(time_nasa_h, v_hub, kind='linear', fill_value='extrapolate')
v_wind = f_wind(time_load_h)
v_wind = np.maximum(v_wind, 0) # Ensure no negative wind speeds

# --- Wind Direction ---
HAS_DIRECTION = False
if 'WD50M' in df_nasa.columns:
    wd_50m = df_nasa['WD50M'].values
    wd_50m = np.where(wd_50m < 0, np.nan, wd_50m)
    # Interpolate circular data (nearest neighbour is safest without complex logic)
    wd_50m = pd.Series(wd_50m).interpolate(method='nearest').values
    
    f_wd = interp1d(time_nasa_h, wd_50m, kind='nearest', fill_value='extrapolate')
    wd_wind = f_wd(time_load_h)
    HAS_DIRECTION = True
    print("  Wind direction data loaded (WD50M)")
else:
    print("  WD50M not found - wind rose skipped")

data_source = f"NASA POWER (MERRA-2), WS50M extrapolated to {hub_height} m"

# =========================================================================
# 5. APPLY TURBINE POWER CURVE
# =========================================================================
P_wind_MW = np.zeros_like(v_wind)

mask_partial = (v_wind >= v_cutin) & (v_wind <= v_rated)
mask_rated = (v_wind > v_rated) & (v_wind <= v_cutout)

P_wind_MW[mask_partial] = P_rated * ((v_wind[mask_partial] - v_cutin) / (v_rated - v_cutin))**3
P_wind_MW[mask_rated] = P_rated

dt_h = dt_load_h


# =========================================================================
# 5b. YAW MISALIGNMENT LOSSES
# =========================================================================
# Yaw system parameters
yaw_rate       = 0.5    # Yaw rotation speed [deg/s] (typical for large turbines)
yaw_deadband   = 8.0    # Yaw controller deadband [deg] (turbine won't yaw for errors < this)
n_cosine       = 2.0    # Cosine exponent (2 = practical, 3 = theoretical)

print(f"\nApplying yaw misalignment losses...")
print(f"  Yaw rate: {yaw_rate} deg/s | Deadband: {yaw_deadband} deg | cos^n exponent: {n_cosine}")

if HAS_DIRECTION:
    # Calculate the rate of wind direction change [deg/timestep]
    # Use circular difference (handles 360->0 wraparound)
    dwd = np.diff(wd_wind)
    dwd = (dwd + 180) % 360 - 180  # Wrap to [-180, 180]
    dwd = np.append(dwd, 0)        # Pad last value
    
    # Maximum yaw correction per timestep [deg]
    max_yaw_per_step = yaw_rate * dt_load_s  # e.g. 0.5 deg/s * 600 s = 300 deg
    
    # Simulate yaw tracking error
    # The nacelle orientation tries to follow the wind but with a lag
    nacelle_dir = np.zeros_like(wd_wind)
    nacelle_dir[0] = wd_wind[0]  # Start aligned
    
    for i in range(1, len(wd_wind)):
        # Error between wind direction and current nacelle orientation
        error = wd_wind[i] - nacelle_dir[i-1]
        error = (error + 180) % 360 - 180  # Wrap to [-180, 180]
        
        # Yaw controller: only act if error exceeds deadband
        if abs(error) > yaw_deadband:
            # Correct towards wind, limited by yaw rate
            correction = np.sign(error) * min(abs(error), max_yaw_per_step)
            nacelle_dir[i] = nacelle_dir[i-1] + correction
        else:
            nacelle_dir[i] = nacelle_dir[i-1]
        
        # Wrap nacelle direction to [0, 360]
        nacelle_dir[i] = nacelle_dir[i] % 360
    
    # Calculate yaw error at each timestep
    yaw_error = wd_wind - nacelle_dir
    yaw_error = (yaw_error + 180) % 360 - 180  # Wrap to [-180, 180]
    yaw_error_abs = np.abs(yaw_error)
    
    # Apply cosine loss: P_actual = P_ideal * cos^n(theta)
    yaw_loss_factor = np.cos(np.radians(yaw_error_abs))**n_cosine
    yaw_loss_factor = np.clip(yaw_loss_factor, 0, 1)
    
    # Store pre-yaw power for comparison
    P_wind_no_yaw = P_wind_MW.copy()
    P_wind_MW = P_wind_MW * yaw_loss_factor
    
    # Statistics
    mean_yaw_error = np.mean(yaw_error_abs)
    max_yaw_error  = np.max(yaw_error_abs)
    E_no_yaw       = np.sum(P_wind_no_yaw) * dt_h
    E_with_yaw     = np.sum(P_wind_MW) * dt_h
    yaw_energy_loss_pct = (1 - E_with_yaw / E_no_yaw) * 100
    
    print(f"  Mean yaw error: {mean_yaw_error:.1f} deg")
    print(f"  Max yaw error:  {max_yaw_error:.1f} deg")
    print(f"  Energy before yaw loss: {E_no_yaw:.0f} MWh")
    print(f"  Energy after yaw loss:  {E_with_yaw:.0f} MWh")
    print(f"  Yaw energy loss: {yaw_energy_loss_pct:.2f}%")
    
    YAW_APPLIED = True
else:
    # Without direction data, apply a flat average yaw loss factor
    # Typical literature value: 2-3% for well-maintained modern turbines
    flat_yaw_loss = 0.02  # 2%
    P_wind_no_yaw = P_wind_MW.copy()
    P_wind_MW = P_wind_MW * (1 - flat_yaw_loss)
    yaw_energy_loss_pct = flat_yaw_loss * 100
    mean_yaw_error = 5.0  # assumed
    
    print(f"  No direction data: applying flat {flat_yaw_loss*100:.0f}% yaw loss")
    YAW_APPLIED = True


# =========================================================================
# Add this line to the CONSOLE SUMMARY section (section 8):
# =========================================================================
# After the existing print statements, add:
if YAW_APPLIED:
    print(f'Mean yaw error:              {mean_yaw_error:.1f} deg')
    print(f'Yaw energy loss:             {yaw_energy_loss_pct:.2f}%')


# =========================================================================
# 6. ENERGY BALANCE
# =========================================================================
Mismatch_MW = P_wind_MW - P_demand_MW

Total_Wind_Gen_MWh = np.sum(P_wind_MW) * dt_h
Total_Demand_MWh = np.sum(P_demand_MW) * dt_h
Surplus_MWh = np.sum(Mismatch_MW[Mismatch_MW > 0]) * dt_h
Deficit_MWh = np.sum(np.abs(Mismatch_MW[Mismatch_MW < 0])) * dt_h

CF_actual = Total_Wind_Gen_MWh / (P_rated * 8760)
Coverage = (Total_Wind_Gen_MWh / Total_Demand_MWh) * 100 if Total_Demand_MWh > 0 else 0
hrs_at_rated = np.sum(P_wind_MW >= P_rated * 0.98) * dt_h

# Monthly breakdown
# Create month indices (1-12) based on day of year
months = np.ceil(time_load_d / 30.44).astype(int)
months = np.clip(months, 1, 12)

monthly_gen = np.zeros(12)
monthly_dem = np.zeros(12)
monthly_vavg = np.zeros(12)

for m in range(1, 13):
    idx_m = (months == m)
    if np.any(idx_m):
        monthly_gen[m-1] = np.sum(P_wind_MW[idx_m]) * dt_h
        monthly_dem[m-1] = np.sum(P_demand_MW[idx_m]) * dt_h
        monthly_vavg[m-1] = np.mean(v_wind[idx_m])

# =========================================================================
# 7. PLOTS
# =========================================================================
# Colors
c_wind = '#0072BD'   # Blue
c_demand = '#D95319' # Orange
c_surplus = '#77AC30' # Green
c_deficit = '#A2142F' # Red
c_gray = '#7F7F7F'

plt.rcParams.update({'font.size': 11})

# ---- FIGURE 1: Annual wind speed ----
plt.figure(figsize=(11, 4))
plt.plot(time_load_d, v_wind, color=c_gray, linewidth=0.3, alpha=0.5, label='10-min values')
# Plot monthly averages
for m in range(1, 13):
    idx_m = (months == m)
    if np.any(idx_m):
        d_start = time_load_d[idx_m][0]
        d_end = time_load_d[idx_m][-1]
        val = monthly_vavg[m-1]
        plt.plot([d_start, d_end], [val, val], color=c_wind, linewidth=2.5)

# Add reference lines
for v, lbl, col in zip([v_cutin, v_rated, v_cutout], ['v_ci', 'v_r', 'v_co'], ['green', 'orange', 'red']):
    plt.axhline(v, linestyle='--', color=col, linewidth=1)
    plt.text(330, v, f'{lbl}={v:.0f}', color=col, fontsize=9)

plt.xlabel('Day of Year')
plt.ylabel('Wind Speed [m/s]')
plt.title(f'Wind Speed at Hub Height ({hub_height} m) – Harvestehude')
plt.xlim(0, 365)
plt.grid(True)
plt.legend(['10-min values', 'Monthly average'], loc='upper right')
plt.tight_layout()

# ---- FIGURE 2: Turbine power curve ----
plt.figure(figsize=(8, 4))
v_pc = np.arange(0, 30.1, 0.1)
P_pc = np.zeros_like(v_pc)
m1 = (v_pc >= v_cutin) & (v_pc <= v_rated)
m2 = (v_pc > v_rated) & (v_pc <= v_cutout)
P_pc[m1] = P_rated * ((v_pc[m1] - v_cutin) / (v_rated - v_cutin))**3
P_pc[m2] = P_rated

plt.plot(v_pc, P_pc, color=c_wind, linewidth=2.5)
plt.fill_between(v_pc, 0, P_rated, where=m1, color=c_wind, alpha=0.1)
plt.fill_between(v_pc, 0, P_rated, where=m2, color=c_surplus, alpha=0.1)

plt.axvline(v_cutin, linestyle=':', color='green')
plt.axvline(v_rated, linestyle=':', color='orange')
plt.axvline(v_cutout, linestyle=':', color='red')

plt.text(v_cutin+0.5, 0.5, f'v_cut-in\n{v_cutin} m/s', color='green', fontsize=9)
plt.text(v_rated+0.5, P_rated*0.5, f'v_rated\n{v_rated} m/s', color='orange', fontsize=9)
plt.text(v_cutout+0.5, P_rated*0.5, f'v_cut-out\n{v_cutout} m/s', color='red', fontsize=9)

plt.xlabel('Wind Speed [m/s]')
plt.ylabel('Power Output [MW]')
plt.title(f'Power Curve – {n_turbines}x Vestas V117-4.2 MW (Total: {P_rated:.1f} MW)')
plt.ylim(0, P_rated+0.8)
plt.xlim(0, 30)
plt.grid(True)
plt.tight_layout()

# ---- FIGURE 3: Annual Generation ----
plt.figure(figsize=(12, 4))
plt.plot(time_load_d, P_wind_MW, color=c_wind, linewidth=0.3, alpha=0.6, label='Wind generation')
plt.axhline(Avg_Load_MW, linestyle='--', color=c_demand, linewidth=1.5, label=f'Avg. Demand ({Avg_Load_MW:.2f} MW)')
plt.axhline(P_rated, linestyle=':', color='black', linewidth=1, label=f'Rated ({P_rated:.1f} MW)')
plt.xlabel('Day of Year')
plt.ylabel('Power [MW]')
plt.title('Annual Wind Power Output – Harvestehude')
plt.xlim(0, 365)
plt.ylim(0, P_rated+0.5)
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()

# ---- FIGURE 4: 7-Day Zoom ----
days_zoom = 7
idx_zoom = time_load_d <= days_zoom
plt.figure(figsize=(11, 4))
plt.plot(time_load_h[idx_zoom], P_wind_MW[idx_zoom], color=c_wind, linewidth=1.2, label='Wind Generation')
plt.plot(time_load_h[idx_zoom], P_demand_MW[idx_zoom], linestyle='--', color=c_demand, linewidth=1.8, label='Household Demand')
plt.axhline(P_rated, linestyle=':', color='black', linewidth=1, label='Rated')
plt.xlabel('Hours')
plt.ylabel('Power [MW]')
plt.title('First 7 Days: Wind Generation vs. Household Demand')
plt.xlim(0, days_zoom*24)
plt.legend()
plt.grid(True)
plt.tight_layout()

# ---- FIGURE 5: Duration Curve ----
plt.figure(figsize=(9, 4))
P_sorted = np.sort(P_wind_MW)[::-1] # Descending sort
hours_axis = np.arange(len(P_sorted)) * dt_h
plt.fill_between(hours_axis, P_sorted, color=c_wind, alpha=0.3)
plt.plot(hours_axis, P_sorted, color=c_wind, linewidth=1.2)
plt.axvline(hrs_at_rated, linestyle='--', color='green')
plt.text(min(hrs_at_rated+100, 8400), P_rated*0.85, f'Hours at rated:\n{hrs_at_rated:.0f} h', color='green')
plt.xlabel('Hours per Year [h]')
plt.ylabel('Power [MW]')
plt.title(f'Annual Power Duration Curve – Wind ({n_turbines}x Vestas V117)')
plt.xlim(0, 8760)
plt.ylim(0, P_rated+0.5)
plt.grid(True)
plt.tight_layout()

# ---- FIGURE 6: Energy Mismatch (7 Days) ----
plt.figure(figsize=(11, 4))
mismatch_zoom = Mismatch_MW[idx_zoom]
time_zoom = time_load_h[idx_zoom]
surplus_plot = np.maximum(mismatch_zoom, 0)
deficit_plot = np.minimum(mismatch_zoom, 0)

plt.fill_between(time_zoom, surplus_plot, 0, color=c_surplus, alpha=0.6, label='Surplus')
plt.fill_between(time_zoom, deficit_plot, 0, color=c_deficit, alpha=0.6, label='Deficit')
plt.axhline(0, color='black', linewidth=1)
plt.xlabel('Hours')
plt.ylabel('Power Mismatch [MW]')
plt.title('Energy Balance: Wind - Load (First 7 Days)')
plt.xlim(0, days_zoom*24)
plt.legend()
plt.grid(True)
plt.tight_layout()

# ---- FIGURE 7: Monthly Balance ----
plt.figure(figsize=(9, 4))
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
x_months = np.arange(12)
width = 0.35

plt.bar(x_months - width/2, monthly_gen, width, label='Wind Generation', color=c_wind, alpha=0.7)
plt.bar(x_months + width/2, monthly_dem, width, label='Household Demand', color=c_demand, alpha=0.7)
plt.xticks(x_months, month_names)
plt.xlabel('Month')
plt.ylabel('Energy [MWh]')
plt.title('Monthly Wind Generation vs. Household Demand')
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()

# ---- FIGURE 8: Wind Distribution + Weibull ----
plt.figure(figsize=(8, 4))
v_pos = v_wind[v_wind > 0.01]
# Fit Weibull (note: scipy weibull_min uses shape (c) and scale)
c_fit, loc_fit, scale_fit = weibull_min.fit(v_pos, floc=0) 

plt.hist(v_wind, bins=np.arange(0, 30.5, 0.5), density=True, color=c_wind, alpha=0.5, edgecolor='white', label='Observed')

x_fit = np.linspace(0, 30, 200)
pdf_fit = weibull_min.pdf(x_fit, c_fit, loc_fit, scale_fit)
plt.plot(x_fit, pdf_fit, 'r-', linewidth=2, label=f'Weibull (k={c_fit:.2f}, λ={scale_fit:.2f})')

plt.axvline(v_cutin, linestyle=':', color='green')
plt.axvline(v_rated, linestyle=':', color='orange')
plt.xlabel('Wind Speed [m/s]')
plt.ylabel('Probability Density')
plt.title(f'Wind Speed Distribution at Hub Height ({hub_height} m)')
plt.xlim(0, 25)
plt.legend()
plt.grid(True)
plt.tight_layout()

# ---- FIGURE 9: Wind Rose ----
if HAS_DIRECTION:
    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1) # Clockwise
    
    n_dir_bins = 16
    dir_edges = np.linspace(0, 360, n_dir_bins + 1)
    dir_width = 360 / n_dir_bins
    
    # Speed bins
    spd_edges = [0, 3, 6, 9, 12, 15, 100]
    spd_labels = ['0-3', '3-6', '6-9', '9-12', '12-15', '>15']
    colors = plt.cm.Blues(np.linspace(0.2, 1, len(spd_edges)-1))
    
    # Create histogram
    # We need to rotate 360 to 0 for proper binning sometimes, but histogram handles it if we are careful
    # For wind rose, typically we center the bins. E.g. N is -11.25 to +11.25.
    # The MATLAB code handled this manually. Let's do similar simple binning.
    
    # Align directions to bin centers (N=0, NNE=22.5, etc.)
    # We shift data by half bin width so 0 becomes center of first bin
    wd_shifted = (wd_wind + dir_width/2) % 360
    
    hist_stacked = np.zeros((n_dir_bins, len(spd_edges)-1))
    
    for i, (s_low, s_high) in enumerate(zip(spd_edges[:-1], spd_edges[1:])):
        mask_spd = (v_wind >= s_low) & (v_wind < s_high)
        # Use histogram to count directions for this speed bin
        counts, _ = np.histogram(wd_shifted[mask_spd], bins=np.linspace(0, 360, n_dir_bins+1))
        hist_stacked[:, i] = counts

    # Convert to percentage
    total_count = len(v_wind)
    hist_pct = hist_stacked / total_count * 100
    
    # Plot bars
    theta = np.linspace(0, 2*np.pi, n_dir_bins, endpoint=False)
    width = np.radians(dir_width) * 0.9
    
    bottom = 0
    for i in range(len(spd_labels)):
        bars = ax.bar(theta, hist_pct[:, i], width=width, bottom=bottom, color=colors[i], label=f"{spd_labels[i]} m/s")
        bottom += hist_pct[:, i]
        
    ax.set_xticklabels(['N','NNE','NE','ENE','E','ESE','SE','SSE',
                        'S','SSW','SW','WSW','W','WNW','NW','NNW'])
    plt.title(f'Wind Rose – Harvestehude ({hub_height} m)', y=1.08)
    plt.legend(loc='lower right', bbox_to_anchor=(1.3, 0))
    plt.tight_layout()

# ---- FIGURE 10: Directional Energy ----
if HAS_DIRECTION:
    plt.figure(figsize=(10, 5))
    
    # Calculate energy per bin
    energy_per_dir = np.zeros(n_dir_bins)
    wd_shifted = (wd_wind + dir_width/2) % 360
    
    # Using digitize to find bin indices
    bin_indices = np.digitize(wd_shifted, np.linspace(0, 360, n_dir_bins+1)) - 1
    bin_indices = np.clip(bin_indices, 0, n_dir_bins-1)
    
    # Sum energy for each bin
    # Use pandas groupby for speed or simple loop
    df_energy = pd.DataFrame({'bin': bin_indices, 'energy': P_wind_MW * dt_h})
    energy_sums = df_energy.groupby('bin')['energy'].sum()
    
    # Fill array (some bins might be empty)
    for b_idx, val in energy_sums.items():
        energy_per_dir[int(b_idx)] = val
        
    energy_pct = energy_per_dir / Total_Wind_Gen_MWh * 100
    
    dir_labels = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
    
    bars = plt.bar(range(n_dir_bins), energy_pct, color=c_wind, alpha=0.7)
    plt.xticks(range(n_dir_bins), dir_labels)
    plt.xlabel('Wind Direction')
    plt.ylabel('Energy Contribution [%]')
    plt.title('Wind Energy Contribution by Direction')
    plt.grid(True, axis='y')
    
    for rect in bars:
        height = rect.get_height()
        if height > 1:
            plt.text(rect.get_x() + rect.get_width()/2.0, height, f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

# =========================================================================
# 8. CONSOLE SUMMARY
# =========================================================================
print('\n' + '='*55)
print('   HARVESTEHUDE WIND ENERGY REPORT (Python Version)')
print(f'   {n_turbines}x Vestas V117-4.2 MW | Hamburg, Germany')
print('='*55)
print(f'Data source:                 {data_source}')
print(f'Households:                  {n_households}')
print(f'Installed capacity:          {P_rated:.1f} MW')
print('-'*55)
print(f'Annual demand:               {Total_Demand_MWh:.0f} MWh')
print(f'Annual wind generation:      {Total_Wind_Gen_MWh:.0f} MWh')
print(f'Wind target (50%):           {Annual_Load_MWh * 0.50:.0f} MWh')
print('-'*55)
print(f'Capacity factor (CF):        {CF_actual*100:.1f} %')
print(f'Wind coverage of demand:     {Coverage:.1f} %')
print(f'Annual surplus:              {Surplus_MWh:.0f} MWh')
print(f'Annual deficit:              {Deficit_MWh:.0f} MWh')
print('='*55)

# =========================================================================
# 9. EXPORT FIGURES AUTOMATICALLY
# =========================================================================
import re

# Map aanmaken als die nog niet bestaat
output_dir = 'figures_sipWind'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"\nMap '{output_dir}' aangemaakt.")

print(f"Figuren opslaan in map '{output_dir}'...")

# Haal alle nummers op van de figuren die nu open staan
fignums = plt.get_fignums()

for i in fignums:
    # Selecteer het figuur
    fig = plt.figure(i)
    
    # Probeer de titel uit de grafiek te halen om als bestandsnaam te gebruiken
    try:
        # Haal titel op van de eerste as (axes)
        original_title = fig.axes[0].get_title()
        
        # Bestandsnaam opschonen: 
        # 1. Vervang spaties door underscores
        # 2. Verwijder alles wat geen letter, cijfer of underscore is
        clean_title = re.sub(r'[^\w\s-]', '', original_title) # verwijder leestekens
        clean_title = re.sub(r'[-\s]+', '_', clean_title)    # spaties naar _
        
        # Als er geen titel is gevonden, gebruik een standaard naam
        if not clean_title:
            clean_title = f"Figure_{i}"
    except:
        clean_title = f"Figure_{i}"

    # Bestandsnaam samenstellen
    filename = f"{output_dir}/{clean_title}.png"
    
    # Opslaan
    # dpi=300 zorgt voor hoge kwaliteit (goed voor verslagen)
    # bbox_inches='tight' zorgt dat de labels niet van de rand vallen
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Opgeslagen: {filename}")

print('='*55)

# =========================================================================
# 10. EXPORT DATA TO CSV (EXCEL FRIENDLY)
# =========================================================================
print(f"\nGenereren van data-bestand...")

# Verzamel de belangrijkste variabelen in een Pandas DataFrame
# We voegen ook de tijd in uren en dagen toe voor het overzicht
df_results = pd.DataFrame({
    'Time_hours': time_load_h,
    'Day_of_Year': time_load_d,
    'Wind_Speed_Hub_ms': v_wind,
    'Wind_Power_Gen_MW': P_wind_MW,
    'Power_Demand_MW': P_demand_MW,
    'Mismatch_MW': Mismatch_MW  # Positief = overschot, Negatief = tekort
})

# Bestandsnaam definiëren
output_csv = 'Harvestehude_Wind_Output_Yearly.csv'

# Opslaan als CSV
# index=False: Geen rijnummers opslaan
# sep=';': Puntkomma als scheidingsteken (standaard voor NL Excel)
# decimal=',': Komma als decimaalteken
df_results.to_csv(output_csv, index=False, sep=';', decimal=',')

print(f"  Bestand succesvol aangemaakt: {output_csv}")
print(f"  Je kunt dit bestand direct openen in Excel.")
print('='*55)


plt.show()