
%% ===================== A) LOCATION =====================
lat = 53.5764;   % Harvestehude latitude (Hamburg)
lon = 9.9867;    % Harvestehude longitude (Hamburg)

% Pick a year with reliable PVGIS coverage.
yearSel = 2020;

%% ===================== B) LOAD DEMAND =====================
% Annual load demand that the PV system must cover
E_load_annual_kWh = 100000;   % <-- CHANGE THIS to your project demand
E_pv_target_kWh   = E_load_annual_kWh;

%% ===================== C) SELECTED PV MODULE =====================
PV.Pmpp_W     = 550;      % rated power at STC [W]
PV.Vmp_V      = 41.96;    % voltage at MPP [V]
PV.Imp_A      = 13.11;    % current at MPP [A]
PV.Voc_V      = 49.90;    % open circuit voltage [V]
PV.Isc_A      = 14.00;    % short circuit current [A]
PV.efficiency = 0.2128;   % module efficiency [-] (example, replace)

%% ===================== D) CONVERTERS / SYSTEM ASSUMPTIONS =====================
% PV array -> DC/DC MPPT -> DC bus -> inverter (if AC load)
PVGIS_loss_percent = 14; % 

% DC maximum system voltage (typical inverter limits: 1000 V)
Vdc_max = 1000; % [V]

% Series modules per string with safety margin for cold Voc rise
safetyMargin = 0.90; 
Ns = floor(safetyMargin * Vdc_max / PV.Voc_V);
Ns = max(Ns, 1);

P_string_W = Ns * PV.Pmpp_W;

%% ===================== E) PVGIS HOURLY DATA (1 kWp REFERENCE) =====================
[pvgis, meta] = get_pvgis_seriescalc(lat, lon, yearSel, 1.0, PVGIS_loss_percent);

t       = pvgis.time;
P_ref_W = pvgis.P_pv_W;

% Basic sanity check: PVGIS should return full-year hourly values (usually 8760)
if numel(P_ref_W) < 8000
    warning('PVGIS returned fewer than ~8000 hourly points. Check API response / selected year.');
end

%% ===================== F) SIZE PV SYSTEM TO HIT ANNUAL DEMAND =====================
E_ref_kWh = sum(P_ref_W) / 1000;     % hourly: sum(W)*h -> Wh; /1000 -> kWh

% Scale factor = required annual energy / reference annual energy
scale_factor = E_pv_target_kWh / E_ref_kWh;

% Initial installed capacity (kWp)
kWp_installed = scale_factor;

% Scale hourly PV power to match installed capacity
P_pv_W = P_ref_W * scale_factor;

% Determine parallel strings required for integer PV topology
P_array_W = kWp_installed * 1000;
Np = ceil(P_array_W / P_string_W);

% Enforce integer strings and update installed capacity
P_array_real_W = Np * P_string_W;
kWp_installed  = P_array_real_W / 1000;

% Rescale PV output so peak power matches integer topology
P_pv_W = P_pv_W * (P_array_real_W / P_array_W);

% Final annual PV energy (after topology rounding)
E_pv_kWh = sum(P_pv_W) / 1000;

%% ===================== G) AREA CALCULATION =====================
peak_power_density_Wm2 = PV.efficiency * 1000;

Area_m2  = P_array_real_W / peak_power_density_Wm2;
Area_km2 = Area_m2 / 1e6;

%% ===================== H) PERFORMANCE METRICS =====================
hours_year = numel(t);

CapacityFactor = E_pv_kWh / (kWp_installed * hours_year);

% Performance Ratio (robust, NO irradiance columns needed)
% Using PVGIS reference yield method:
% Yf = E_pv / P_installed, Yr = E_ref / 1kWp, PR = Yf / Yr
Yf = E_pv_kWh / kWp_installed;  % [kWh/kWp]
Yr = E_ref_kWh / 1.0;           % [kWh/kWp]
PR = Yf / Yr;

%% ===================== I) PLOTS (ENTIRE YEAR) =====================
figure;
plot(t, P_pv_W/1000);
xlabel('Time');
ylabel('PV Power [kW]');
title('Hamburg-Harvestehude: PV Power (Year)');
grid on;

% Optional: plot temperature if PVGIS returned it
if ismember('T2m', pvgis.Properties.VariableNames)
    figure;
    plot(t, pvgis.T2m);
    xlabel('Time');
    ylabel('Ambient Temperature [°C]');
    title('Hamburg-Harvestehude: Temperature (Year)');
    grid on;
end

%% ===================== J) OUTPUT SUMMARY =====================
fprintf('\n===== SOLAR SYSTEM – HAMBURG (HARVESTEHUDE) =====\n');
fprintf('Location: lat=%.4f, lon=%.4f\n', lat, lon);
fprintf('Year used: %d\n', yearSel);

fprintf('\n--- Demand & Energy ---\n');
fprintf('Annual demand target: %.1f kWh/year\n', E_load_annual_kWh);
fprintf('Annual PV generation (modelled): %.1f kWh/year\n', E_pv_kWh);

fprintf('\n--- PV Module (replace with your datasheet) ---\n');
fprintf('Pmpp: %.1f W | Vmp: %.2f V | Imp: %.2f A | Voc: %.2f V | Isc: %.2f A\n', ...
    PV.Pmpp_W, PV.Vmp_V, PV.Imp_A, PV.Voc_V, PV.Isc_A);

fprintf('\n--- PV Topology ---\n');
fprintf('DC max: %.0f V, safety margin: %.0f%%\n', Vdc_max, safetyMargin*100);
fprintf('Series modules per string (Ns): %d\n', Ns);
fprintf('Parallel strings (Np): %d\n', Np);
fprintf('Installed PV capacity (topology-adjusted): %.2f kWp\n', kWp_installed);

fprintf('\n--- Area ---\n');
fprintf('Module efficiency used: %.4f\n', PV.efficiency);
fprintf('Total PV area: %.2f m^2 (%.6f km^2)\n', Area_m2, Area_km2);

fprintf('\n--- Performance ---\n');
fprintf('Capacity factor: %.3f\n', CapacityFactor);
fprintf('Performance Ratio (PR): %.3f\n', PR);

%% ========================================================================
%% Helper: PVGIS API call (robust)
function [T, meta] = get_pvgis_seriescalc(lat, lon, yearSel, kWp, loss_percent)

    % Try v5_2 first; if it fails, automatically try v5_3
    urls = { ...
        'https://re.jrc.ec.europa.eu/api/v5_2/seriescalc', ...
        'https://re.jrc.ec.europa.eu/api/v5_3/seriescalc' ...
    };

    opts = weboptions('Timeout', 60);

    lastErr = [];
    for u = 1:numel(urls)
        try
            data = webread(urls{u}, ...
                'lat', lat, ...
                'lon', lon, ...
                'raddatabase', 'PVGIS-SARAH2', ...   % Europe-appropriate dataset
                'startyear', yearSel, ...
                'endyear', yearSel, ...
                'pvcalculation', 1, ...
                'peakpower', kWp, ...
                'loss', loss_percent, ...
                'angle', 30, ...
                'aspect', 0, ...
                'outputformat', 'json', ...
                opts);

            meta = data.meta;
            H = data.outputs.hourly;

            % time strings like 'YYYYMMDD:HHMM'
            timeStr = string({H.time})';
            time = datetime(timeStr, 'InputFormat', 'yyyyMMdd:HHmm', 'TimeZone', 'UTC');

            % PV power [W]
            P = [H.P]';

            vars = struct();
            vars.P_pv_W = P;

            % Optional fields if present
            if isfield(H, 'T2m')
                vars.T2m = [H.T2m]';
            end

            T = struct2table(vars);
            T.time = time;
            T = movevars(T, 'time', 'Before', 1);
            return;

        catch ME
            lastErr = ME;
        end
    end

    % If both endpoints failed, throw the last error
    rethrow(lastErr);
end
