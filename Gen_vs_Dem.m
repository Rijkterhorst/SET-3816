data = readtable('structured_weather_data_v4.xlsx','Sheet','One year weather data','Range','C:H');

Tout  = data{:,2};

Delta_T_gen = 82.7 - 45;    %Kelvin
Flow_rate = 84.7;   %kg/s
Heat_capacity = 4186;   %J/kg*K
P_doublet = (Delta_T_gen*Flow_rate*Heat_capacity)/1000000;  %in MW
Doublet_amount = 1;
P_gen = P_doublet * Doublet_amount;

Heat_loss_coefficient = 207.7486;   % W/K
Qheat_hh = Heat_loss_coefficient*(16 - Tout)/1000000; %household heat
Qheat_hh(Qheat_hh < 0) = 0;      % no cooling
Qheat = Qheat_hh * 10869;       %total heat 
time = 1:length(Qheat);

Q_fulfilled = sum(P_gen >= Qheat);      %where generation suffices
Coverage = (Q_fulfilled/length(Qheat))*100;     %coverage percentage
fprintf('Coverage percentage = %.2f %%\n', Coverage);

Mismatch = P_gen - Qheat;   % "+" -> surplus, "-" -> deficit
TotalMismatch = sum(Mismatch);
fprintf('Total mismatch = %.2f MWh\n', TotalMismatch);
%% Figure P_gen and Qheat
figure
plot(time, Qheat)
hold on
yline(P_gen,'r','LineWidth',2)
xlabel('Time (hours)')
ylabel('Power (MW)')
title('Heating Demand vs Generation Capacity')
legend('Heating Demand','Generation Capacity')
grid on

%% Figure P_gen, Qheat and Mismatch
figure
plot(time, Qheat,'b','LineWidth',1.2)
hold on

yline(P_gen,'r','LineWidth',2)
plot(time, Mismatch,'m','LineWidth',1)
yline(0,'k--')
xlabel('Time')
ylabel('Power (MW)')
title('Generation, Demand and Mismatch')
legend('Q_{heat}','P_{gen}','Mismatch')
grid on

%% Figure Mismatch 
figure
plot(time, Mismatch,'m','LineWidth',1.5)
yline(0,'k--')

xlabel('Time (hours)')
ylabel('Mismatch (MW)')
title('Mismatch')
grid on