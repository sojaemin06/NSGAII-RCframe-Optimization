clc; clear all;

rng('shuffle');
b_range=300:100:500;
h_range=500:50:800;
dimension_range = [16,22];
stirrup_range=[10,13];
conblock = 25;
fck_range=[24, 27, 30];
fy_range=[400,500];
Es=200000;

% --- 비용 및 CO2 계산을 위한 상수 정의 ---
SG_steel = 7.85; % ton/m^3
Cc = 80000; % won/m^3
Cs = 1000000; % won/ton

% <<< 추가: 내재탄소 계산을 위한 상수 >>>
DENSITY_CONCRETE_KG_M3 = 2400; % kg/m^3
DENSITY_STEEL_KG_M3 = 7850;    % kg/m^3
ECF_CONCRETE = 0.15;           % kgCO2e/kg
ECF_STEEL = 1.99;              % kgCO2e/kg

% n값 선정
n = 1000;
% <<< 변경: CO2 열 추가로 최종 데이터 열 개수 19 -> 20으로 변경
Beam_Data = zeros(n,22); 

for i=1:1:n
    while true
    b = randsample(b_range,1);
    h = randsample(h_range,1);
    
    stirrup =randsample(stirrup_range,1);
    fck = randsample(fck_range,1);
    fy = randsample(fy_range,1);
    dimension = randsample(dimension_range,1);
    
    d_initial = h - 100;

    ConBlock = ceil(conblock*(4/3));
    space_list = [25, dimension, ConBlock];
    space = max(space_list);
    
    if dimension <= 16
        PI_size = 40;
    elseif dimension <= 28
        PI_size = 50;
    else
        PI_size = 60;
    end
    total_size = PI_size + stirrup;

    if fck <= 40
        ecu = 0.0033; Beta = 0.8; Eta = 1.0;
    elseif fck <= 50
        ecu = 0.0032; Beta = 0.8; Eta = 0.97;
    elseif fck <= 60
        ecu = 0.0031; Beta = 0.76; Eta = 0.95;
    elseif fck <= 70
        ecu = 0.0030; Beta = 0.74; Eta = 0.91;
    elseif fck <= 80
        ecu = 0.0029; Beta = 0.72; Eta = 0.87;
    else
        ecu = 0.0028; Beta = 0.70; Eta = 0.84;
    end

    rho_b = Beta*Eta*(0.85*fck/fy)*(ecu/(ecu+(fy/Es)));

    min_rho_T = rho_b * 0.3;
    max_rho_T = rho_b * 0.4;
    rho_s_T = min_rho_T + (max_rho_T - min_rho_T) * rand;
    min_rho_C = 0.4 * rho_s_T;
    max_rho_C = 0.5 * rho_s_T;
    rho_s_C = min_rho_C + (max_rho_C - min_rho_C) * rand;
    
    n_b = floor((b-(total_size + dimension/2)*2)/(dimension + space))+1;
    N_r = ceil((rho_s_T*b*d_initial)/((dimension/2)^2*pi));
    N_n = ceil(N_r/n_b);
    
    N_r_C = ceil((rho_s_C * b * d_initial) / ((dimension/2)^2 * pi));
    N_n_C = ceil(N_r_C / n_b);

    d_actual = h - (total_size + dimension/2) - (space + dimension) * (N_n - 1) / 2;
    
    strup_space_range = 100:50:0.5*d_actual;
    
    if isempty(strup_space_range)
        continue;
    end
    
    if length(strup_space_range) == 1
        strup_space = strup_space_range;
    else
        strup_space = randsample(strup_space_range, 1);
    end
    
    if strup_space > 600
        strup_space = 600;
    end

    % <<< 변경: CO2 계산에 필요한 상수들을 함수에 전달 >>>
    Data = STM_Beam_Function_02([b,d_actual,rho_s_T,rho_s_C,fck,fy, ...
        dimension,PI_size,stirrup,conblock,ConBlock,total_size,space,strup_space, ...
        SG_steel,Cc,Cs,h,N_n, ...
        DENSITY_CONCRETE_KG_M3, DENSITY_STEEL_KG_M3, ECF_CONCRETE, ECF_STEEL]);
        
    % <<< 변경: 반환된 Data의 열 개수가 13개로 늘어남
    Beam_Data(i,:) = [Data, dimension, PI_size, n_b, N_r, N_n, N_r_C, N_n_C];

    Ramda = 1;
    Mcr = 0.63*Ramda*sqrt(fck)*b*(h^2)/6/10^6;
            if Beam_Data(i,4) >= 1.2*Mcr 
                break
            else
                continue
            end
    end
    fprintf('진행률: %3.0f%% (%d/%d)\n', i/n*100, i, n);
end