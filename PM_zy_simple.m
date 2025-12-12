clc; clear all;
rng('shuffle');


n=1000;
Column_Data=zeros(n,26);
Column_Mdata=cell(n,1);

% for 루프를 while 루프로 변경하여 n개의 '고유한' 데이터가 채워질 때까지 반복
cnt = 1; 
while cnt <= n
    % ============================================================= %
    % ===== 1. 단면 및 철근 배근 설정 ============================== %
    % ============================================================= %
    
    aggregate = 25;
    Stirrup_size_range=[10,13];
    Stirrup_size = randsample(Stirrup_size_range,1);
    Es = 200000;
    MainRebar_range = [25,29,32];
    min_rho = 0.01;
    max_rho = 0.08;
    SG_steel = 7.85; % ton/m^3
    Cc = 80000; % won/m^3
    Cs = 1000000; % won/ton
    
    DENSITY_CONCRETE_KG_M3 = 2400; % kg/m^3
    DENSITY_STEEL_KG_M3 = 7850;    % kg/m^3
    ECF_CONCRETE = 0.15;           % kgCO2e/kg
    ECF_STEEL = 1.99;              % kgCO2e/kg


    while true
        MainRebar_size = randsample(MainRebar_range,1);

        if MainRebar_size <= 28
            Cover_size = 50;
        elseif MainRebar_size <= 16
            Cover_size = 40;
        else
            Cover_size = 60;
        end

        Area_MainRebar = (pi * MainRebar_size^2) / 4;
        Area_StirrupRebar = (pi * Stirrup_size^2) / 4;
        b_range = 600:50:1000;
        b = randsample(b_range, 1);
        h_range = b:50:b*1.5;
        h = randsample(h_range, 1);
        Ag = b * h;
        Stirrup_verticle_range = 100:50:min([MainRebar_size*16, Stirrup_size*48, b]);
        Stirrup_verticle_size = randsample(Stirrup_verticle_range, 1);
        rho = min_rho + (max_rho - min_rho) * rand;
        total_rebar_area = Ag * rho;
        required_rebar = ceil(total_rebar_area / Area_MainRebar);
    
        if mod(required_rebar, 2) ~= 0
            required_rebar = required_rebar + 1;
        end
        
        rho = (required_rebar * Area_MainRebar) / Ag;
        total_rebar_area = Ag * rho;
    
        corner_rebars = 4;
        remain = required_rebar - corner_rebars;
        side_rebars = [0 0 0 0];
        while remain >= 2
            pair = randsample(["front-back", "left-right"], 1);
            switch pair
                case "front-back"
                    side_rebars(1) = side_rebars(1) + 1;
                    side_rebars(2) = side_rebars(2) + 1;
                case "left-right"
                    side_rebars(3) = side_rebars(3) + 1;
                    side_rebars(4) = side_rebars(4) + 1;
            end
            remain = remain - 2;
        end
    
        rebar_clear_spacing = max([40, MainRebar_size * 1.5, aggregate * 4/3]);
        min_center_spacing = rebar_clear_spacing + MainRebar_size;
        clear_cover = Cover_size + Stirrup_size + MainRebar_size / 2;
        effective_h = h - 2 * clear_cover;
        effective_b = b - 2 * clear_cover;
        
        n_h = side_rebars(1) + 2; % 강축(h방향) 철근 층 수
        n_b = side_rebars(3) + 2; % 약축(b방향) 철근 층 수
    
        
        spacing_h = calc_spacing(effective_h, n_h);
        spacing_b = calc_spacing(effective_b, n_b);
    
        if spacing_h >= min_center_spacing && spacing_b >= min_center_spacing && required_rebar >= 4
            break;
        end
    end
    
    % ============================================================= %
    % ===== 2. 강축 (Z축) 방향 강도 계산 =========================== %
    % ============================================================= %
    fck_range = [27, 30, 35, 40];
    fy_range = [400, 500];
    fck = randsample(fck_range,1);
    fy = randsample(fy_range,1);
    Rebar_YieldStrain = fy / Es;
    
    count = 0.003:-0.00001:-50 * Rebar_YieldStrain;
    
    % 강축(Mz) 계산을 위한 변수
    Pnb_z = zeros(1, length(count));
    Mnb_z = zeros(1, length(count));
    d_prime_z = zeros(n_h, 1);
    m_distance_z = zeros(n_h, 1);
    esi_z = zeros(n_h, length(count));
    fsi_z = zeros(n_h, length(count));
    nsi_z = zeros(n_h, length(count));
    
    % 강축 방향의 층별 철근 단면적 계산
    area_z_intermediate = zeros(1, side_rebars(1));
    for r = 1:side_rebars(1)
        area_z_intermediate(:,r) = Area_MainRebar * 2;
    end
    area_rabar_cols_z = [Area_MainRebar * n_b, area_z_intermediate, Area_MainRebar * n_b];
    
    d_prime_1_z = h - clear_cover;
    centroid_z = h / 2;
    
    for i = 1:length(count)
        x = count(i);
        Neutral_depth = (0.003 / (0.003 - x)) * d_prime_1_z;
        if fck < 28, Beta = 0.85; else, Beta = max((0.85 - (fck - 28) * 0.05 / 7), 0.65); end
        a = Beta * Neutral_depth;
        if a > h, a = h; end
        
        for j = 1:n_h
            d_prime_z(j,1) = h-clear_cover-spacing_h*(j-1);
            esi_z(j,i) = ((Neutral_depth - d_prime_z(j,1)) / Neutral_depth) * 0.003;
            if abs(esi_z(j,i)) > Rebar_YieldStrain, fsi_z(j,i) = sign(esi_z(j,i)) * fy; else, fsi_z(j,i) = Es * esi_z(j,i); end
            if d_prime_z(j,1) < a, nsi_z(j,i) = (fsi_z(j,i) - 0.85 * fck) * area_rabar_cols_z(1,j) * 10^-3; else, nsi_z(j,i) = fsi_z(j,i) * area_rabar_cols_z(1,j) * 10^-3; end
            m_distance_z(j,1) = centroid_z - d_prime_z(j,1);
        end
        Pnb_z(1,i) = (0.85 * fck * a * b * 10^-3) + sum(nsi_z(:,i));
        Mnb_z(1,i) = ((0.85 * fck * a * b * (centroid_z - a/2) * 10^-3) + sum(nsi_z(:,i) .* m_distance_z(:,1))) * 10^-3;
    end
    Pn0 = (0.85 * fck * (Ag - total_rebar_area) + fy * total_rebar_area) * 10^-3;
    Pnb_z(1,1) = Pn0; Mnb_z(1,1) = 0;
    
    
    % ============================================================= %
    % ===== 3. 약축 (Y축) 방향 강도 계산 =========================== %
    % ============================================================= %
    Pnb_y = zeros(1, length(count));
    Mnb_y = zeros(1, length(count));
    d_prime_y = zeros(n_b, 1);
    m_distance_y = zeros(n_b, 1);
    esi_y = zeros(n_b, length(count));
    fsi_y = zeros(n_b, length(count));
    nsi_y = zeros(n_b, length(count));
    
    area_y_intermediate = zeros(1, side_rebars(3));
    for r = 1:side_rebars(3)
        area_y_intermediate(:,r) = Area_MainRebar * 2;
    end
    area_rabar_cols_y = [Area_MainRebar * n_h, area_y_intermediate, Area_MainRebar * n_h];
    
    d_prime_1_y = b - clear_cover;
    centroid_y = b / 2;
    
    for i = 1:length(count)
        x = count(i);
        Neutral_depth = (0.003 / (0.003 - x)) * d_prime_1_y;
        if fck < 28, Beta = 0.85; else, Beta = max((0.85 - (fck - 28) * 0.05 / 7), 0.65); end
        a = Beta * Neutral_depth;
        if a > b, a = b; end
        
        for j = 1:n_b
            d_prime_y(j,1) = b-clear_cover-spacing_b * (j-1);
            esi_y(j,i) = ((Neutral_depth - d_prime_y(j,1)) / Neutral_depth) * 0.003;
            if abs(esi_y(j,i)) > Rebar_YieldStrain, fsi_y(j,i) = sign(esi_y(j,i)) * fy; else, fsi_y(j,i) = Es * esi_y(j,i); end
            if d_prime_y(j,1) < a, nsi_y(j,i) = (fsi_y(j,i) - 0.85 * fck) * area_rabar_cols_y(1,j) * 10^-3; else, nsi_y(j,i) = fsi_y(j,i) * area_rabar_cols_y(1,j) * 10^-3; end
            m_distance_y(j,1) = centroid_y - d_prime_y(j,1);
        end
        Pnb_y(1,i) = (0.85 * fck * a * h * 10^-3) + sum(nsi_y(:,i));
        Mnb_y(1,i) = ((0.85 * fck * a * h * (centroid_y - a/2) * 10^-3) + sum(nsi_y(:,i) .* m_distance_y(:,1))) * 10^-3;
    end
    Pnb_y(1,1) = Pn0; Mnb_y(1,1) = 0;
    
    % ====================================================================== %
    % ===== 4. 설계강도(ΦPn-ΦM) 계산 ======================================== %
    % ====================================================================== %
    [PI_Pnb_z, PI_Mnb_z, Alpha_PI_Pnb_z, Alpha_PI_Mnb_z] = calculate_design_strength(Pnb_z, Mnb_z, count, fy, Rebar_YieldStrain, Pn0);
    [PI_Pnb_y, PI_Mnb_y, Alpha_PI_Pnb_y, Alpha_PI_Mnb_y] = calculate_design_strength(Pnb_y, Mnb_y, count, fy, Rebar_YieldStrain, Pn0);
    
    Pnt = -(total_rebar_area * fy) * 10^-3;
    phi_tension = 0.85;
    PI_Pnt = phi_tension * Pnt;
    
    Pnb_z(end+1) = Pnt;
    Mnb_z(end+1) = 0;
    Alpha_PI_Pnb_z(end+1) = PI_Pnt;
    Alpha_PI_Mnb_z(end+1) = 0;
    
    Pnb_y(end+1) = Pnt;
    Mnb_y(end+1) = 0;
    Alpha_PI_Pnb_y(end+1) = PI_Pnt;
    Alpha_PI_Mnb_y(end+1) = 0;

    % =============================================== %
    % ===== 5. 전단강도 계산 ========================= %
    % =============================================== %
    Vn_c_z = 1/6*sqrt(fck)*h*d_prime_1_z;
    Vn_s_z = (Area_StirrupRebar*2*fy*d_prime_1_z)/Stirrup_verticle_size;
    Vn_c_y = 1/6*sqrt(fck)*b*d_prime_1_y;
    Vn_s_y = (Area_StirrupRebar*2*fy*d_prime_1_y)/Stirrup_verticle_size;
    PI_Vn_z = 0.75*(Vn_c_z + Vn_s_z)*10^-3;
    PI_Vn_y = 0.75*(Vn_c_y + Vn_s_y)*10^-3;
    
    % =============================================== %
    % ===== 6. 비용 및 CO2 계산 ====================== %
    % =============================================== %
    % --- 비용(Cost) 계산 ---
    % 주철근 및 콘크리트 비용
    Cost1 = (Ag - total_rebar_area)*10^(-6)*Cc + total_rebar_area*10^(-6)*SG_steel*Cs;
    % 띠철근(Stirrup) 비용 (단위 m당 비용으로 환산)
    stirrup_len_one_m = ((b-2*Cover_size)+(h-2*Cover_size))*2*10^(-3);
    stirrup_area_m2 = ((Stirrup_size/2)^2*pi*10^(-6));
    stirrup_spacing_m = Stirrup_verticle_size / 1000;
    Cost2 = (stirrup_len_one_m * stirrup_area_m2 / stirrup_spacing_m) * SG_steel * Cs;
    Cost = Cost1 + Cost2;

    % --- <<< 추가: 내재탄소(CO2) 계산 >>> ---
    % 주철근 및 콘크리트 CO2
    concrete_area_mm2 = Ag - total_rebar_area;
    concrete_vol_per_m = concrete_area_mm2 * 10^(-6);
    steel_vol_per_m = total_rebar_area * 10^(-6);
    
    concrete_mass_per_m = concrete_vol_per_m * DENSITY_CONCRETE_KG_M3;
    steel_mass_per_m = steel_vol_per_m * DENSITY_STEEL_KG_M3;
    
    CO2_1 = (concrete_mass_per_m * ECF_CONCRETE) + (steel_mass_per_m * ECF_STEEL);
    
    % 띠철근(Stirrup) CO2 (단위 m당 CO2)
    stirrup_vol_per_m = (stirrup_len_one_m * stirrup_area_m2) / stirrup_spacing_m;
    stirrup_mass_per_m = stirrup_vol_per_m * DENSITY_STEEL_KG_M3;
    CO2_2 = stirrup_mass_per_m * ECF_STEEL;
    
    CO2 = CO2_1 + CO2_2; % 단위 m당 총 CO2 배출량

    % --- <<< 추가: 단면별 단위중량(UnitWeight) 계산 >>> ---
    total_steel_vol_per_m = steel_vol_per_m + stirrup_vol_per_m;
    total_concrete_vol_per_m = Ag*10^(-6) - total_steel_vol_per_m;
    
    total_mass_per_m = (total_concrete_vol_per_m * DENSITY_CONCRETE_KG_M3) + (total_steel_vol_per_m * DENSITY_STEEL_KG_M3);
    UnitWeight = (total_mass_per_m * 9.81) / 1000; % 최종 단위중량 (kN/m^3)


    % --- <<< 추가: P-M 상관도 체적(성능 지표) 계산 >>> ---
    % 2차원 P-M 다이어그램의 면적을 계산하여 합산 (MATLAB의 polyarea 함수 사용)
    % M 값은 항상 양수가 되도록 절댓값을 취해줍니다.
    Area_z = polyarea(abs(Mnb_z), Pnb_z);
    Area_y = polyarea(abs(Mnb_y), Pnb_y);
    PM_Volume = Area_z + Area_y; % 체적의 근사치로 사용



    % =============================================== %
    % ===== 7. 데이터 집계 및 중복 확인 ============= %
    % =============================================== %
    
    % <<< 변경: candidate_row에 CO2를 마지막 열로 추가
    candidate_row = [h,b,Alpha_PI_Pnb_z(1),PI_Vn_z,PI_Vn_y,0,0,required_rebar,MainRebar_size,rho,side_rebars+2,spacing_h,spacing_b,Stirrup_size,Stirrup_verticle_size,fck,fy,Cover_size,aggregate,Cost,CO2,UnitWeight,PM_Volume];
    
    is_duplicate = false;
    if cnt > 1
        if ismember(candidate_row, Column_Data(1:cnt-1,:), 'rows')
            is_duplicate = true;
        end
    end

    if ~is_duplicate
        Column_moment_data=[Mnb_z; Pnb_z; Mnb_y; Pnb_y; Alpha_PI_Mnb_z; Alpha_PI_Pnb_z; Alpha_PI_Mnb_y; Alpha_PI_Pnb_y];
        Column_Mdata{cnt,:} = Column_moment_data;
        Column_Data(cnt,:) = candidate_row;
        
        fprintf('진행률: %3.0f%% (%d/%d) - 데이터 추가됨\n', cnt/n*100, cnt, n);
        cnt = cnt + 1;
    else
        fprintf('중복 데이터 발생, 재생성합니다... (현재 %d개 완료)\n', cnt-1);
    end
end


% ==================================== %
% ===== 함수 ========================= %
% ==================================== %

function spacing = calc_spacing(length_clear, n)
    if n <= 1
        spacing = 0;
    else
        spacing = length_clear / (n - 1);
    end
end



function [PI_Pnb, PI_Mnb, Alpha_PI_Pnb, Alpha_PI_Mnb] = calculate_design_strength(Pnb, Mnb, count, fy, Rebar_YieldStrain, Pn0)
    PI_Pnb = zeros(size(Pnb));
    PI_Mnb = zeros(size(Mnb));
    
    for i = 1:length(count)
        x = count(i);
        if fy <= 400, et_limit_tension = -0.005; else, et_limit_tension = -2.5 * Rebar_YieldStrain; end
        
        if x <= et_limit_tension, PI = 0.85;
        elseif x < -Rebar_YieldStrain, PI = 0.65 + 0.20 * ((-x - Rebar_YieldStrain) / (-et_limit_tension - Rebar_YieldStrain));
        else, PI = 0.65;
        end
        PI_Pnb(i) = PI * Pnb(i);
        PI_Mnb(i) = PI * Mnb(i);
    end
    
    Alpha = 0.8; % 띠철근 기둥
    PI_Pn_max = Alpha * PI_Pnb(1);
    
    Alpha_PI_Pnb = min(PI_Pnb, PI_Pn_max);
    Alpha_PI_Mnb = PI_Mnb;
end

% [fileName, pathName] = uiputfile('pm_dataset.mat', '데이터 저장 위치를 선택하세요'); if ~isequal(fileName, 0), save(fullfile(pathName, fileName), 'Column_Mdata', 'Column_Data', '-v7.3'); end