function f = STM_Beam_Function_02(X)
            
            b = X(1);
            d_actual = X(2); 
            rho_s_T = X(3);
            rho_s_C = X(4);
            fck = X(5);
            fy = X(6);
            dimension = X(7);
            PI_size = X(8);
            stirrup = X(9);
            conblock = X(10);
            ConBlock = X(11);
            total_size = X(12);
            space = X(13);
            strup_space = X(14);
            SG_steel = X(15);
            Cc = X(16);
            Cs = X(17);
            h = X(18); 
            N_n = X(19);
            % <<< 추가: 생성기에서 CO2 계산 상수들을 받아옴 >>>
            DENSITY_CONCRETE_KG_M3 = X(20);
            DENSITY_STEEL_KG_M3 = X(21);
            ECF_CONCRETE = X(22);
            ECF_STEEL = X(23);
            
            % % SECTION ANALYSIS
            
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

            Es=200000;
            
            d_t = d_actual + (space + dimension)*(0.5*N_n-0.5);
            d_prime = (dimension/2)+total_size;

            c = ((rho_s_T-rho_s_C)*d_actual*fy) / (Eta*0.85*fck*Beta);
            es_T = ecu*(d_t/c - 1);
            es_C = ecu*(1 - d_prime/c);
            
            if es_C >= fy/Es
                M = (rho_s_T-rho_s_C)*b*d_actual*fy*(d_actual - Beta*c/2)/10^6 + rho_s_C*b*d_actual*fy*(d_actual-d_prime)/10^6;
            elseif es_C < fy/Es
                c = ((-660*rho_s_C*b*d_actual+rho_s_T*fy*b*d_actual)+sqrt(((660*rho_s_C*b*d_actual-rho_s_T*b*d_actual*fy)^2)+4*Beta*Eta*0.85*fck*b*660*d_prime*rho_s_C*b*d_actual))/(2*Beta*Eta*0.85*fck*b);
                es_T = ecu*(d_t/c - 1);
                es_C = ecu*(1 - d_prime/c);
                fs_prime = Es*es_C;
                M = (rho_s_T*fy-rho_s_C*fs_prime)*b*d_actual*(d_actual - Beta*c/2)/10^6 + rho_s_C*b*d_actual*fy*(d_actual-d_prime)/10^6;
            end
            
            if es_T <= 0.002
                Pi = 0.65;
            elseif es_T < 0.005
                Pi = 0.65+(es_T-0.002)*(200/3);
            else 
                Pi = 0.85;
            end
            
            PiM = Pi*M;
            
            area_stirrup = (pi * stirrup^2) / 4;
            V_c = 1/6*sqrt(fck)*b*d_actual;
            V_s = (area_stirrup*2*fy*d_actual)/strup_space;
            PiVn = 0.75*(V_c + V_s)*10^-3;

            % --- 비용(Cost) 계산 ---
            Cost1 = (b*h - ((rho_s_T+rho_s_C)*b*d_actual))*10^(-6)*Cc + ((rho_s_T+rho_s_C)*b*d_actual)*10^(-6)*SG_steel*Cs;
            Cost2 = (((b-2*PI_size)+(h-2*PI_size))*2*10^(-3)) * ((stirrup*1/2)^2*pi*10^(-6)) * SG_steel * Cs * (1000/strup_space); % 단위 m당 비용으로 환산
            Cost = Cost1+Cost2;

            % --- <<< 추가: 내재탄소(CO2) 계산 >>> ---
            % 주철근 및 콘크리트 CO2
            total_main_steel_area_mm2 = (rho_s_T + rho_s_C) * b * d_actual;
            concrete_area_mm2 = (b * h) - total_main_steel_area_mm2;
            
            concrete_vol_per_m = concrete_area_mm2 * 10^(-6); % m^2 * 1m
            main_steel_vol_per_m = total_main_steel_area_mm2 * 10^(-6); % m^2 * 1m

            concrete_mass_per_m = concrete_vol_per_m * DENSITY_CONCRETE_KG_M3;
            main_steel_mass_per_m = main_steel_vol_per_m * DENSITY_STEEL_KG_M3;
            
            CO2_1 = (concrete_mass_per_m * ECF_CONCRETE) + (main_steel_mass_per_m * ECF_STEEL);

            % 스트럽(전단철근) CO2
            stirrup_len_one_m = ((b-2*PI_size)+(h-2*PI_size))*2*10^(-3); % 단일 스트럽의 길이 (m)
            stirrup_area_m2 = ((stirrup*1/2)^2*pi*10^(-6)); % 스트럽 철근의 단면적 (m^2)
            stirrup_vol_per_m = stirrup_len_one_m * stirrup_area_m2 * (1000/strup_space); % 단위 m당 스트럽의 체적
            
            stirrup_mass_per_m = stirrup_vol_per_m * DENSITY_STEEL_KG_M3;
            
            CO2_2 = stirrup_mass_per_m * ECF_STEEL;
            
            CO2 = CO2_1 + CO2_2; % 단위 m당 총 CO2 배출량

            % --- <<< 추가: 단면별 단위중량(UnitWeight) 계산 >>> ---
            total_steel_vol_per_m = main_steel_vol_per_m + stirrup_vol_per_m;
            total_concrete_vol_per_m = (b*h*10^(-6)) - total_steel_vol_per_m;
        
            total_mass_per_m = (total_concrete_vol_per_m * DENSITY_CONCRETE_KG_M3) + (total_steel_vol_per_m * DENSITY_STEEL_KG_M3);
            UnitWeight = (total_mass_per_m * 9.81) / 1000; % 최종 단위중량 (kN/m^3)

            % <<< 변경: 최종 반환 값에 Cost와 CO2를 포함하여 출력 >>>
            f = [b,d_actual,h,PiM,M,rho_s_T,rho_s_C,PiVn,stirrup,strup_space,fck,fy,Cost,CO2,UnitWeight];
end