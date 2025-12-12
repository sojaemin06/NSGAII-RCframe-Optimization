clc; clear all;
rng('shuffle');

% 공통 설정
aggregate = 25;   % 굵은골재 [mm]
Cover_size = 40;
Stirrup_size = 10;
Es = 200000;
MainRebar_range = 16:1:32;
rho_range = 0.01:0.005:0.08;


while true
    % --------------------------- 단면 및 철근 수 설정 ---------------------------
    MainRebar_size = randsample(MainRebar_range, 1);
    Area_MainRebar = (pi * MainRebar_size^2) / 4;

    b_range = 300:100:600; % 최소폭규정(세장비 15이하)
    b = randsample(b_range, 1);
    h_range = b:100:b*2; % 장변 1~2배 설정
    h = randsample(h_range, 1);
    Ag = b * h;

    rho = randsample(rho_range, 1);
    total_rebar_area = Ag * rho;
    required_rebar = ceil(total_rebar_area / Area_MainRebar);
    if mod(required_rebar, 2) ~= 0
        required_rebar = required_rebar + 1; % 철근수 4면배근 조건
    end
    rho = (required_rebar * Area_MainRebar) / Ag;
    total_rebar_area = Ag *rho;

    % --------------------------- 철근 배근 ---------------------------
    corner_rebars = 4;
    remain = required_rebar - corner_rebars;
    side_rebars = [0 0 0 0];  % [전, 후, 좌, 우]

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

    % --------------------------- 순간격 검토 ---------------------------
    % 최소 순간격 기준 → 중심간 간격으로 환산
    rebar_clear_spacing = max([40, MainRebar_size * 1.5, aggregate * 4/3]);
    min_center_spacing = rebar_clear_spacing + MainRebar_size;

    % 유효 배근 길이
    clear_cover = Cover_size + Stirrup_size + MainRebar_size / 2;
    effective_h = h - 2 * clear_cover;
    effective_b = b - 2 * clear_cover;

    % 중심간 간격 계산
    n_h = side_rebars(1)+2; % 전+코너2개
    n_b = side_rebars(3)+2; % 좌+코너2개
    spacing_h = calc_spacing(effective_h, n_h);
    spacing_b = calc_spacing(effective_b, n_b);

    % 조건 확인
    if spacing_h >= min_center_spacing && spacing_b >= min_center_spacing
        break;  % 조건 만족 → 루프 탈출
    end
end

% --------------------------- 출력 ---------------------------
fprintf('\n===== 결과 출력 =====\n');
fprintf('단면: b = %d mm, h = %d mm\n', b, h);
fprintf('Main Rebar size: %d mm\n', MainRebar_size);
fprintf('총 철근 수: %d개 (corner + side)\n', required_rebar);
fprintf('철근비: %.4f\n', rho);
fprintf('면별 철근 수 [전, 후, 좌, 우]: [%d %d %d %d]\n', side_rebars);
fprintf('전후 방향 중심간 간격: %.1f mm → 기준 %.1f mm\n', spacing_h, min_center_spacing);
fprintf('좌우 방향 중심간 간격: %.1f mm → 기준 %.1f mm\n', spacing_b, min_center_spacing);




   



fck_range = [21, 24, 27];
fy_range = [300, 350, 400, 500];
centroid = h/2;
d_prime_1 = h-(Cover_size+Stirrup_size+(MainRebar_size/2));  % 최외각철근까지 거리
% Stirrup_verticle_range = 100:10:min([MainRebar_size*16, Stirrup_size*48, h]);
% Stirrup_verticle_size = randsample(Stirrup_verticle_range, 1);

fck = randsample(fck_range ,1);  % 콘크리트압축강도설정
fy = randsample(fy_range ,1);  % 철근항복강도설정
Rebar_YieldStrain = fy/Es;  % 철근항복변형률계산


count = 0.003:-0.00001:-100*Rebar_YieldStrain;
d_prime = zeros(n_h, 1);
esi = zeros(n_h, length(count));
fsi = zeros(n_h, length(count));
nsi = zeros(n_h, length(count));



area=zeros(1,side_rebars(1));
for r = 1:side_rebars(1)
    area(:,r) = Area_MainRebar * 2;
end
area_rabar_cols = [Area_MainRebar*(side_rebars(3)+2), area, Area_MainRebar*(side_rebars(3)+2)];
        



for i = 1:length(count)
    x = count(i);
    Neutral_depth = (0.003/(0.003-x))*d_prime_1;  % 중립축깊이
    if fck < 28
        Beta = 0.85;
    else
        Beta = max((0.85 - (fck - 28)*0.05/7),0.65);
    end
    a = ceil(Beta*Neutral_depth);  % 등가직사각형 높이
    
    
    if a > h
        a = h;
    end
    
    for j = 1:n_h
        
        d_prime(j,1)=h-clear_cover-spacing_h*(j-1);
        esi(j,i)=((Neutral_depth-d_prime(j,1))/Neutral_depth)*0.003;
        if abs(esi(j,i))>Rebar_YieldStrain
            fsi(j,i)=sign(esi(j,i))*fy;
        else
            fsi(j,i)=Es*esi(j,i);
        end
        if d_prime(j,1) < a
            nsi(j,i)=(    (fsi(j,i)-0.85*fck)*area_rabar_cols(1,j)   )*10^-3;
        else
            nsi(j,i)=(     fsi(j,i)*area_rabar_cols(1,j)    )*10^-3;
        end
        m_distance(j,1)=centroid-d_prime(j,1);
    end
    Pnb(1,i)=(0.85*fck*a*b*10^-3)+sum(nsi(:,i));
    Mnb(1,i)=(     (0.85*fck*a*b*(centroid-a/2)*10^-3)+sum(nsi(:,i) .* m_distance(:,1))     )*10^-3;



    Pn0=(     0.85*fck*(Ag-total_rebar_area)+fy*total_rebar_area       )*10^-3;
    Pnb(1,1)=Pn0;
    Mn0=0;
    Mnb(1,1)=Mn0;





    if fy <= 400
        if x <= Rebar_YieldStrain
            PI = 0.65;
        elseif x <= 0.005
            PI = 0.65 + ((x-Rebar_YieldStrain)/(0.005-Rebar_YieldStrain))*(0.85-0.65);
        else
            PI = 0.85;
        end
    end
    if fy > 400
        if x <= Rebar_YieldStrain
            PI = 0.65;
        elseif x <= 2.5*Rebar_YieldStrain
            PI = 0.65 + ((x-Rebar_YieldStrain)/(2.5*Rebar_YieldStrain-Rebar_YieldStrain))*(0.85-0.65);
        else
            PI = 0.85;
        end
    end



    PI_Pnb=PI*Pnb;
    PI_Mnb=PI*Mnb;

    
    Alpha=0.8;
    PI_Pnb_max = Alpha * PI_Pnb(1,1);
    if PI_Pnb >= PI_Pnb_max
        Alpha_PI_Pnb(1,i) = PI_Pnb_max;
        Alpha_PI_Mnb(1,i) = PI_Mnb(1,i);
    else
        Alpha_PI_Pnb(1,i) = PI_Pnb(1,i);
        Alpha_PI_Mnb(1,i) = PI_Mnb(1,i);
    end


end

  







% ======================================================================= %
% ===== 특정 지점 찾기 부분 ===================================== %
% ======================================================================= %
target_strains = [-Rebar_YieldStrain, 0, Rebar_YieldStrain];
special_points_M = zeros(1, 3);
special_points_P = zeros(1, 3);

for k = 1:length(target_strains)
    [~, idx] = min(abs(count - target_strains(k)));
    special_points_M(k) = Mnb(idx);
    special_points_P(k) = Pnb(idx);
end


% ======================================================================= %
% ===== 그래프 그리기 부분 (범례 핸들 지정) ============================= %
% ======================================================================= %
figure;
hold on;

% 각 plot의 핸들을 변수(h1, h2, ...)에 저장합니다.
h1 = plot(Mnb, Pnb, 'b-', 'LineWidth', 2);
h2 = plot(PI_Mnb, PI_Pnb, 'r--', 'LineWidth', 2);
h3 = plot(Alpha_PI_Mnb, Alpha_PI_Pnb, 'g-', 'LineWidth', 1.5);

line_X_coords = [zeros(1, 3); special_points_M];
line_Y_coords = [zeros(1, 3); special_points_P];
h4 = plot(line_X_coords, line_Y_coords, 'k--', 'LineWidth', 1.2);
h5 = plot(special_points_M, special_points_P, 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 8);

grid on;
ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';
xlabel('Moment (kN·m)', 'FontSize', 12);
ylabel('Axial Force (kN)', 'FontSize', 12);
title('P-M Interaction Diagram for Rectangular Column', 'FontSize', 14);

% legend 함수에 핸들(h1~h5)을 직접 전달하여 아이콘을 명확하게 합니다.
% 3개의 점선은 대표로 첫 번째 핸들 h4(1)만 사용합니다.
legend([h1, h2, h3, h4(1), h5], ...
       {'Nominal Strength ($$P_n-M_n$$)', ...
        'Design Strength ($$\phi P_n - \phi M_n$$)', ...
        'Design Strength with $$P_{n,max}$$',...
        'Load Eccentricity Lines ($$e = M/P$$)',...
        'Special Points ($$\epsilon_s = -\epsilon_y, 0, \epsilon_y$$)'}, ...
       'Location', 'best', 'Interpreter', 'latex');

hold off;




function spacing = calc_spacing(length_clear, n)
    if n <= 1
        spacing = 0;
    else
        spacing = length_clear / (n - 1);
    end
end


