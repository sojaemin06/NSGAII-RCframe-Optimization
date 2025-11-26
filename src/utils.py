
import pandas as pd
import openseespy.opensees as ops

def load_pm_data_for_column(h5_file, column_index):
    """h5py(.mat) 파일에서 특정 기둥 단면의 P-M 상호작용 데이터를 로드하는 함수."""
    try:
        ref = h5_file['Column_Mdata'][0, column_index]; pm_data = h5_file[ref][()]
        return pd.DataFrame(pm_data, columns=['Mnb_z','Pnb_z','Mnb_y','Pnb_y','Alpha_PI_Mnb_z','Alpha_PI_Pnb_z','Alpha_PI_Mnb_y','Alpha_PI_Pnb_y'])
    except Exception: return pd.DataFrame()

def get_pm_capacity_from_df(slope, pm_df, axis='z'):
    """P-M 상호작용 데이터와 현재 하중경로(slope)를 이용해 공칭강도(Pn, Mn)를 계산하는 함수."""
    if pm_df.empty: return 0.0, 0.0
    moment_col, axial_col = (f'Alpha_PI_Mnb_{axis}', f'Alpha_PI_Pnb_{axis}')
    design_m, design_p = pm_df[moment_col].values, pm_df[axial_col].values
    curve_slopes = abs(design_p) / (abs(design_m) + 1e-9)
    for i in range(len(curve_slopes) - 1):
        if curve_slopes[i] >= slope >= curve_slopes[i+1]:
            m1, p1, m2, p2 = design_m[i], design_p[i], design_m[i+1], design_p[i+1]
            if abs(m2 - m1) < 1e-9: continue

            # 두 점을 잇는 직선의 방정식: P = a*M + b
            a1 = (p2 - p1) / (m2 - m1); b1 = p1 - a1 * m1

            # 하중 경로선(P = slope*M)과의 교점 찾기
            if abs(a1 - slope) < 1e-9: continue
            Mn = -b1 / (a1 - slope)
            Pn = slope * Mn
            return Pn, Mn # 축력(Pn)과 모멘트(Mn)를 함께 반환
    return 0.0, 0.0

def get_precalculated_strength(element_type, index, col_df, beam_df):
    """단면 데이터프레임에서 사전 계산된 공칭강도(Pn, Vn 등)를 가져오는 함수."""
    strengths = {}
    try:
        if element_type == 'Column':
            row = col_df.iloc[index]; strengths.update({'Pn': row['PI_Pn_max'], 'Vn_y': row['PI_Vn_y'], 'Vn_z': row['PI_Vn_z']})
        elif element_type == 'Beam':
            row = beam_df.iloc[index]; strengths.update({'Pn': float('inf'), 'Vn_y': float('inf'), 'Vn_z': row['PiVn'], 'Mn_y': float('inf'), 'Mn_z': row['PiM']})
    except IndexError: return {'Pn': 0, 'Vn_y': 0, 'Vn_z': 0, 'Mn_y': 0, 'Mn_z': 0}
    return strengths

def extract_local_element_forces(column_elem_ids, beam_elem_ids):
    """OpenSees 해석 후, 모든 부재의 로컬 좌표계 부재력을 추출하여 DataFrame으로 반환하는 함수."""
    all_forces = []
    def append_column_forces(eid):
        try:
            f=ops.eleResponse(eid,'localForce')
            all_forces.append({'ElementType':'Column','ElementID':eid,'Node':'i','Axial (kN)':-f[0],'Shear-y (kN)':f[2],'Shear-z (kN)':f[1],'Torsion (kNm)':f[3],'Moment-y (kNm)':f[5],'Moment-z (kNm)':-f[4]})
            all_forces.append({'ElementType':'Column','ElementID':eid,'Node':'j','Axial (kN)':f[6],'Shear-y (kN)':-f[8],'Shear-z (kN)':-f[7],'Torsion (kNm)':-f[9],'Moment-y (kNm)':-f[11],'Moment-z (kNm)':f[10]})
        except: pass
    def append_beam_forces(eid):
        try:
            f=ops.eleResponse(eid,'localForce')
            all_forces.append({'ElementType':'Beam','ElementID':eid,'Node':'i','Axial (kN)':f[0],'Shear-y (kN)':-f[1],'Shear-z (kN)':-f[2],'Torsion (kNm)':f[3],'Moment-y (kNm)':f[4],'Moment-z (kNm)':-f[5]})
            all_forces.append({'ElementType':'Beam','ElementID':eid,'Node':'j','Axial (kN)':f[6],'Shear-y (kN)':f[7],'Shear-z (kN)':f[8],'Torsion (kNm)':f[9],'Moment-y (kNm)':-f[10],'Moment-z (kNm)':f[11]})
        except: pass
    for eid in column_elem_ids:append_column_forces(eid)
    for eid in beam_elem_ids:append_beam_forces(eid)
    if not all_forces: return pd.DataFrame()
    df_all = pd.DataFrame(all_forces)
    df_all['ElementType'] = pd.Categorical(df_all['ElementType'],categories=['Column','Beam'],ordered=True)
    df_all = df_all.sort_values(['ElementType','ElementID'])
    df_max = (df_all.groupby(['ElementType','ElementID'], observed=True).agg({col:lambda x: x.iloc[0] if abs(x.iloc[0]) > abs(x.iloc[1]) else x.iloc[1] for col in ['Axial (kN)','Shear-y (kN)','Shear-z (kN)','Torsion (kNm)','Moment-y (kNm)','Moment-z (kNm)']}).reset_index())
    return df_max
