import openseespy.opensees as ops
import pandas as pd
import numpy as np
import math
import sys
import os
import src.config as cfg 
from src.utils import *

class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

def extract_local_element_forces(column_elem_ids, beam_elem_ids):
    all_forces = []
    def append_column_forces(eid):
        try:
            f=ops.eleResponse(eid,'localForce')
            all_forces.append({'ElementType':'Column','ElementID':eid,'Node':'i','Axial (kN)':-f[0],'Shear-y (kN)':f[1],'Shear-z (kN)':f[2],'Torsion (kNm)':f[3],'Moment-y (kNm)':f[4],'Moment-z (kNm)':f[5]})
            all_forces.append({'ElementType':'Column','ElementID':eid,'Node':'j','Axial (kN)':f[6],'Shear-y (kN)':-f[7],'Shear-z (kN)':-f[8],'Torsion (kNm)':-f[9],'Moment-y (kNm)':-f[10],'Moment-z (kNm)':-f[11]})
        except: pass
    def append_beam_forces(eid):
        try:
            f=ops.eleResponse(eid,'localForce')
            all_forces.append({'ElementType':'Beam','ElementID':eid,'Node':'i','Axial (kN)':f[0],'Shear-y (kN)':f[1],'Shear-z (kN)':f[2],'Torsion (kNm)':f[3],'Moment-y (kNm)':f[4],'Moment-z (kNm)':f[5]})
            all_forces.append({'ElementType':'Beam','ElementID':eid,'Node':'j','Axial (kN)':f[6],'Shear-y (kN)':f[7],'Shear-z (kN)':f[8],'Torsion (kNm)':f[9],'Moment-y (kNm)':-f[10],'Moment-z (kNm)':f[11]})
        except: pass
    for eid in column_elem_ids:append_column_forces(eid)
    for eid in beam_elem_ids:append_beam_forces(eid)
    if not all_forces: return pd.DataFrame()
    df_all = pd.DataFrame(all_forces)
    df_all['ElementType'] = pd.Categorical(df_all['ElementType'],categories=['Column','Beam'],ordered=True)
    df_all = df_all.sort_values(['ElementType','ElementID'])
    return (df_all.groupby(['ElementType','ElementID'], observed=True).agg({col:lambda x: x.iloc[0] if abs(x.iloc[0]) > abs(x.iloc[1]) else x.iloc[1] for col in ['Axial (kN)','Shear-y (kN)','Shear-z (kN)','Torsion (kNm)','Moment-y (kNm)','Moment-z (kNm)']}).reset_index())

def build_model_for_section(col_indices, col_rotations, beam_indices, col_map, beam_map, beam_sections, column_sections):
    ops.wipe(); ops.model('basic', '-ndm', 3, '-ndf', 6)
    E, nu = 2.5791e7, 0.167; G = E / (2 * (1 + nu))
    num_locs = len(cfg.COLUMN_LOCATIONS); node_map = {}; node_id = 1
    for k in range(cfg.FLOORS + 1):
        for i, (x, y) in enumerate(cfg.COLUMN_LOCATIONS):
            ops.node(node_id, x, y, k * cfg.H); node_map[(k, i)] = node_id
            if k == 0: ops.fix(node_id, 1, 1, 1, 1, 1, 1)
            node_id += 1
    ops.geomTransf('PDelta', 1, 1, 0, 0); ops.geomTransf('PDelta', 2, 0, 1, 0); ops.geomTransf('PDelta', 3, 0, 0, 1)
    for k in range(1, cfg.FLOORS + 1):
        m = node_map[(k, 0)]
        for i in range(1, num_locs): ops.equalDOF(m, node_map[(k, i)], 1, 2, 6)
    c_ids, b_ids, eid_cnt = [], [], 1; n_cols = num_locs * cfg.FLOORS
    for k in range(cfg.FLOORS):
        for i in range(num_locs):
            g_idx = col_map[k*num_locs+i+1]; tag = (col_rotations[g_idx]+1) if len(col_rotations)>0 else 1
            sec = col_indices[g_idx]; b, h = column_sections[sec]; A, Iz, Iy = b*h, 0.7*b*h**3/12, 0.7*h*b**3/12
            ops.element('elasticBeamColumn', eid_cnt, node_map[(k,i)], node_map[(k+1,i)], A, E, G, Iz+Iy, Iy, Iz, tag)
            c_ids.append(eid_cnt); eid_cnt += 1
    for k in range(1, cfg.FLOORS + 1):
        for i, (l1, l2) in enumerate(cfg.BEAM_CONNECTIONS):
            g_idx = beam_map[n_cols+(k-1)*len(cfg.BEAM_CONNECTIONS)+i+1]; sec = beam_indices[g_idx]
            b, h = beam_sections[sec]; A, Iz, Iy = b*h, 0.35*b*h**3/12, 0.35*h*b**3/12
            ops.element('elasticBeamColumn', eid_cnt, node_map[(k,l1)], node_map[(k,l2)], A, E, G, Iz+Iy, Iy, Iz, 3)
            b_ids.append(eid_cnt); eid_cnt += 1
    return c_ids, b_ids, node_map

def evaluate(individual, DL, LL, h5_file, patterns_by_floor, col_map, beam_map, beam_sections, column_sections, beam_sections_df, column_sections_df, beam_lengths, chromosome_structure, num_columns, num_beams):
    fail = {"cost": float('inf'), "co2": float('inf'), "violation": 9.9, "max_drift_ratio": float('inf'), "absolute_margins": {"strength": 9.9, "drift": 9.9, "wind_disp": 9.9, "deflection": 9.9, "scwb": 9.9}}
    num_locs = len(cfg.COLUMN_LOCATIONS); len_sec, len_rot = chromosome_structure['col_sec'], chromosome_structure['col_rot']
    col_idx, col_rot, beam_idx = individual[:len_sec], individual[len_sec:len_sec+len_rot], individual[len_sec+len_rot:]
    total_w, story_w = 0.0, [0.0]*cfg.FLOORS
    for i in range(num_columns):
        s = col_idx[col_map[i+1]]; b, h = column_sections[s]; w = b*h*cfg.H*column_sections_df.iloc[s]['UnitWeight']
        total_w += w; story_w[i//num_locs] += w
    for i in range(num_beams):
        s = beam_idx[beam_map[num_columns+i+1]]; b, h = beam_sections[s]; w = b*h*beam_lengths[i%len(cfg.BEAM_CONNECTIONS)]*beam_sections_df.iloc[s]['UnitWeight']
        total_w += w; story_w[i//len(cfg.BEAM_CONNECTIONS)] += w
    slab = cfg.FLOOR_AREA * cfg.DL_AREA_LOAD; total_w += slab*cfg.FLOORS
    for f in range(cfg.FLOORS): story_w[f] += slab
    Ta = cfg.PERIOD_CT * ((cfg.FLOORS*cfg.H)**cfg.PERIOD_X)
    Cs = max(min(cfg.SDS/(cfg.R_COEFF/cfg.I_FACTOR), cfg.SD1/(Ta*(cfg.R_COEFF/cfg.I_FACTOR))), 0.01, 0.044*cfg.SDS*cfg.I_FACTOR)
    base_shear = Cs * total_w
    try: c_ids, b_ids, node_map = build_model_for_section(col_idx, col_rot, beam_idx, col_map, beam_map, beam_sections, column_sections)
    except: return fail
    defl_ratios = []
    for i, eid in enumerate(b_ids):
        L, s = beam_lengths[i%len(cfg.BEAM_CONNECTIONS)], beam_idx[beam_map[num_columns+i+1]]; b, h = beam_sections[s]
        ll, trib = LL.get((i//len(cfg.BEAM_CONNECTIONS))+1, 2.0), cfg.BEAM_TRIBUTARY_WIDTHS[i%len(cfg.BEAM_CONNECTIONS)]
        w_DL, w_LL = (beam_sections_df.iloc[s]['UnitWeight']*b*h + DL*trib), ll*trib; E, I_eff = 2.5791e7, 0.35*(b*h**3)/12
        defl = ((2.0/(1+50*beam_sections_df.iloc[s].get('rho_c',0.0)))*(5*(w_DL+0.5*w_LL)*L**4/(384*E*I_eff))) + (5*w_LL*L**4/(384*E*I_eff))
        defl_ratios.append(defl/(L/240.0))
    max_defl = max(defl_ratios) if defl_ratios else 0.0
    mass_nodes = [node_map[(k,0)] for k in range(1, cfg.FLOORS+1)]
    for k in range(cfg.FLOORS): ops.mass(mass_nodes[k], story_w[k]/9.81, story_w[k]/9.81, 0.01*story_w[k]/9.81, 1e-9, 1e-9, 1e-9)
    phi_x, phi_y = [(k+1)*cfg.H for k in range(cfg.FLOORS)], [(k+1)*cfg.H for k in range(cfg.FLOORS)]
    try:
        eig = ops.eigen(1)
        if eig and eig[0]>1e-9:
            ex, ey = [ops.nodeEigenvector(n,1,1) for n in mass_nodes], [ops.nodeEigenvector(n,1,2) for n in mass_nodes]
            if any(abs(v)>1e-9 for v in ex): phi_x = ex
            if any(abs(v)>1e-9 for v in ey): phi_y = ey
    except: pass
    seis_x = [ (story_w[f]*phi_x[f])/sum(story_w[i]*phi_x[i] for i in range(cfg.FLOORS))*base_shear for f in range(cfg.FLOORS)]
    seis_y = [ (story_w[f]*phi_y[f])/sum(story_w[i]*phi_y[i] for i in range(cfg.FLOORS))*base_shear for f in range(cfg.FLOORS)]
    forces, ok, dr_x, dr_y, w_x, w_y = [], True, [], [], [], []
    ops.timeSeries('Linear', 1); ops.system('ProfileSPD'); ops.numberer('RCM'); ops.constraints('Transformation'); ops.integrator('LoadControl', 1.0); ops.algorithm('Newton'); ops.analysis('Static')
    for i, (name, fact) in enumerate(cfg.LOAD_COMBINATIONS):
        tag, converged = i+1, False
        with SuppressOutput():
            try:
                ops.reset(); ops.pattern('Plain', tag, 1)
                for b_idx, eid in enumerate(b_ids):
                    s = beam_idx[beam_map[num_columns+b_idx+1]]; b, h = beam_sections[s]; floor = (b_idx//len(cfg.BEAM_CONNECTIONS))+1
                    conn, trib = b_idx%len(cfg.BEAM_CONNECTIONS), cfg.BEAM_TRIBUTARY_WIDTHS[b_idx%len(cfg.BEAM_CONNECTIONS)]
                    w = (b*h*beam_sections_df.iloc[s]['UnitWeight'] + DL*trib)*fact["DL"]
                    if conn in patterns_by_floor.get(floor, set()): w += LL.get(floor, 2.0)*trib*fact["LL"]
                    if abs(w)>1e-6: ops.eleLoad('-ele', eid, '-type', '-beamUniform', 0, -w)
                for c_idx, eid in enumerate(c_ids):
                    s = col_idx[col_map[c_idx+1]]; w = column_sections[s][0]*column_sections[s][1]*cfg.H*column_sections_df.iloc[s]['UnitWeight']*fact["DL"]
                    n1, n2 = ops.eleNodes(eid); ops.load(n1, 0,0,-w/2,0,0,0); ops.load(n2, 0,0,-w/2,0,0,0)
                qh = 0.000613 * 2.01*((cfg.FLOORS*cfg.H/cfg.ZG)**(2/cfg.ALPHA)) * cfg.KZT * cfg.KD * (cfg.BASIC_WIND_SPEED**2)
                for k in range(cfg.FLOORS):
                    qz = 0.000613 * 2.01*(((k+1)*cfg.H/cfg.ZG)**(2/cfg.ALPHA)) * cfg.KZT * cfg.KD * (cfg.BASIC_WIND_SPEED**2)
                    p = (qz * cfg.G_FACTOR * cfg.CP_WINDWARD) + (qh * cfg.G_FACTOR * abs(cfg.CP_LEEWARD))
                    Fx, Fy = (seis_x[k]*fact["Ex"] + p*cfg.BUILDING_WIDTH_Y*cfg.H*fact["Wx"])/num_locs, (seis_y[k]*fact["Ey"] + p*cfg.BUILDING_WIDTH_X*cfg.H*fact["Wy"])/num_locs
                    for loc in range(num_locs):
                        n = node_map.get((k+1, loc))
                        if n: ops.load(n, Fx, Fy, 0,0,0,0)
                for solver in ['BandGeneral', 'UmfPack']:
                    if converged: break
                    for algo in ['Newton', 'ModifiedNewton']:
                        ops.system(solver); ops.algorithm(algo)
                        if ops.analyze(1) == 0: converged = True; break
                if converged:
                    if name == "ASCE-S-E1":
                        for k in range(1, cfg.FLOORS+1): dr_x.append(abs(ops.nodeDisp(node_map[(k,0)],1)-ops.nodeDisp(node_map[(k-1,0)],1))*(cfg.CD_FACTOR/(0.7*cfg.I_FACTOR))/cfg.H)
                    elif name == "ASCE-S-E5":
                        for k in range(1, cfg.FLOORS+1): dr_y.append(abs(ops.nodeDisp(node_map[(k,0)],2)-ops.nodeDisp(node_map[(k-1,0)],2))*(cfg.CD_FACTOR/(0.7*cfg.I_FACTOR))/cfg.H)
                    elif name == "ASCE-S-W1":
                        for k in range(1, cfg.FLOORS+1): w_x.append(abs(ops.nodeDisp(node_map[(k,0)],1)))
                    elif name == "ASCE-S-W3":
                        for k in range(1, cfg.FLOORS+1): w_y.append(abs(ops.nodeDisp(node_map[(k,0)],2)))
            except: pass
        if not converged: ok = False; break
        df = extract_local_element_forces(c_ids, b_ids)
        if df.empty: ok = False; break
        df['Combo'] = name; forces.append(df)
        try:
            with SuppressOutput():
                ops.remove('loadPattern', tag)
        except:
            pass
    if not ok or not forces: return fail
    df_all = pd.concat(forces, ignore_index=True)
    max_f = df_all.groupby(['ElementType', 'ElementID'], observed=True).agg({f:lambda x: x.iloc[x.abs().argmax()] for f in ['Axial (kN)', 'Shear-y (kN)', 'Shear-z (kN)', 'Torsion (kNm)', 'Moment-y (kNm)', 'Moment-z (kNm)']}).reset_index().set_index('ElementID')
    s_ratios = []
    for eid in sorted(list(set(c_ids) | set(b_ids))):
        try:
            row = max_f.loc[eid]; p, vy, vz, my, mz = abs(row['Axial (kN)']), abs(row['Shear-y (kN)']), abs(row['Shear-z (kN)']), abs(row['Moment-y (kNm)']), abs(row['Moment-z (kNm)'])
            if row['ElementType'] == 'Column':
                s = col_idx[col_map[c_ids.index(eid)+1]]; pm = load_pm_data_for_column(h5_file, s); str_d = get_precalculated_strength('Column', s, column_sections_df, beam_sections_df)
                pnz, mnz = get_pm_capacity_from_df(p/(mz+1e-9), pm, axis='z'); pny, mny = get_pm_capacity_from_df(p/(my+1e-9), pm, axis='y'); dz, dy = mz/mnz, my/mny
                s_ratios.append(max(p/pnz, p/pny, vy/str_d['Vn_y'], vz/str_d['Vn_z'], dz, dy, (dz**1.5)+(dy**1.5)))
            else:
                s = beam_idx[beam_map[num_columns+b_ids.index(eid)+1]]; str_d = get_precalculated_strength('Beam', s, column_sections_df, beam_sections_df)
                s_ratios.append(max(vy/str_d['Vn_z'], mz/str_d['Mn_z']))
        except: s_ratios.append(9.9)
    max_s, mean_s = max(s_ratios), np.mean([r for r in s_ratios if r < 9.0])
    act_dr = max(max(dr_x) if dr_x else [0.0], max(dr_y) if dr_y else [0.0])
    act_w = max(w_x[-1]/(cfg.FLOORS*cfg.H/400.0) if w_x else 0.0, w_y[-1]/(cfg.FLOORS*cfg.H/400.0) if w_y else 0.0)
    scwb = []
    for k in range(1, cfg.FLOORS + 1):
        for i in range(num_locs):
            mb, mc = 0.0, 0.0
            for b_idx, (u, v) in enumerate(cfg.BEAM_CONNECTIONS):
                if u == i or v == i: mb += beam_sections_df.iloc[beam_idx[beam_map[num_columns+(k-1)*len(cfg.BEAM_CONNECTIONS)+b_idx+1]]]['PiM']
            for c_idx in [ (k-1)*num_locs+i, k*num_locs+i ]:
                if 0 <= c_idx < num_columns:
                    pm = load_pm_data_for_column(h5_file, col_idx[col_map[c_idx+1]]); _, mn0 = get_pm_capacity_from_df(0, pm, axis='z'); mc += mn0
            if mb > 0: scwb.append(1.2*mb/(mc+1e-9))
    act_scwb = max(scwb) if scwb else 0.0
    cost, co2 = 0, 0
    for i in range(num_columns):
        s = col_idx[col_map[i+1]]; b, h = column_sections[s]; row = column_sections_df.iloc[s]; v_c = b*h*cfg.H*(1-row['rho']); mass_s = b*h*cfg.H*row['rho']*7.85
        cost += v_c*80000 + mass_s*1000000 + 2*(b+h)*cfg.H*cfg.FORMWORK_UNIT_COST; co2 += v_c*2400*0.15 + mass_s*1000*1.99
    for i in range(num_beams):
        s = beam_idx[beam_map[num_columns+i+1]]; b, h = beam_sections[s]; L = beam_lengths[i%len(cfg.BEAM_CONNECTIONS)]; row = beam_sections_df.iloc[s]; v_c = b*h*L*(1-(row.get('rho_t',0.01)+row.get('rho_c',0.005))); mass_s = b*h*L*(row.get('rho_t',0.01)+row.get('rho_c',0.005))*7.85
        cost += v_c*80000 + mass_s*1000000 + (2*h+b)*L*cfg.FORMWORK_UNIT_COST; co2 += v_c*2400*0.15 + mass_s*1000*1.99
    # Fixed Limits and Keys for Logging
    limits, scales = {'strength': 1.0, 'drift': 0.02, 'wind_disp': 1.0, 'deflection': 1.0, 'scwb': 1.0}, {'strength': 1.0, 'drift': 0.02, 'wind_disp': 1.0, 'deflection': 1.0, 'scwb': 0.2}
    vals = {'strength': max_s, 'drift': act_dr, 'wind_disp': act_w, 'deflection': max_defl, 'scwb': act_scwb}
    violation = sum(min(1.0, max(0, vals[k]-limits[k])/scales[k]) for k in limits)
    if not ok or math.isinf(act_dr): violation = max(violation, 1.0)
    abs_violations = {k: max(0, vals[k]-limits[k]) for k in limits}
    return {"cost": cost, "co2": co2, "mean_strength_ratio": mean_s, "violation": violation, "max_drift_ratio": act_dr, "absolute_margins": abs_violations, "N_types": len(set(col_idx))+len(set(beam_idx))}
