import numpy as np
import traceback


def parse_reflect_regions():
    """Parse reflectance region strings into boundaries"""
    regions_list_str = [
        '<0.5', '0.5-0.55', '0.55-0.6', '0.6-0.65', '0.65-0.7', '0.7-0.75',
        '0.75-0.8', '0.8-0.85', '0.85-0.9', '0.9-0.95', '0.95-1.0', '1.0-1.05',
        '1.05-1.1', '1.1-1.15', '1.15-1.2', '1.2-1.25', '1.25-1.3', '1.3-1.35',
        '1.35-1.4', '1.4-1.45', '1.45-1.5', '1.5-1.55', '1.55-1.6', '1.6-1.65',
        '1.65-1.7', '1.7-1.75', '1.75-1.8', '1.8-1.85', '1.85-1.9', '1.9-1.95',
        '1.95-2.0', '2.0-2.05', '2.05-2.1', '2.1-2.15', '2.15-2.2', '2.2-2.25',
        '2.25-2.3', '2.3-2.35', '2.35-2.4', '2.4-2.45', '2.45-2.5', '＞2.5'
    ]
    parsed = []
    for r_str in regions_list_str:
        col_name = (r_str.replace('-', '_').replace('<', 'less_than_')
                              .replace('>', 'greater_than_').replace('＞', 'greater_than_'))
        low, high = -float('inf'), float('inf')
        if '<' in r_str:
            try:
                high = float(r_str[1:])
            except ValueError:
                pass
        elif '＞' in r_str or '>' in r_str:
            try:
                low = float(r_str[1:])
            except ValueError:
                pass
        elif '-' in r_str:
            parts = r_str.split('-')
            try:
                low = float(parts[0])
                high = float(parts[1])
            except ValueError:
                pass
        parsed.append({'name': r_str, 'col': col_name, 'low': low, 'high': high})
    return parsed


def calculate_reflect_std(ratios, df, regions):
    total = ratios.sum()
    if total < 1e-9:
        return 0.0
    valid_cols = []
    mid_points = []
    for region in regions:
        col = region['col']
        if col in df.columns:
            valid_cols.append(col)
            if region['low'] == -float('inf'):
                mid = region['high'] - 0.025
            elif region['high'] == float('inf'):
                mid = region['low'] + 0.025
            else:
                mid = (region['low'] + region['high']) / 2
            mid_points.append(mid)
    if not valid_cols:
        return 0.0
    reflect_matrix = df[valid_cols].values
    mixture_dist = np.dot(ratios, reflect_matrix) / total
    mixture_dist = mixture_dist / 100
    mid_points = np.array(mid_points)
    mean = np.dot(mixture_dist, mid_points)
    variance = np.dot(mixture_dist, (mid_points - mean) ** 2)
    return np.sqrt(variance)


def calculate_property(ratios, prop_type, df, regions):
    total = ratios.sum()
    if abs(total) < 1e-10:
        return 0.0
    col_name = 'csr' if prop_type.lower() == 'csr' else prop_type
    if col_name == 'reflect_std_max':
        return calculate_reflect_std(ratios, df, regions)
    return np.dot(ratios, df[col_name]) / total


def build_constraints(quantity, constraints, perc_constraints, df, regions,
                      main_volatile=None, max_diff=None):
    cons = []
    for col in ['ash', 'sulfur', 'reflect_std_max']:
        if col == 'reflect_std_max':
            cons.append({
                'type': 'ineq',
                'fun': (lambda x, d=df, c=col: constraints[c] - calculate_reflect_std(x, d, regions))
            })
        else:
            cons.append({
                'type': 'ineq',
                'fun': (lambda x, c=col, d=df: (np.dot(x, d[c]) / (x.sum()+1e-10)) - constraints[c][0])
            })
            cons.append({
                'type': 'ineq',
                'fun': (lambda x, c=col, d=df: constraints[c][1] - (np.dot(x, d[c]) / (x.sum()+1e-10)))
            })
    if main_volatile is not None and max_diff is not None:
        vmin = max(main_volatile - max_diff, 0)
        vmax = main_volatile + max_diff
    else:
        vmin, vmax = constraints['volatile']
    cons.append({
        'type': 'ineq',
        'fun': (lambda x, d=df, vmin=vmin: (np.dot(x, d['volatile']) / (x.sum()+1e-10)) - vmin)
    })
    cons.append({
        'type': 'ineq',
        'fun': (lambda x, d=df, vmax=vmax: vmax - (np.dot(x, d['volatile']) / (x.sum()+1e-10)))
    })
    for prop in ['G_value', 'Y_value', 'CSR']:
        cons.append({
            'type': 'ineq',
            'fun': (lambda x, p=prop, d=df: calculate_property(x, p, d, regions) - constraints[p][0])
        })
        cons.append({
            'type': 'ineq',
            'fun': (lambda x, p=prop, d=df: constraints[p][1] - calculate_property(x, p, d, regions))
        })
    cons.append({'type': 'eq', 'fun': (lambda x, q=quantity: x.sum() - q)})

    def make_minp_fun(idxs, min_lim):
        return lambda x, idx=idxs, lim=min_lim: np.sum(x[idx]) - lim * (x.sum()+1e-10)

    def make_maxp_fun(idxs, max_lim):
        return lambda x, idx=idxs, lim=max_lim: lim * (x.sum()+1e-10) - np.sum(x[idx])

    if perc_constraints:
        for cinfo in perc_constraints:
            idxs = cinfo['indices_in_df']
            if not idxs:
                continue
            minp = cinfo['min_perc']
            maxp = cinfo['max_perc']
            cons.append({'type': 'ineq', 'fun': make_minp_fun(list(idxs), minp)})
            cons.append({'type': 'ineq', 'fun': make_maxp_fun(list(idxs), maxp)})
    return cons


def calculate_constraint_violation(solution, target_total, constraints, df,
                                   perc_constraints=None, regions=None):
    total_violation = 0
    total_diff = abs(solution.sum() - target_total)
    total_violation += total_diff * 10
    for prop in ['ash', 'sulfur', 'volatile', 'G_value', 'Y_value', 'CSR']:
        value = calculate_property(solution, prop, df, regions)
        lower, upper = constraints[prop]
        if value < lower:
            total_violation += (lower - value) * 100
        elif value > upper:
            total_violation += (value - upper) * 100
    reflect_std = calculate_reflect_std(solution, df, regions)
    if reflect_std > constraints['reflect_std_max']:
        total_violation += (reflect_std - constraints['reflect_std_max']) * 50
    if perc_constraints:
        for c in perc_constraints:
            idxs = c['indices_in_df']
            minp = c['min_perc']
            maxp = c['max_perc']
            total = solution.sum()
            perc = np.sum(solution[idxs]) / (total + 1e-10)
            if perc < minp:
                total_violation += (minp - perc) * 1000
            if perc > maxp:
                total_violation += (perc - maxp) * 1000
    if 'volatile_diff_max' in constraints:
        used_idx = [i for i, v in enumerate(solution) if v > 0.01]
        if used_idx:
            used_v = [df.iloc[i]['volatile'] for i in used_idx]
            diff = max(used_v) - min(used_v)
            if diff > constraints['volatile_diff_max']:
                total_violation += (diff - constraints['volatile_diff_max']) * 1000
    return total_violation


def find_best_volatile_solution(df, quantity, constraints, perc_constraints,
                                regions, optimizer, max_diff, depth=1, path=None,
                                main_coal_name=None, main_volatile=None):
    import copy
    if path is None:
        path = []
    if df is None or df.empty or len(df) == 0:
        print(f"{'  '*depth}[递归{depth}] 子集为空，返回None")
        return None
    try:
        v_list = df['volatile'].tolist()
        vmin, vmax = min(v_list), max(v_list)
        print(f"{'  '*depth}[递归{depth}] 主煤: {main_coal_name}, max_diff: {max_diff}, 挥发分约束: {main_volatile-max_diff if main_volatile is not None else '-'}~{main_volatile+max_diff if main_volatile is not None else '-'}")
        print(f"{'  '*depth}  子集煤种: {df['coal_name'].tolist()}")
        print(f"{'  '*depth}  子集挥发分区间: {vmin} ~ {vmax}")
        cons = build_constraints(quantity, constraints, perc_constraints, df, regions,
                                 main_volatile=main_volatile, max_diff=max_diff)
        def is_valid_constraint(c):
            return isinstance(c, dict) and 'fun' in c
        cons_list = [c for c in (cons if isinstance(cons, list) else [cons] if cons is not None else []) if is_valid_constraint(c)]
        constraints_list = [c for c in (constraints if isinstance(constraints, list) else [constraints] if constraints is not None else []) if is_valid_constraint(c)]
        all_constraints = constraints_list + cons_list
        if main_volatile is not None and 'volatile' in df.columns:
            df_sorted = df.copy()
            df_sorted['volatile_diff'] = abs(df_sorted['volatile'] - main_volatile)
            df_sorted['is_main'] = (df_sorted['coal_name'] == main_coal_name)
            df_sorted = df_sorted.sort_values(['is_main', 'volatile_diff'], ascending=[False, True])
            top10_idx = df_sorted.index[:10].tolist()
            x0 = [0.0] * len(df)
            for idx in top10_idx:
                x0[idx] = quantity / 10
        else:
            n_coals = len(df)
            x0 = [quantity / n_coals] * n_coals
        bounds = [(0, row['stock']) for _, row in df.iterrows()]
        from scipy.optimize import minimize
        try:
            result = minimize(
                fun=lambda x: np.dot(x, df['price']),
                x0=x0,
                method='SLSQP',
                bounds=bounds,
                constraints=all_constraints
            )
        except Exception as opt_e:
            print(f"{'  '*depth}[递归{depth}] 优化器异常: {opt_e}")
            traceback.print_exc()
            return None
        print(f"{'  '*depth}[递归{depth}] 优化success: {getattr(result, 'success', None)}, message: {getattr(result, 'message', None)}")
        if not getattr(result, 'success', False):
            print(f"{'  '*depth}[递归{depth}] 优化无解，直接返回None")
            return None
        x = result.x
        used_idx = [i for i, v in enumerate(x) if v > 0.01]
        used_v = [df.iloc[i]['volatile'] for i in used_idx]
        if not used_v:
            print(f"{'  '*depth}[递归{depth}] 优化解全为0，返回None")
            return None
        used_vmin, used_vmax = min(used_v), max(used_v)
        diff = used_vmax - used_vmin
        print(f"{'  '*depth}  实际配比煤种挥发分区间: {used_vmin} ~ {used_vmax}, 差值: {diff}")
        if diff <= max_diff:
            avg_volatile = np.dot(x, df['volatile']) / (x.sum() + 1e-10)
            vmin, vmax = constraints['volatile']
            if not (vmin <= avg_volatile <= vmax):
                print(f"{'  '*depth}[递归{depth}] 挥发分加权均值{avg_volatile:.2f}超出区间[{vmin}, {vmax}]，丢弃该解")
            else:
                metrics = optimizer._calc_plan_metrics(result.x, None, df=df)
                cost = float(np.dot(result.x, df['price']))
                print(f"{'  '*depth}[递归{depth}] 满足挥发分差值和区间，返回解")
                return (result.x, metrics, cost, df.copy())
        real_idx_max = used_idx[used_v.index(used_vmax)]
        real_idx_min = used_idx[used_v.index(used_vmin)]
        res1 = None
        if df.iloc[real_idx_max]['coal_name'] != main_coal_name:
            df2 = df.drop(df.index[real_idx_max]).reset_index(drop=True)
            res1 = find_best_volatile_solution(df2, quantity, constraints, perc_constraints,
                                               regions, optimizer, max_diff, depth=depth+1,
                                               path=copy.deepcopy(path)+[('max', df.iloc[real_idx_max]['coal_name'])],
                                               main_coal_name=main_coal_name, main_volatile=main_volatile)
            if res1:
                return res1
        res2 = None
        if df.iloc[real_idx_min]['coal_name'] != main_coal_name and real_idx_min != real_idx_max:
            df2 = df.drop(df.index[real_idx_min]).reset_index(drop=True)
            res2 = find_best_volatile_solution(df2, quantity, constraints, perc_constraints,
                                               regions, optimizer, max_diff, depth=depth+1,
                                               path=copy.deepcopy(path)+[('min', df.iloc[real_idx_min]['coal_name'])],
                                               main_coal_name=main_coal_name, main_volatile=main_volatile)
            if res2:
                return res2
        print(f"{'  '*depth}[递归{depth}] 剔除后仍无解，返回None")
        return None
    except Exception as e:
        print(f"{'  '*depth}[递归{depth}] 异常: {e}")
        traceback.print_exc()
        return None