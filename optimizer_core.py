# optimizer_core.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import traceback

# ------------------ 基础功能 ------------------

def parse_reflect_regions():
    regions_list_str = [
        '<0.5', '0.5-0.55', '0.55-0.6', '0.6-0.65', '0.65-0.7', '0.7-0.75',
        '0.75-0.8', '0.8-0.85', '0.85-0.9', '0.9-0.95', '0.95-1.0', '1.0-1.05',
        '1.05-1.1', '1.1-1.15', '1.15-1.2', '1.2-1.25', '1.25-1.3', '1.3-1.35',
        '1.35-1.4', '1.4-1.45', '1.45-1.5', '1.5-1.55', '1.55-1.6', '1.6-1.65',
        '1.65-1.7', '1.7-1.75', '1.75-1.8', '1.8-1.85', '1.85-1.9', '1.9-1.95',
        '1.95-2.0', '2.0-2.05', '2.05-2.1', '2.1-2.15', '2.15-2.2', '2.2-2.25',
        '2.25-2.3', '2.3-2.35', '2.35-2.4', '2.4-2.45', '2.45-2.5', '＞2.5']

    parsed = []
    for r_str in regions_list_str:
        col_name = r_str.replace('-', '_').replace('<', 'less_than_').replace('>', 'greater_than_').replace('＞', 'greater_than_')
        low, high = -float('inf'), float('inf')
        if '<' in r_str:
            high = float(r_str[1:])
        elif '＞' in r_str or '>' in r_str:
            low = float(r_str[1:])
        elif '-' in r_str:
            parts = r_str.split('-')
            low, high = float(parts[0]), float(parts[1])
        parsed.append({'name': r_str, 'col': col_name, 'low': low, 'high': high})
    return parsed

# ------------------ 属性计算 ------------------

def calculate_property(ratios, prop_type, df):
    total = ratios.sum()
    if abs(total) < 1e-10:
        return 0.0
    col_name = 'csr' if prop_type.lower() == 'csr' else prop_type
    return np.dot(ratios, df[col_name]) / total

def calculate_reflect_std(ratios, df, parsed_regions):
    total = ratios.sum()
    if total < 1e-9:
        return 0.0
    valid_cols = []
    mid_points = []
    for region in parsed_regions:
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
    reflect_matrix = df[valid_cols].values
    mixture_dist = np.dot(ratios, reflect_matrix) / total
    mixture_dist = mixture_dist / 100
    mean = np.dot(mixture_dist, mid_points)
    variance = np.dot(mixture_dist, (mid_points - mean) ** 2)
    return np.sqrt(variance)

def calculate_plastic_temp(ratios, df):
    total = ratios.sum()
    return np.dot(ratios, df['ΔT'].values) / total if total > 0 else 0.0

# ------------------ 约束构建 ------------------

def build_constraints(quantity, constraints, parsed_individual_constraints, df, main_volatile=None, max_diff=None, parsed_regions=None):
    cons = []
    # 质量指标
    for col in ['ash', 'sulfur']:
        cons.append({'type': 'ineq', 'fun': lambda x, c=col: (np.dot(x, df[c]) / x.sum()) - constraints[c][0]})
        cons.append({'type': 'ineq', 'fun': lambda x, c=col: constraints[c][1] - (np.dot(x, df[c]) / x.sum())})

    if main_volatile is not None and max_diff is not None:
        vmin = max(main_volatile - max_diff, 0)
        vmax = main_volatile + max_diff
    else:
        vmin, vmax = constraints['volatile']
    cons.append({'type': 'ineq', 'fun': lambda x: (np.dot(x, df['volatile']) / x.sum()) - vmin})
    cons.append({'type': 'ineq', 'fun': lambda x: vmax - (np.dot(x, df['volatile']) / x.sum())})

    for prop in ['G_value', 'Y_value', 'CSR']:
        cons.append({'type': 'ineq', 'fun': lambda x, p=prop: calculate_property(x, p, df) - constraints[p][0]})
        cons.append({'type': 'ineq', 'fun': lambda x, p=prop: constraints[p][1] - calculate_property(x, p, df)})

    cons.append({'type': 'eq', 'fun': lambda x: x.sum() - quantity})

    if 'reflect_std_max' in constraints and parsed_regions:
        cons.append({'type': 'ineq', 'fun': lambda x: constraints['reflect_std_max'] - calculate_reflect_std(x, df, parsed_regions)})

    if parsed_individual_constraints:
        for c in parsed_individual_constraints:
            idxs = c['indices_in_df']
            minp = c['min_perc']
            maxp = c['max_perc']
            cons.append({'type': 'ineq', 'fun': lambda x, idxs=idxs, minp=minp: np.sum(x[idxs]) - minp * (x.sum() + 1e-10)})
            cons.append({'type': 'ineq', 'fun': lambda x, idxs=idxs, maxp=maxp: maxp * (x.sum() + 1e-10) - np.sum(x[idxs])})

    return cons

def calculate_constraint_violation(solution, quantity, df, constraints, parsed_individual_constraints=None, parsed_regions=None):
    violation = 0
    total = solution.sum()
    violation += abs(total - quantity) * 10
    
    for key in ['ash', 'sulfur', 'volatile', 'G_value', 'Y_value', 'CSR']:
        value = calculate_property(solution, key, df)
        lower, upper = constraints[key]
        if value < lower:
            violation += (lower - value) * 100
        elif value > upper:
            violation += (value - upper) * 100

    if 'reflect_std_max' in constraints and parsed_regions:
        reflect_std = calculate_reflect_std(solution, df, parsed_regions)
        if reflect_std > constraints['reflect_std_max']:
            violation += (reflect_std - constraints['reflect_std_max']) * 50

    if parsed_individual_constraints:
        for c in parsed_individual_constraints:
            idxs = c['indices_in_df']
            minp = c['min_perc']
            maxp = c['max_perc']
            perc = np.sum(solution[idxs]) / (solution.sum() + 1e-10)
            if perc < minp:
                violation += (minp - perc) * 1000
            elif perc > maxp:
                violation += (perc - maxp) * 1000

    if 'volatile_diff_max' in constraints:
        used_idx = [i for i, v in enumerate(solution) if v > 0.01]
        if used_idx:
            used_v = [df.iloc[i]['volatile'] for i in used_idx]
            diff = max(used_v) - min(used_v)
            if diff > constraints['volatile_diff_max']:
                violation += (diff - constraints['volatile_diff_max']) * 1000

    return violation

# ------------------ 解后处理 ------------------
def postprocess_solution(solution, total, use_integer, rounding_precision):
    """
    将解进行取整处理并调整使总量近似于 total。
    rounding_precision: 0 表示整数，1 表示一位小数
    """
    if use_integer:
        ratio_percent_raw = solution / total * 100
        ratio_int = np.floor(ratio_percent_raw).astype(int)
        remainder = ratio_percent_raw - ratio_int
        diff = 100 - ratio_int.sum()
        if diff > 0:
            ratio_int[np.argsort(-remainder)[:diff]] += 1
        elif diff < 0:
            ratio_int[np.argsort(remainder)[:abs(diff)]] -= 1
        ratio_int = np.maximum(0, ratio_int)
        tons = ratio_int / 100 * total
        if rounding_precision == 0:
            return np.round(tons).astype(int)
        else:
            return np.round(tons, rounding_precision)
    else:
        return np.round(solution, 2)
    
# ------------------ 主煤策略优化 ------------------
def find_best_volatile_solution(df, quantity, constraints, parsed_individual_constraints, parsed_regions, max_diff, main_volatile):
    """
    尝试递归剔除最小/最大挥发分煤种，寻找满足 max_diff 限制的最优解。
    """
    try:
        def valid_solution(x):
            used_idx = [i for i, v in enumerate(x) if v > 0.01]
            used_v = df.iloc[used_idx]['volatile']
            return used_v.max() - used_v.min() <= max_diff

        cons = build_constraints(
            quantity,
            constraints,
            parsed_individual_constraints,
            df,
            main_volatile=main_volatile,
            max_diff=max_diff,
            parsed_regions=parsed_regions
        )
        x0 = np.ones(len(df)) * (quantity / len(df))
        bounds = [(0, row['stock']) for _, row in df.iterrows()]
        result = minimize(
            fun=lambda x: np.dot(x, df['price']),
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        if not result.success or not valid_solution(result.x):
            if len(df) <= 2:
                return None
            volatiles = df['volatile']
            vmax_idx = volatiles.idxmax()
            vmin_idx = volatiles.idxmin()
            df_drop_max = df.drop(index=vmax_idx)
            df_drop_min = df.drop(index=vmin_idx)
            res1 = find_best_volatile_solution(df_drop_max, quantity, constraints, parsed_individual_constraints, parsed_regions, max_diff, main_volatile)
            if res1:
                return res1
            res2 = find_best_volatile_solution(df_drop_min, quantity, constraints, parsed_individual_constraints, parsed_regions, max_diff, main_volatile)
            return res2
        return result.x
    except Exception as e:
        print("优化失败：", e)
        traceback.print_exc()
        return None