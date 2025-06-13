import numpy as np
import pandas as pd
from scipy.optimize import minimize
import traceback


def parse_reflect_regions():
    regions_list_str = [
        '<0.5', '0.5-0.55', '0.55-0.6', '0.6-0.65', '0.65-0.7', '0.7-0.75', '0.75-0.8', '0.8-0.85',
            '0.85-0.9', '0.9-0.95', '0.95-1.0', '1.0-1.05', '1.05-1.1', '1.1-1.15', '1.15-1.2', '1.2-1.25',
            '1.25-1.3', '1.3-1.35', '1.35-1.4', '1.4-1.45', '1.45-1.5', '1.5-1.55', '1.55-1.6', '1.6-1.65',
            '1.65-1.7', '1.7-1.75', '1.75-1.8', '1.8-1.85', '1.85-1.9', '1.9-1.95', '1.95-2.0', '2.0-2.05',
            '2.05-2.1', '2.1-2.15', '2.15-2.2', '2.2-2.25', '2.25-2.3', '2.3-2.35', '2.35-2.4', '2.4-2.45',
            '2.45-2.5', '＞2.5' # 注意这里是全角大于号，与代码其他地方可能不同
    ]
    parsed = []
    for r_str in regions_list_str:
        col_name = r_str.replace('-', '_').replace('<', 'less_than_').replace('>', 'greater_than_').replace('＞', 'greater_than_')
        low, high = -float('inf'), float('inf')
        if '<' in r_str:
            try:
                high = float(r_str[1:])
            except ValueError:
                pass # Keep inf
        elif '＞' in r_str or '>' in r_str: # Handle both full-width and half-width greater than
            try:
                low = float(r_str[1:])
            except ValueError:
                pass # Keep inf
        elif '-' in r_str:
            parts = r_str.split('-')
            try:
                low = float(parts[0])
                high = float(parts[1])
            except ValueError:
                pass # Keep inf
        parsed.append({'name': r_str, 'col': col_name, 'low': low, 'high': high})
    return parsed


def calculate_reflect_std(ratios, df, parsed_reflect_regions):
    # 计算权重总和，如果接近零则返回标准差为0
    total = ratios.sum()
    if total < 1e-9:
        return 0.0
    
    # 初始化有效列和中点列表
    valid_cols = []
    mid_points = []
    
    # 遍历每个反射区域
    for region in parsed_reflect_regions:
        col = region['col']
        # 检查列是否存在于DataFrame中
        if col in df.columns:
            valid_cols.append(col)
            # 计算每个区域的中点值
            if region['low'] == -float('inf'):
                mid = region['high'] - 0.025
            elif region['high'] == float('inf'):
                mid = region['low'] + 0.025
            else:
                mid = (region['low'] + region['high']) / 2
            mid_points.append(mid)
    
    # 如果没有有效列，则返回标准差为0
    if not valid_cols:
        return 0.0
    
    # 获取有效列的数据作为反射矩阵
    reflect_matrix = df[valid_cols].values  # shape: (n_coal, n_bins)
    
    # 计算混合分布：将比率与反射矩阵相乘并归一化
    mixture_dist = np.dot(ratios, reflect_matrix) / total  # shape: (n_bins,)
    
    # 将混合分布转换为比例（除以100）
    mixture_dist = mixture_dist / 100
    
    # 将中点列表转换为NumPy数组
    mid_points = np.array(mid_points)
    
    # 计算混合分布的均值
    mean = np.dot(mixture_dist, mid_points)
    
    # 计算混合分布的方差
    variance = np.dot(mixture_dist, (mid_points - mean) ** 2)
    
    # 返回标准差（方差的平方根）
    return np.sqrt(variance)


def calculate_property(ratios, prop_type, df, parsed_reflect_regions):
    total = ratios.sum()
    if abs(total) < 1e-10:
        return 0.0
    col_name = 'csr' if prop_type.lower() == 'csr' else prop_type
    if col_name == 'reflect_std_max':
        return calculate_reflect_std(ratios, df, parsed_reflect_regions)
    return np.dot(ratios, df[col_name]) / total


def build_constraints(quantity, df, constraints, parsed_individual_perc_constraints=None, main_volatile=None, max_diff=None, parsed_reflect_regions=None):
    cons = []
    # 灰分、硫分等
    for col in ['ash', 'sulfur', 'reflect_std_max']:
        if col == 'reflect_std_max':
            cons.append({
                'type': 'ineq',
                'fun': lambda x, d=df, c=constraints, pr=parsed_reflect_regions: c[col] - calculate_reflect_std(x, d, pr)
            })
        else:
            cons.append({
                'type': 'ineq',
                'fun': lambda x, ccol=col, d=df, c=constraints: (np.dot(x, d[ccol]) / x.sum()) - c[ccol][0]
            })
            cons.append({
                'type': 'ineq',
                'fun': lambda x, ccol=col, d=df, c=constraints: c[ccol][1] - (np.dot(x, d[ccol]) / x.sum())
            })
    # 挥发分约束（动态）
    if main_volatile is not None and max_diff is not None:
        vmin = max(main_volatile - max_diff, 0)
        vmax = main_volatile + max_diff
    else:
        vmin, vmax = constraints['volatile']
    cons.append({
        'type': 'ineq', 
        'fun': lambda x, d=df, vmin=vmin: (np.dot(x, d['volatile']) / x.sum()) - vmin
    })
    cons.append({
        'type': 'ineq', 
        'fun': lambda x, d=df, vmax=vmax: vmax - (np.dot(x, d['volatile']) / x.sum())
    })
    # G/CSR/Y
    for prop in ['G_value', 'Y_value', 'CSR']:
        cons.append({
            'type': 'ineq',
            'fun': lambda x, p=prop, d=df, c=constraints, pr=parsed_reflect_regions: calculate_property(x, p, d, pr) - c[p][0]
        })
        cons.append({
            'type': 'ineq',
            'fun': lambda x, p=prop, d=df, c=constraints, pr=parsed_reflect_regions: c[p][1] - calculate_property(x, p, d, pr)
        })
    cons.append({
        'type': 'eq', 
        'fun': lambda x, q=quantity: x.sum() - q
    })
    # 多煤种占比约束
    if parsed_individual_perc_constraints:
        def make_minp_fun(p_indices, min_lim):
            return lambda x: np.sum(x[p_indices]) - min_lim * (x.sum() + 1e-10)
        def make_maxp_fun(p_indices, max_lim):
            return lambda x: max_lim * (x.sum() + 1e-10) - np.sum(x[p_indices])
        for info in parsed_individual_perc_constraints:
            current_perc_constrained_indices = info['indices_in_df']
            min_perc_limit = info['min_perc']
            max_perc_limit = info['max_perc']
            if not current_perc_constrained_indices:
                continue
            cons.append({
                'type': 'ineq', 
                'fun': make_minp_fun(list(current_perc_constrained_indices), min_perc_limit)
            })
            cons.append({
                'type': 'ineq', 
                'fun': make_maxp_fun(list(current_perc_constrained_indices), max_perc_limit)
            })
    return cons


def calculate_constraint_violation(solution, target_total, df, constraints, parsed_individual_perc_constraints=None, parsed_reflect_regions=None):
    total_violation = 0
    total_diff = abs(solution.sum() - target_total)
    total_violation += total_diff * 10
    # 质量指标约束
    for prop in ['ash', 'sulfur', 'volatile', 'G_value', 'Y_value', 'CSR']:
        value = calculate_property(solution, prop, df, parsed_reflect_regions)
        lower, upper = constraints[prop]
        if value < lower:
            total_violation += (lower - value) * 100
        elif value > upper:
            total_violation += (value - upper) * 100
    # 反射率标准差
    reflect_std = calculate_reflect_std(solution, df, parsed_reflect_regions)
    if reflect_std > constraints['reflect_std_max']:
        total_violation += (reflect_std - constraints['reflect_std_max']) * 50
    # 多煤种独立占比约束
    if parsed_individual_perc_constraints:
        for c in parsed_individual_perc_constraints:
            idxs = c['indices_in_df']
            minp = c['min_perc']
            maxp = c['max_perc']
            total = solution.sum()
            perc = np.sum(solution[idxs]) / (total + 1e-10)
            if perc < minp:
                total_violation += (minp - perc) * 1000
            if perc > maxp:
                total_violation += (perc - maxp) * 1000
    # 挥发分最大差值约束
    if 'volatile_diff_max' in constraints:
        used_idx = [i for i, v in enumerate(solution) if v > 0.01]
        if used_idx:
            used_v = [df.iloc[i]['volatile'] for i in used_idx]
            vmin, vmax = min(used_v), max(used_v)
            diff = vmax - vmin
            if diff > constraints['volatile_diff_max']:
                total_violation += (diff - constraints['volatile_diff_max']) * 1000
    return total_violation


def calc_plan_metrics(solution, df, constraints, parsed_reflect_regions, stock_left=None):
    """计算方案的主要指标和库存满足情况，支持传入当前库存和df"""
    total_quantity = solution.sum()
    if stock_left is not None:
        stock_ok = all(solution[i] <= stock_left.iloc[i] + 1e-3 for i in range(len(solution)))
    else:
        stock_ok = all(solution[i] <= df.iloc[i]['stock'] + 1e-3 for i in range(len(solution)))
    metrics = {
        '总配煤量': total_quantity, 
        '库存满足': '是' if stock_ok else '否'
    }
    # 质量指标
    for key in ['ash', 'sulfur', 'volatile', 'G_value', 'Y_value', 'CSR', 'reflect_std_max']:
        if key == 'reflect_std_max':
            val = calculate_reflect_std(solution, df, parsed_reflect_regions)
        else:
            val = calculate_property(solution, key, df, parsed_reflect_regions)
        metrics[key] = val
    # 展示每个煤种剩余库存
    if stock_left is not None:
        metrics['剩余库存'] = ', '.join([
            f"{df.iloc[i]['coal_name']}:{max(0, round(stock_left.iloc[i]-solution[i],2))}吨"
            for i in range(len(solution)) if solution[i] > 0.01
        ])
    return metrics


def find_best_volatile_solution(df, quantity, constraints, parsed_individual_perc_constraints, parsed_reflect_regions, max_diff, depth=1, path=None, main_coal_name=None, main_volatile=None):
    import copy
    if path is None:
        path = []
    if df is None or df.empty or len(df) == 0:
        print(f"{'  '*depth}[递归{depth}] 子集为空，返回None")
        return None
    try:
        # 打印主煤、max_diff、子集煤种、挥发分区间
        v_list = df['volatile'].tolist()
        vmin, vmax = min(v_list), max(v_list)
        print(f"{'  '*depth}[递归{depth}] 主煤: {main_coal_name}, max_diff: {max_diff}, 挥发分约束: {main_volatile-max_diff if main_volatile is not None else '-'}~{main_volatile+max_diff if main_volatile is not None else '-'}")
        print(f"{'  '*depth}  子集煤种: {df['coal_name'].tolist()}")
        print(f"{'  '*depth}  子集挥发分区间: {vmin} ~ {vmax}")
        # 构建约束，参数合法性检查
        cons = build_constraints(quantity, df, constraints, parsed_individual_perc_constraints, main_volatile, max_diff, parsed_reflect_regions)
        def is_valid_constraint(c):
            return isinstance(c, dict) and 'fun' in c
        cons_list = [c for c in (cons if isinstance(cons, list) else [cons] if cons is not None else []) if is_valid_constraint(c)]
        constraints_list = [c for c in (constraints if isinstance(constraints, list) else [constraints] if constraints is not None else []) if is_valid_constraint(c)]
        all_constraints = constraints_list + cons_list
        # x0初始化：主煤和挥发分最接近主煤的前9个煤种均分
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
        # bounds合法性检查
        bounds = [(0, row['stock']) for _, row in df.iterrows()]
        # 优化
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
        # 打印优化结果
        print(f"{'  '*depth}[递归{depth}] 优化success: {getattr(result, 'success', None)}, message: {getattr(result, 'message', None)}")
        if not getattr(result, 'success', False):
            print(f"{'  '*depth}[递归{depth}] 优化无解，直接返回None")
            return None
        # 取实际配比>0.01的煤种挥发分区间
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
            # 新增：校验加权平均挥发分是否在[min, max]区间
            avg_volatile = np.dot(x, df['volatile']) / (x.sum() + 1e-10)
            vmin, vmax = constraints['volatile']
            if not (vmin <= avg_volatile <= vmax):
                print(f"{'  '*depth}[递归{depth}] 挥发分加权均值{avg_volatile:.2f}超出区间[{vmin}, {vmax}]，丢弃该解")
                # 继续递归剔除
            else:
                metrics = calc_plan_metrics(result.x, df, constraints, parsed_reflect_regions)
                cost = float(np.dot(result.x, df['price']))
                print(f"{'  '*depth}[递归{depth}] 满足挥发分差值和区间，返回解")
                return (result.x, metrics, cost, df.copy())
        # 剔除最大/最小煤（主煤不能剔除）
        real_idx_max = used_idx[used_v.index(used_vmax)]
        real_idx_min = used_idx[used_v.index(used_vmin)]
        res1 = None
        if df.iloc[real_idx_max]['coal_name'] != main_coal_name:
            df2 = df.drop(df.index[real_idx_max]).reset_index(drop=True)
            res1 = find_best_volatile_solution(df2, quantity, constraints, parsed_individual_perc_constraints, parsed_reflect_regions, max_diff, depth=depth+1, path=copy.deepcopy(path)+[('max', df.iloc[real_idx_max]['coal_name'])], main_coal_name=main_coal_name, main_volatile=main_volatile)
            if res1:
                return res1
        res2 = None
        if df.iloc[real_idx_min]['coal_name'] != main_coal_name and real_idx_min != real_idx_max:
            df2 = df.drop(df.index[real_idx_min]).reset_index(drop=True)
            res2 = find_best_volatile_solution(df2, quantity, constraints, parsed_individual_perc_constraints, parsed_reflect_regions, max_diff, depth=depth+1, path=copy.deepcopy(path)+[('min', df.iloc[real_idx_min]['coal_name'])], main_coal_name=main_coal_name, main_volatile=main_volatile)
            if res2:
                return res2
        print(f"{'  '*depth}[递归{depth}] 剔除后仍无解，返回None")
        return None
    except Exception as e:
        print(f"{'  '*depth}[递归{depth}] 异常: {e}")
        traceback.print_exc()
        return None


def run_group_optimization(df, quantity, constraints, parsed_individual_perc_constraints, parsed_reflect_regions, max_diff):
    best_solution = None
    best_cost = float('inf')
    best_metrics = None
    best_main_coal = None
    best_df = None
    total_main = len(df)
    for idx, (i, main_coal) in enumerate(df.iterrows()):
        main_volatile = main_coal['volatile']
        print(f"[主煤进度] {idx+1}/{total_main} 主煤: {main_coal['coal_name']} 挥发分: {main_volatile}")
        # 找到所有与主煤挥发分差值在max_diff内的煤
        valid_indices = df.index[abs(df['volatile'] - main_volatile) <= max_diff].tolist()
        if len(valid_indices) == 0:
            continue
        sub_df = df.iloc[valid_indices].reset_index(drop=True)
        print(f"  子集煤种: {list(sub_df['coal_name'])}")
        print(f"  子集挥发分区间: {sub_df['volatile'].min()} ~ {sub_df['volatile'].max()}")
         # 递归剔除法优化，传递主煤名
        res = find_best_volatile_solution(sub_df, quantity, constraints, parsed_individual_perc_constraints, parsed_reflect_regions, max_diff, depth=1, main_coal_name=main_coal['coal_name'], main_volatile=main_volatile)
        if res:
            solution, metrics, cost, sub_df_result = res
            print(f"    [主煤:{main_coal['coal_name']}] 可行解 成本: {cost:.2f}")
            if cost < best_cost:
                best_cost = cost
                best_solution = solution
                best_main_coal = main_coal['coal_name']
                best_df = sub_df_result
                best_metrics = metrics
    if best_solution is None:
        return None
    return best_solution, best_main_coal, best_df, best_metrics
