import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from functools import lru_cache
import logging
from datetime import datetime
import os
import traceback
import optimizer_core

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class CoalBlendOptimizer:
    def __init__(self, master):
        self.master = master
        self.df = None
        self.num_individual_perc_constraints = 12  # 增加到12组
        self.current_page = 0  # 当前页码
        self.constraints_per_page = 6  # 每页显示6组
        self.ind_perc_coals_strs = [tk.StringVar() for _ in range(self.num_individual_perc_constraints)]
        self.ind_perc_min_strs = [tk.StringVar() for _ in range(self.num_individual_perc_constraints)]
        self.ind_perc_max_strs = [tk.StringVar() for _ in range(self.num_individual_perc_constraints)]
        self.parsed_individual_perc_constraints = [] # 用于存储解析后的约束
        self.constraints = {
            'G_value': (80, 88),
            'CSR': (60, 68),
            'Y_value': (15, 25),
            'ash': (8, 10),
            'sulfur': (0.5, 1.0),
            'volatile': (18, 28),
            'reflect_std_max': 0.3
        }
        self.integer_constraint = tk.BooleanVar(value=False)
        self.rounding_precision = tk.IntVar(value=0)  # 0=整数，1=小数点后1位
        self.parsed_reflect_regions = optimizer_core.parse_reflect_regions()
        self._setup_style()
        self.build_ui()
        self._bind_shortcuts()
        self._bind_events()
        self.status_bar = tk.Label(self.master, text=" 就绪 ", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.historical_accuracy = 0.8
        
    def _setup_style(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # 主色调：科技蓝 + 生态绿
        style.configure('.', 
                       background='#F8F9FA',  # 背景色
                       foreground='#2C3E50',  # 文字色
                       font=('Microsoft YaHei', 10))
        
        # 按钮梯度配色
        style.map('TButton',
                 foreground=[('active', '#FFFFFF'), ('disabled', '#A0A0A0')],
                 background=[('active', '#2C81BA'), ('!active', '#3498DB')],
                 relief=[('pressed', 'sunken'), ('!pressed', 'raised')])
        
        # 特殊样式
        style.configure('Accent.TButton', 
                       background='#27AE60',  # 强调绿色
                       foreground='white',
                       font=('Microsoft YaHei', 10, 'bold'))
        
        # 输入框样式
        style.configure('TEntry',
                      fieldbackground='#FFFFFF',
                      bordercolor='#BDC3C7',
                      lightcolor='#BDC3C7',
                      darkcolor='#BDC3C7')
        
        # 卡片式面板
        style.configure('Card.TFrame',
                      background='white',
                      relief='solid',
                      borderwidth=1,
                      bordercolor='#EAECEE')
        
        # 进度条现代样式
        style.configure("Modern.Horizontal.TProgressbar",
                       thickness=20,
                       troughcolor='#EAECEE',
                       troughrelief='flat',
                       pbarrelief='flat',
                       background='#3498DB',
                       lightcolor='#3498DB',
                       darkcolor='#2C81BA')
        
        # 标签样式
        style.configure('Header.TLabel',
                       font=('Microsoft YaHei', 14, 'bold'),
                       foreground='#2C3E50')
        
        style.configure('SubHeader.TLabel',
                       font=('Microsoft YaHei', 12),
                       foreground='#34495E')
        
        style.configure('StatusBar.TLabel',
                       background='#F8F9FA',
                       foreground='#7F8C8D',
                       font=('Microsoft YaHei', 9))
        
        # 表格样式
        style.configure('Treeview',
                       background='#FFFFFF',
                       foreground='#2C3E50',
                       fieldbackground='#FFFFFF',
                       rowheight=25)
        
        style.configure('Treeview.Heading',
                       background='#F8F9FA',
                       foreground='#2C3E50',
                       font=('Microsoft YaHei', 9, 'bold'))
        
        # 滚动条样式
        style.configure('Vertical.TScrollbar',
                       background='#F8F9FA',
                       troughcolor='#EAECEE',
                       width=10)
        
        style.configure('Horizontal.TScrollbar',
                       background='#F8F9FA',
                       troughcolor='#EAECEE',
                       height=10)
        
        # 下拉框样式
        style.configure('TCombobox',
                       background='#FFFFFF',
                       foreground='#2C3E50',
                       fieldbackground='#FFFFFF',
                       arrowcolor='#3498DB')
        
        # 标签页样式
        style.configure('TNotebook',
                       background='#F8F9FA',
                       tabmargins=[2, 5, 2, 0])
        
        style.configure('TNotebook.Tab',
                       padding=[10, 5],
                       background='#EAECEE',
                       foreground='#2C3E50',
                       font=('Microsoft YaHei', 9))
        
        style.map('TNotebook.Tab',
                 background=[('selected', '#FFFFFF')],
                 foreground=[('selected', '#3498DB')])
        
        # 工具栏按钮样式
        style.configure('Toolbar.TButton',
                       padding=8,
                       font=('Microsoft YaHei', 9),
                       background='#3498DB',
                       foreground='white')
        
        style.map('Toolbar.TButton',
                 background=[('active', '#2C81BA')],
                 foreground=[('active', 'white')])

    def _bind_events(self):
        # 使单击窗口时自动获取焦点
        self.master.bind_all("<Button-1>", lambda e: self._handle_focus(e))
        
        # 更多事件绑定...
        
    def _handle_focus(self, event):
        """处理焦点设置，避免str对象错误"""
        try:
            if hasattr(event.widget, 'focus_set'):
                event.widget.focus_set()
        except:
            pass

    def build_ui(self):
        self.master.title("智能配煤系统")
        self.master.geometry("1200x800")
        
        # 创建主菜单
        self._create_menu()
        
        # 创建工具栏
        self._create_toolbar()
        
        # 主容器使用现代布局
        main_container = ttk.Frame(self.master)
        main_container.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)
        
        # 左侧面板（输入区）
        left_panel = ttk.Frame(main_container, style='Card.TFrame')
        left_panel.grid(row=0, column=0, padx=8, pady=8, sticky='nsew')
        
        # 右侧面板（结果区） 
        right_panel = ttk.Frame(main_container, style='Card.TFrame')
        right_panel.grid(row=0, column=1, padx=8, pady=8, sticky='nsew')
        
        # 配置网格权重
        main_container.columnconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=2)
        main_container.rowconfigure(0, weight=1)
        
        # 添加现代化标题
        self._build_header(left_panel)
        
        # 左侧面板：数据配置和参数设置
        self._build_left_panel(left_panel)
        
        # 右侧面板：结果显示和图表
        self._build_right_panel(right_panel)
        
        # 底部状态栏
        self.status_bar = ttk.Label(self.master, text=" 就绪 ", style='StatusBar.TLabel')
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 绑定快捷键
        self._bind_shortcuts()

    def _build_header(self, parent):
        """添加现代化标题"""
        header_frame = ttk.Frame(parent, style='Card.TFrame')
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 系统标题
        title_label = ttk.Label(header_frame, 
                              text="智能配煤系统", 
                              style='Header.TLabel')
        title_label.pack(side=tk.LEFT, padx=5)
        
        # 版本信息
        version_label = ttk.Label(header_frame,
                                text="v30.8",
                                style='SubHeader.TLabel',
                                foreground='#7F8C8D')
        version_label.pack(side=tk.LEFT, padx=5)
        
        # 添加分隔线
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=5)

    def _create_menu(self):
        menubar = tk.Menu(self.master)
        self.master.config(menu=menubar)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="选择煤种文件", command=self.load_coal_file, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.master.quit, accelerator="Alt+F4")
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="关于", command=self.show_about)

    def _create_toolbar(self):
        toolbar = ttk.Frame(self.master)
        toolbar.pack(fill=tk.X, padx=5, pady=2)
        # 文件操作按钮
        ttk.Button(toolbar, text="选择煤种文件", command=self.load_coal_file, style='Toolbar.TButton').pack(side=tk.LEFT, padx=2)
        # 分隔符
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        # 模型操作按钮
        ttk.Button(toolbar, text="开始计算", command=self.run_optimization, style='Toolbar.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="重置参数", command=self.reset_parameters, style='Toolbar.TButton').pack(side=tk.LEFT, padx=2)
        # 新增多方案优化按钮
        ttk.Button(toolbar, text="多方案优化", command=self.multi_plan_optimization, style='Toolbar.TButton').pack(side=tk.LEFT, padx=2)
        # 状态显示
        self.file_status = ttk.Label(toolbar, text=" 等待文件选择 ", style='StatusBar.TLabel')
        self.file_status.pack(side=tk.RIGHT, padx=5)

    def multi_plan_optimization(self):
        dialog = MultiPlanDialog(self.master, self.constraints, self.df, self)
        self.master.wait_window(dialog.top)
        if not dialog.result:
            return
        plans = dialog.result  # list of dict，每个dict包含quantity、constraints、individual_perc_constraints
        M = len(plans)
        N = len(self.df)
        prices = self.df['price'].values
        stocks = self.df['stock'].values
        # 变量顺序: [方案1煤1, 方案1煤2, ..., 方案1煤N, 方案2煤1, ..., 方案M煤N]
        def total_cost(x):
            return np.sum(x * np.tile(prices, M))
        # 约束列表
        constraints = []
        # 每个方案的总量约束
        for j, plan in enumerate(plans):
            def cons_fun(x, j=j, plan=plan):
                return np.sum(x[j*N:(j+1)*N]) - plan['quantity']
            constraints.append({'type': 'eq', 'fun': cons_fun})
        # 每个方案的质量约束和多煤种占比约束
        for j, plan in enumerate(plans):
            # 质量约束
            cons = plan['constraints']
            # 反射率标准差
            if 'reflect_std_max' in cons:
                def reflect_std_max_fun(x, j=j, cons=cons):
                    return cons['reflect_std_max'] - self._calculate_reflect_std(x[j*N:(j+1)*N], self.df)
                constraints.append({'type': 'ineq', 'fun': reflect_std_max_fun})
            # 其它质量指标
            for col in ['ash', 'sulfur', 'volatile']:
                def min_fun(x, j=j, col=col, cons=cons):
                    return (np.dot(x[j*N:(j+1)*N], self.df[col]) / (x[j*N:(j+1)*N].sum()+1e-10)) - cons[col][0]
                def max_fun(x, j=j, col=col, cons=cons):
                    return cons[col][1] - (np.dot(x[j*N:(j+1)*N], self.df[col]) / (x[j*N:(j+1)*N].sum()+1e-10))
                constraints.append({'type': 'ineq', 'fun': min_fun})
                constraints.append({'type': 'ineq', 'fun': max_fun})
            # G/CSR/Y
            for prop in ['G_value', 'Y_value', 'CSR']:
                def min_fun(x, j=j, prop=prop, cons=cons):
                    return self._calculate_property(x[j*N:(j+1)*N], prop) - cons[prop][0]
                def max_fun(x, j=j, prop=prop, cons=cons):
                    return cons[prop][1] - self._calculate_property(x[j*N:(j+1)*N], prop)
                constraints.append({'type': 'ineq', 'fun': min_fun})
                constraints.append({'type': 'ineq', 'fun': max_fun})
            # 新增：反射率分布范围约束
            if 'reflect_dist_user_bounds' in cons and hasattr(self, 'parsed_reflect_regions'):
                user_min_reflect, user_max_reflect = cons['reflect_dist_user_bounds']
                for region_info in self.parsed_reflect_regions:
                    r_low, r_high, r_col = region_info['low'], region_info['high'], region_info['col']
                    if r_col not in self.df.columns:
                        continue
                    is_outside = False
                    if r_high <= user_min_reflect:
                        is_outside = True
                    elif r_low >= user_max_reflect:
                        is_outside = True
                    if is_outside:
                        def reflect_dist_fun(x, j=j, col=r_col):
                            return 0.001 - (np.dot(x[j*N:(j+1)*N], self.df[col]) / (x[j*N:(j+1)*N].sum() + 1e-9))
                        constraints.append({'type': 'ineq', 'fun': reflect_dist_fun})
            # 多煤种占比约束
            if 'individual_perc_constraints' in plan and plan['individual_perc_constraints']:
                for cinfo in plan['individual_perc_constraints']:
                    idxs = cinfo['indices_in_df']
                    minp = cinfo['min_perc']
                    maxp = cinfo['max_perc']
                    def minp_fun(x, j=j, idxs=idxs, minp=minp):
                        return np.sum(x[j*N+np.array(idxs)]) - minp * (x[j*N:(j+1)*N].sum()+1e-10)
                    def maxp_fun(x, j=j, idxs=idxs, maxp=maxp):
                        return maxp * (x[j*N:(j+1)*N].sum()+1e-10) - np.sum(x[j*N+np.array(idxs)])
                    constraints.append({'type': 'ineq', 'fun': minp_fun})
                    constraints.append({'type': 'ineq', 'fun': maxp_fun})
        # 所有方案的同一煤种总用量不超过库存
        for i in range(N):
            def stock_fun(x, i=i):
                return stocks[i] - np.sum([x[j*N + i] for j in range(M)])
            constraints.append({'type': 'ineq', 'fun': stock_fun})
        # 变量边界
        bounds = [(0, stocks[i%N]) for i in range(N*M)]
        # 初始解
        x0 = np.zeros(N*M)
        for j, plan in enumerate(plans):
            x0[j*N:(j+1)*N] = plan['quantity'] / N
        # 优化
        from scipy.optimize import minimize
        result = minimize(
            fun=total_cost,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': 2000,
                'ftol': 1e-6,
                'disp': False,
                'eps': 1e-8
            }
        )
        results = []
        if result.success:
            x_opt = result.x.reshape(M, N)
            stock_left = stocks.copy()
            for j, plan in enumerate(plans):
                solution = x_opt[j]
                # 新增：严格校验
                violation = self._calculate_constraint_violation(solution, plan['quantity'], self.df)
                if violation > 0:
                    msg = f"优化器输出解不满足所有约束，违规值：{violation:.2f}"
                    results.append((None, None, msg))
                else:
                    metrics = self._calc_plan_metrics(solution, plan, stock_left)
                    msg = "优化成功"
                    results.append((solution, metrics, msg))
                    stock_left = stock_left - solution
                    stock_left = np.maximum(stock_left, 0)
        else:
            for j, plan in enumerate(plans):
                results.append((None, None, f"方案{j+1}优化失败: {result.message}"))
        self._show_multi_plan_results(plans, results)
        if not result.success:
            from tkinter import messagebox
            messagebox.showwarning("多方案优化失败", result.message)

    def _single_plan_optimize(self, plan, stock_left):
        """单个方案优化，返回解、指标、消息，库存由stock_left决定"""
        quantity = plan['quantity']
        constraints = plan['constraints']
        old_constraints = self.constraints.copy()
        self.constraints = constraints.copy()
        n_coals = len(self.df)
        x0 = np.ones(n_coals) * (quantity / n_coals)
        cons = self._build_constraints(quantity)
        result = minimize(
            fun=lambda x: np.dot(x, self.df['price']),
            x0=x0,
            method='SLSQP',
            bounds=[(0, stock_left.iloc[i]) for i in range(n_coals)],
            constraints=cons,
            options={
                'maxiter': 1000,
                'ftol': 1e-8,
                'disp': False,
                'eps': 1e-10
            }
        )
        self.constraints = old_constraints  # 恢复
        if not result.success:
            raise Exception("无法找到可行解（可能因库存不足）")
        # 优化后严格校验
        violation = self._calculate_constraint_violation(result.x, quantity, self.df)
        if violation > 0:
            print("优化器输出解不满足所有约束，违规值：", violation)
            for prop in ['ash', 'sulfur', 'volatile', 'G_value', 'Y_value', 'CSR']:
                value = self._calculate_property(result.x, prop, self.df)
                print(f"{prop}: {value}, 约束: {self.constraints[prop]}")
            reflect_std = self._calculate_reflect_std(result.x, self.df)
            print(f"reflect_std_max: {reflect_std}, 约束: {self.constraints['reflect_std_max']}")
            if hasattr(self, 'parsed_individual_perc_constraints') and self.parsed_individual_perc_constraints:
                for c in self.parsed_individual_perc_constraints:
                    idxs = c['indices_in_df']
                    minp = c['min_perc']
                    maxp = c['max_perc']
                    total = result.x.sum()
                    perc = np.sum(result.x[idxs]) / (total + 1e-10)
                    print(f"多煤种占比: {perc}, 约束: [{minp}, {maxp}]")
            if 'volatile_diff_max' in self.constraints:
                used_idx = [i for i, v in enumerate(result.x) if v > 0.01]
                if used_idx:
                    used_v = [self.df.iloc[i]['volatile'] for i in used_idx]
                    vmin, vmax = min(used_v), max(used_v)
                    diff = vmax - vmin
                    print(f"实际挥发分区间: {vmin}~{vmax}, 差值: {diff}, 约束: {self.constraints['volatile_diff_max']}")
            raise Exception("优化无可行解，所有约束必须严格满足！")
        solution = self._postprocess_solution(result.x, self.integer_constraint.get(), 10 ** (-self.rounding_precision.get()))
        metrics = self._calc_plan_metrics(solution, plan, stock_left)
        # 调试：打印每个煤种配比和多煤种占比约束
        print('优化后配比：')
        for idx, v in enumerate(solution):
            print(self.df.iloc[idx]['coal_name'], v, '占比:', v/solution.sum())
        print('当前多煤种占比约束：', self.parsed_individual_perc_constraints)
        return solution, metrics, "优化成功"

    def _calc_plan_metrics(self, solution, plan, stock_left=None, df=None):
        """计算方案的主要指标和库存满足情况，支持传入当前库存和df"""
        if df is None:
            df = self.df
        total_quantity = solution.sum()
        if stock_left is not None:
            stock_ok = all(solution[i] <= stock_left.iloc[i]+1e-3 for i in range(len(solution)))
        else:
            stock_ok = all(solution[i] <= df.iloc[i]['stock']+1e-3 for i in range(len(solution)))
        metrics = {
            '总配煤量': total_quantity,
            '库存满足': '是' if stock_ok else '否',
        }
        # 质量指标
        for key in ['ash', 'sulfur', 'volatile', 'G_value', 'Y_value', 'CSR', 'reflect_std_max']:
            if key == 'reflect_std_max':
                val = self._calculate_reflect_std(solution, df)
            else:
                val = self._calculate_property(solution, key, df)
            metrics[key] = val
        # 展示每个煤种剩余库存
        if stock_left is not None:
            metrics['剩余库存'] = ', '.join([
                f"{df.iloc[i]['coal_name']}:{max(0, round(stock_left.iloc[i]-solution[i],2))}吨"
                for i in range(len(solution)) if solution[i]>0.01
            ])
        return metrics

    def _show_multi_plan_results(self, plans, results):
        for i in range(self.result_notebook.index('end')):
            if self.result_notebook.tab(i, 'text') == '多方案结果':
                self.result_notebook.forget(i)
                break
        multi_tab = ttk.Notebook(self.result_notebook)
        for idx, (plan, (solution, metrics, msg)) in enumerate(zip(plans, results)):
            frame = ttk.Frame(multi_tab)
            if solution is not None:
                self._build_ratio_table(frame, solution, self.df)
                self._build_metric_panel(frame, solution, self.df)
                self._build_reflect_chart(frame, solution, self.df)
                summary = ttk.LabelFrame(frame, text="方案汇总", padding=5)
                summary.pack(fill=tk.X, padx=5, pady=5)
                for k, v in metrics.items():
                    ttk.Label(summary, text=f"{k}: {v}", font=('Microsoft YaHei', 9)).pack(side=tk.LEFT, padx=8)
                ttk.Label(summary, text=msg, foreground='#27AE60').pack(side=tk.LEFT, padx=8)
            else:
                ttk.Label(frame, text=msg, foreground='#E74C3C', font=('Microsoft YaHei', 11)).pack(padx=20, pady=20)
            multi_tab.add(frame, text=f"方案{idx+1}")
        self.result_notebook.add(multi_tab, text="多方案结果")
        self.result_notebook.select(multi_tab)

    def _build_left_panel(self, parent):
        # 数据源配置
        data_frame = ttk.LabelFrame(parent, text=" 数据源配置 ", padding=10, style='Card.TFrame')
        data_frame.pack(fill=tk.X, pady=5)
        # 移除模型选择相关UI
        # 只保留数据源和参数部分
        # 基本参数
        param_frame = ttk.LabelFrame(parent, text=" 基本参数 ", padding=10, style='Card.TFrame')
        param_frame.pack(fill=tk.X, pady=5)
        # 配煤总量
        quantity_frame = ttk.Frame(param_frame)
        quantity_frame.pack(fill=tk.X, pady=5)
        ttk.Label(quantity_frame, text="配煤总量（吨）:", style='SubHeader.TLabel').pack(side=tk.LEFT)
        self.quantity_entry = ttk.Entry(quantity_frame, width=15, font=('Microsoft YaHei', 10))
        self.quantity_entry.pack(side=tk.LEFT, padx=10)
        # 质量指标约束
        constraint_frame = ttk.LabelFrame(parent, text=" 质量指标约束 ", padding=10, style='Card.TFrame')
        constraint_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        # 创建滚动区域
        canvas = tk.Canvas(constraint_frame, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(constraint_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        # 添加约束参数
        self.entries = {}
        params = [
            ('灰分 (%)', 'ash'),
            ('硫分 (%)', 'sulfur'),
            ('挥发分 (%)', 'volatile'),
            ('G 值', 'G_value'),
            ('Y 值 (mm)', 'Y_value'),
            ('CSR', 'CSR'),
            ('反射率标准差', 'reflect_std_max')
        ]
        for i, (label, key) in enumerate(params):
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill=tk.X, pady=3)
            ttk.Label(frame, text=label, width=14, anchor=tk.E, style='SubHeader.TLabel').pack(side=tk.LEFT)
            if key == 'reflect_std_max':
                entry = ttk.Entry(frame, width=10, font=('Microsoft YaHei', 10))
                entry.insert(0, str(self.constraints[key]))
                self.entries[key] = entry
                entry.pack(side=tk.LEFT, padx=5)
            else:
                min_entry = ttk.Entry(frame, width=8, font=('Microsoft YaHei', 10))
                min_entry.insert(0, str(self.constraints[key][0]))
                max_entry = ttk.Entry(frame, width=8, font=('Microsoft YaHei', 10))
                max_entry.insert(0, str(self.constraints[key][1]))
                self.entries[key] = (min_entry, max_entry)
                min_entry.pack(side=tk.LEFT, padx=5)
                max_entry.pack(side=tk.LEFT, padx=5)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 新增最大挥发分差值输入框
        frame = ttk.Frame(scrollable_frame)
        frame.pack(fill=tk.X, pady=3)
        ttk.Label(frame, text="最大挥发分差值(%)", width=14, anchor=tk.E, style='SubHeader.TLabel').pack(side=tk.LEFT)
        volatile_diff_entry = ttk.Entry(frame, width=10, font=('Microsoft YaHei', 10))
        volatile_diff_entry.insert(0, "5")
        self.entries['volatile_diff_max'] = volatile_diff_entry
        volatile_diff_entry.pack(side=tk.LEFT, padx=5)
        
        # 精度设置区域
        precision_frame = ttk.LabelFrame(parent, text=" 配量精度设置 ", padding=10)
        precision_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(precision_frame, 
                      text="启用取整约束",
                      variable=self.integer_constraint,
                      command=self._toggle_precision).pack(side=tk.LEFT, padx=(0, 5))
        
        self.precision_options = {"整数": 0, "一位小数": 1}
        self.precision_combo = ttk.Combobox(precision_frame,
                                            textvariable=tk.StringVar(), # Temporary, will be handled by event
                                            values=list(self.precision_options.keys()),
                                            width=10,
                                            state='disabled') # Initially disabled
        self.precision_combo.pack(side=tk.LEFT)
        self.precision_combo.bind("<<ComboboxSelected>>", self._update_rounding_from_combo)
        
        # Set initial combobox value based on self.rounding_precision
        for text, val in self.precision_options.items():
            if val == self.rounding_precision.get():
                self.precision_combo.set(text)
                break
        if not self.precision_combo.get(): # Default if not found
             self.precision_combo.current(0) # Default to "整数"
             self.rounding_precision.set(self.precision_options[self.precision_combo.get()])

        # Ensure toggle state is correct on init
        self._toggle_precision() # Call to set initial state based on integer_constraint

    def _toggle_precision(self):
        """切换精度设置可用性"""
        if self.integer_constraint.get():
            self.precision_combo.config(state="readonly")
        else:
            self.precision_combo.config(state="disabled")
            
    def _update_precision(self, event=None):
        """动态更新精度级别"""
        pass  # 可扩展精度级别

    def _update_rounding_from_combo(self, event=None):
        """从下拉框更新rounding_precision变量"""
        selected_text = self.precision_combo.get()
        if selected_text in self.precision_options:
            self.rounding_precision.set(self.precision_options[selected_text])

    def _build_right_panel(self, parent):
        # 添加多种煤种独立占比约束区域到右侧面板 (原进度信息位置)
        multi_percentage_constraint_frame = ttk.LabelFrame(parent, text=" 多种煤种独立占比约束 (共 {} 组) ".format(self.num_individual_perc_constraints), padding=10, style='Card.TFrame')
        multi_percentage_constraint_frame.pack(fill=tk.X, pady=5, side=tk.TOP)

        # 创建约束显示区域
        self.constraints_container = ttk.Frame(multi_percentage_constraint_frame)
        self.constraints_container.pack(fill=tk.X, pady=5)

        # 注意：self.ind_perc_coals_entries等列表在此处需要重新初始化，因为它们属于UI元素
        self.ind_perc_coals_entries = [] 
        self.ind_perc_min_entries = []
        self.ind_perc_max_entries = []

        # 创建翻页按钮区域
        page_control_frame = ttk.Frame(multi_percentage_constraint_frame)
        page_control_frame.pack(fill=tk.X, pady=5)

        # 上一页按钮
        self.prev_page_btn = ttk.Button(page_control_frame, text="上一页", 
                                      command=self._prev_page,
                                      style='Accent.TButton')
        self.prev_page_btn.pack(side=tk.LEFT, padx=5)

        # 页码显示
        self.page_label = ttk.Label(page_control_frame, 
                                  text=f"第 {self.current_page + 1} 页，共 {(self.num_individual_perc_constraints + self.constraints_per_page - 1) // self.constraints_per_page} 页")
        self.page_label.pack(side=tk.LEFT, padx=10)

        # 下一页按钮
        self.next_page_btn = ttk.Button(page_control_frame, text="下一页", 
                                      command=self._next_page,
                                      style='Accent.TButton')
        self.next_page_btn.pack(side=tk.LEFT, padx=5)

        # 初始化显示第一页
        self._update_constraints_display()

        # 结果显示区域
        result_frame = ttk.LabelFrame(parent, text=" 计算结果 ", padding=10, style='Card.TFrame')
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5, side=tk.TOP)
        
        # 创建结果显示的Notebook
        self.result_notebook = ttk.Notebook(result_frame)
        self.result_notebook.pack(fill=tk.BOTH, expand=True)
        
        # 添加各个标签页
        self.ratio_frame = ttk.Frame(self.result_notebook)
        self.metric_frame = ttk.Frame(self.result_notebook)
        self.reflect_frame = ttk.Frame(self.result_notebook)
        
        self.result_notebook.add(self.ratio_frame, text="配比明细")
        self.result_notebook.add(self.metric_frame, text="质量指标")
        self.result_notebook.add(self.reflect_frame, text="反射率分布")

    def _update_constraints_display(self):
        """更新约束显示区域"""
        # 清除现有内容
        for widget in self.constraints_container.winfo_children():
            widget.destroy()
        
        # 清空旧的输入框列表
        self.ind_perc_coals_entries = []
        self.ind_perc_min_entries = []
        self.ind_perc_max_entries = []

        # 计算当前页的起始和结束索引
        start_idx = self.current_page * self.constraints_per_page
        end_idx = min(start_idx + self.constraints_per_page, self.num_individual_perc_constraints)

        # 显示当前页的约束
        for i in range(start_idx, end_idx):
            row_frame = ttk.Frame(self.constraints_container)
            row_frame.pack(fill=tk.X, pady=3)

            ttk.Label(row_frame, text=f"约束组{i+1}煤种:", style='SubHeader.TLabel', width=15).pack(side=tk.LEFT, anchor='w')
            
            coal_entry = ttk.Entry(row_frame, textvariable=self.ind_perc_coals_strs[i], width=25, font=('Microsoft YaHei', 9))
            coal_entry.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
            self.ind_perc_coals_entries.append(coal_entry)

            ttk.Label(row_frame, text="最小%:", style='SubHeader.TLabel').pack(side=tk.LEFT, padx=(5,0))
            min_entry = ttk.Entry(row_frame, textvariable=self.ind_perc_min_strs[i], width=5, font=('Microsoft YaHei', 9))
            min_entry.pack(side=tk.LEFT, padx=2)
            self.ind_perc_min_entries.append(min_entry)

            ttk.Label(row_frame, text="最大%:", style='SubHeader.TLabel').pack(side=tk.LEFT, padx=(5,0))
            max_entry = ttk.Entry(row_frame, textvariable=self.ind_perc_max_strs[i], width=5, font=('Microsoft YaHei', 9))
            max_entry.pack(side=tk.LEFT, padx=2)
            self.ind_perc_max_entries.append(max_entry)

        # 更新页码显示
        total_pages = (self.num_individual_perc_constraints + self.constraints_per_page - 1) // self.constraints_per_page
        self.page_label.config(text=f"第 {self.current_page + 1} 页，共 {total_pages} 页")

        # 更新按钮状态
        self.prev_page_btn.config(state='normal' if self.current_page > 0 else 'disabled')
        self.next_page_btn.config(state='normal' if self.current_page < total_pages - 1 else 'disabled')

    def _prev_page(self):
        """显示上一页约束"""
        if self.current_page > 0:
            self.current_page -= 1
            self._update_constraints_display()

    def _next_page(self):
        """显示下一页约束"""
        total_pages = (self.num_individual_perc_constraints + self.constraints_per_page - 1) // self.constraints_per_page
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self._update_constraints_display()

    def _bind_shortcuts(self):
        self.master.bind('<Control-o>', lambda e: self.load_coal_file())
        self.master.bind('<Return>', lambda e: self.run_optimization())

    def _parse_reflect_regions(self):
        """Wrapper for optimizer_core.parse_reflect_regions"""
        return optimizer_core.parse_reflect_regions()

    def show_about(self):
        about_text = """配煤系统"""

        messagebox.showinfo("关于", about_text)

    def reset_parameters(self):
        self.quantity_entry.delete(0, tk.END)
        for key in self.entries:
            if key == 'reflect_std_max':
                self.entries[key].delete(0, tk.END)
                self.entries[key].insert(0, str(self.constraints[key]))
            elif key == 'volatile_diff_max':
                self.entries[key].delete(0, tk.END)
                self.entries[key].insert(0, str(self.constraints[key]))
            else:
                self.entries[key][0].delete(0, tk.END)
                self.entries[key][0].insert(0, str(self.constraints[key][0]))
                self.entries[key][1].delete(0, tk.END)
                self.entries[key][1].insert(0, str(self.constraints[key][1]))
        
        # 清空所有多种独立煤种占比约束设置
        for i in range(self.num_individual_perc_constraints):
            self.ind_perc_coals_strs[i].set("")
            self.ind_perc_min_strs[i].set("")
            self.ind_perc_max_strs[i].set("")
        self.parsed_individual_perc_constraints = []
        
        # 重置页码到第一页
        self.current_page = 0
        self._update_constraints_display()
        
        self.status_bar.config(text=" 参数已重置 ")

    def load_coal_file(self):
        try:
            filepath = filedialog.askopenfilename(filetypes=[("CSV文件", "*.csv"), ("Excel文件", "*.xlsx")])
            if not filepath:
                return
            
            self.file_status.config(text=" 正在加载文件... ")
            self.status_bar.config(text=" 正在读取文件... ")
            
            if filepath.endswith('.csv'):
                self.df = pd.read_csv(filepath, encoding=self._detect_encoding(filepath))
            else:
                self.df = pd.read_excel(filepath)
            
            if self.df is not None and not self.df.empty:
                self._preprocess_data()
                self.file_status.config(text=f" 已加载：{filepath.split('/')[-1]} ")
                self.status_bar.config(text=" 文件加载成功 ")
            else:
                messagebox.showerror("文件错误", "加载的文件数据为空")
        except Exception as e:
            self.file_status.config(text=" 文件加载失败 ", style='StatusBar.TLabel')
            self.status_bar.config(text=f" 错误：{str(e)} ")
            messagebox.showerror("文件错误", f"文件加载失败：{str(e)}")

    def _detect_encoding(self, filepath):
        encodings = ['utf-8', 'gbk', 'gb18030', 'utf-16']
        for enc in encodings:
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    f.read(1024)
                return enc
            except UnicodeDecodeError:
                continue
        return 'utf-8'

    def _preprocess_data(self):
        regions = [
            '<0.5', '0.5-0.55', '0.55-0.6', '0.6-0.65', '0.65-0.7', '0.7-0.75', '0.75-0.8', '0.8-0.85',
            '0.85-0.9', '0.9-0.95', '0.95-1.0', '1.0-1.05', '1.05-1.1', '1.1-1.15', '1.15-1.2', '1.2-1.25',
            '1.25-1.3', '1.3-1.35', '1.35-1.4', '1.4-1.45', '1.45-1.5', '1.5-1.55', '1.55-1.6', '1.6-1.65',
            '1.65-1.7', '1.7-1.75', '1.75-1.8', '1.8-1.85', '1.85-1.9', '1.9-1.95', '1.95-2.0', '2.0-2.05',
            '2.05-2.1', '2.1-2.15', '2.15-2.2', '2.2-2.25', '2.25-2.3', '2.3-2.35', '2.35-2.4', '2.4-2.45',
            '2.45-2.5', '＞2.5'
        ]
        
        col_names = [region.replace('-', '_').replace('<', 'less_than_').replace('>', 'greater_than_') for region in regions]
        
        required_cols = [
            'coal_name', 'price', 'ash', 'sulfur', 'volatile', 'G_value', 'Y_value', 'CSR',
            'reflect_avg', 'reflect_std', 'stock', 'Re', 'F', 'b', 'ΔT', 'In', 'K', 'Na',
            'Tmax', 'C_act', 'V'
        ]
        required_cols.extend(col_names)
        
        for col in required_cols:
            if col not in self.df.columns:
                self.df[col] = 0.0 if col in col_names else np.nan
        
        self.df.fillna({
            'price': self.df['price'].mean(),
            'stock': self.df['stock'].max(),
            'reflect_avg': 1.0,
            'reflect_std': 0.1
        }, inplace=True)
        
        numeric_cols = required_cols[1:]
        self.df[numeric_cols] = self.df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        self.df.fillna(0, inplace=True)

    def _validate_inputs(self):
        if self.df is None or self.df.empty:
            messagebox.showwarning("输入错误", "请先选择煤种数据文件")
            return False
        
        if not self.quantity_entry.get().strip():
            messagebox.showwarning("输入错误", "请输入配煤数量")
            return False
        
        try:
            quantity = float(self.quantity_entry.get())
            if quantity <= 0:
                raise ValueError("配煤数量必须大于0")
        except ValueError as e:
            messagebox.showwarning("输入错误", f"无效的配煤数量：{str(e)}")
            return False
        
        return True

    def _update_constraints(self):
        try:
            for key in self.entries:
                if key == 'reflect_std_max':
                    value = self.entries[key].get()
                    if not value:
                        raise ValueError("反射率标准差未输入")
                    value = float(value)
                    if value <= 0:
                        raise ValueError("反射率标准差必须大于0")
                    self.constraints[key] = value
                elif key == 'volatile_diff_max':
                    value = self.entries[key].get()
                    if not value:
                        raise ValueError("最大挥发分差值未输入")
                    value = float(value)
                    if value < 0:
                        raise ValueError("最大挥发分差值不能为负数")
                    self.constraints[key] = value
                else:
                    min_val = self.entries[key][0].get()
                    max_val = self.entries[key][1].get()
                    if not min_val or not max_val:
                        raise ValueError(f"{key} 约束值未输入")
                    min_val = float(min_val)
                    max_val = float(max_val)
                    if min_val > max_val:
                        raise ValueError(f"{key} 下限值不能大于上限值")
                    self.constraints[key] = (min_val, max_val)
        except ValueError as e:
            messagebox.showerror("参数错误", f"约束条件设置错误：\n{str(e)}")
            raise

        # 解析多煤种独立占比约束
        parsed = []
        for i in range(self.num_individual_perc_constraints):
            coals_str = self.ind_perc_coals_strs[i].get().strip()
            min_perc_str = self.ind_perc_min_strs[i].get().strip()
            max_perc_str = self.ind_perc_max_strs[i].get().strip()
            if not coals_str:
                continue
            current_constraint_coals_list = [coal.strip() for coal in coals_str.split(',') if coal.strip()]
            if not current_constraint_coals_list:
                continue
            try:
                current_min_perc = float(min_perc_str) if min_perc_str else 0.0
                current_max_perc = float(max_perc_str) if max_perc_str else 100.0
                if not (0 <= current_min_perc <= 100 and 0 <= current_max_perc <= 100):
                    continue
                if current_min_perc > current_max_perc:
                    continue
            except ValueError:
                continue
            current_indices_in_df = []
            for coal_name in current_constraint_coals_list:
                if coal_name in self.df['coal_name'].values:
                    idx = self.df.index[self.df['coal_name'] == coal_name].tolist()[0]
                    current_indices_in_df.append(idx)
            if not current_indices_in_df:
                continue
            parsed.append({
                'coals_list': [self.df['coal_name'].iloc[k] for k in current_indices_in_df],
                'indices_in_df': current_indices_in_df,
                'min_perc': current_min_perc / 100.0,
                'max_perc': current_max_perc / 100.0
            })
        self.parsed_individual_perc_constraints = parsed

    def _build_constraints(self, quantity, df=None, main_volatile=None, max_diff=None):
        if df is None:
            df = self.df
        return optimizer_core.build_constraints(
            quantity,
            self.constraints,
            getattr(self, 'parsed_individual_perc_constraints', None),
            df,
            self.parsed_reflect_regions,
            main_volatile=main_volatile,
            max_diff=max_diff,
        )

    def _calculate_property(self, ratios, prop_type, df=None):
        if df is None:
            df = self.df
        return optimizer_core.calculate_property(
            ratios, prop_type, df, self.parsed_reflect_regions
        )

    def _calculate_reflect_std(self, ratios, df):
        return optimizer_core.calculate_reflect_std(
            ratios, df, self.parsed_reflect_regions
        )

    def _calculate_plastic_temp(self, ratios, coal_data):
        total = ratios.sum()
        if np.abs(total) < 1e-10:  # 修改这里，使用更安全的浮点数比较
            return 0.0
        return np.dot(ratios, coal_data['ΔT'].values) / total

    def update_progress(self, value, message="", detail=""):
        """更新进度条和状态信息 (修改为只更新状态栏)"""
        # self.progress_var.set(value) # 进度条UI已移除
        if message: # 只有在提供了message时才更新主状态栏
            self.status_bar.config(text=f" {message} ")
        
        # self.progress_label.config(text=message) # 进度标签UI已移除
        # if detail:
            # self.progress_text.insert(tk.END, f"{detail}\n") # 进度文本框UI已移除
            # self.progress_text.see(tk.END)
        
        # 如果有日志记录器并且提供了detail，可以考虑记录detail
        if hasattr(self, 'logger') and detail:
            self.logger.info(f"Progress Detail: {detail}")

        self.master.update()

    def _log_metric(self, category, metric, value):
        """基础日志记录方法，允许子类重写增强功能"""
        if hasattr(self, 'logger'):
            if isinstance(value, dict):
                for k, v in value.items():
                    self.logger.info(f"{category} - {metric} - {k}: {v}")
            else:
                self.logger.info(f"{category} - {metric}: {value}")
        return

    def _prepare_ml_data(self):
        """准备机器学习模型的训练数据，专注于残差学习"""
        if self.history_data is None or self.history_data.empty:
            print("错误：历史数据为空")
            messagebox.showwarning("警告", "没有历史数据用于训练")
            return None, None
            
        total_rows = len(self.history_data)
        self.update_progress(0, "准备训练数据...", f"开始处理{total_rows}行数据")
        
        # 使用向量化操作优化数据处理
        X_data = []
        y_data = {'G': [], 'CSR': [], 'Y': []}
        
        # 预先计算所有煤种数据
        coal_data_dict = {name: data for name, data in self.df.groupby('coal_name')}
        
        for idx, row in self.history_data.iterrows():
            try:
                progress = (idx + 1) / total_rows * 100
                self.update_progress(progress, f"处理第{idx+1}/{total_rows}行数据...")
                
                coal_names = row['coal_names']
                ratios = np.array(row['ratios'])  # 确保是numpy数组
                total = ratios.sum()
                
                if np.abs(total - 1.0) > 1e-6:
                    continue
                    
                # 使用字典快速获取煤种数据
                row_coal_data = pd.concat([coal_data_dict[name] for name in coal_names])
                if len(row_coal_data) != len(coal_names):
                    continue
                
                # 使用向量化操作计算特征
                coal_values = row_coal_data[['G_value', 'CSR', 'Y_value', 'ash', 'sulfur', 'volatile', 'reflect_avg', 'V']].values
                weighted_values = np.dot(ratios, coal_values) / total
                
                # 计算物理模型预测值
                phy_G = self._calculate_advanced_property('G_value', ratios)
                phy_CSR = self._calculate_advanced_property('CSR', ratios)
                phy_Y = self._calculate_advanced_property('Y_value', ratios)
                
                # 计算残差
                residual_G = row['G_actual'] - phy_G
                residual_CSR = row['CSR_actual'] - phy_CSR
                residual_Y = row['Y_actual'] - phy_Y
                
                # 构建特征字典
                features = {
                    'phy_G': phy_G,
                    'phy_CSR': phy_CSR,
                    'phy_Y': phy_Y,
                    'G_base': weighted_values[0],
                    'CSR_base': weighted_values[1],
                    'Y_base': weighted_values[2],
                    'ash': weighted_values[3],
                    'sulfur': weighted_values[4],
                    'volatile': weighted_values[5],
                    'reflect_avg': weighted_values[6],
                    'V': weighted_values[7],
                    'reflect_std': self._calculate_reflect_std(ratios, row_coal_data),
                    'ratio_std': np.std(ratios),
                    'ratio_max': np.max(ratios),
                    'ratio_min': np.min(ratios),
                    'ratio_range': np.max(ratios) - np.min(ratios),
                    'num_coals': len(coal_names),
                    'entropy': -np.sum(ratios * np.log(ratios + 1e-10))
                }
                
                # 添加主煤特征
                main_coal_idx = np.argmax(ratios)
                main_coal = row_coal_data.iloc[main_coal_idx]
                features.update({
                    'main_coal_ratio': ratios[main_coal_idx],
                    'main_coal_G': main_coal['G_value'],
                    'main_coal_CSR': main_coal['CSR'],
                    'main_coal_Y': main_coal['Y_value']
                })
                
                # 添加高级特征
                advanced_features = {
                    'plastic_temp': self._calculate_plastic_temp(ratios, row_coal_data),
                    'weighted_tmax': np.dot(ratios, row_coal_data['Tmax'].values) / total,
                    'weighted_cact': np.dot(ratios, row_coal_data['C_act'].values) / total,
                    'weighted_re': np.dot(ratios, row_coal_data['Re'].values) / total,
                    'weighted_f': np.dot(ratios, row_coal_data['F'].values) / total,
                    'weighted_b': np.dot(ratios, row_coal_data['b'].values) / total
                }
                features.update(advanced_features)
                
                # 添加非线性组合特征
                features.update({
                    'G_CSR_ratio': weighted_values[0] / (weighted_values[1] + 1e-10),
                    'ash_vol_ratio': weighted_values[3] / (weighted_values[5] + 1e-10),
                    'V_vol_ratio': weighted_values[7] / (weighted_values[5] + 1e-10),
                    'reflect_tmax_ratio': weighted_values[6] / (advanced_features['weighted_tmax'] + 1e-10)
                })
                
                X_data.append(features)
                y_data['G'].append(residual_G)
                y_data['CSR'].append(residual_CSR)
                y_data['Y'].append(residual_Y)
                
            except Exception as e:
                self.update_progress(progress, f"处理第{idx+1}行数据时出错", str(e))
                continue
        
        if not X_data:
            self.update_progress(0, "错误：没有成功处理任何数据")
            messagebox.showwarning("警告", "没有有效的训练数据")
            return None, None
            
        self.update_progress(100, "数据处理完成", f"成功处理{len(X_data)}行数据")
        return pd.DataFrame(X_data), {k: np.array(v) for k, v in y_data.items()}

    def _calculate_ml_model(self, prop_type, ratios):
        """使用训练好的模型预测残差"""
        try:
            if prop_type == 'G_value':
                target = 'G'
            elif prop_type == 'CSR':
                target = 'CSR'
            elif prop_type == 'Y_value':
                target = 'Y'
            else:
                return None
                
            if target not in self.ml_models:
                messagebox.showwarning("警告", f"没有训练好的{target}残差模型")
                return None
                
            # 只获取非零配比对应的煤种数据
            active_indices = np.where(ratios > 0)[0]
            active_ratios = ratios[active_indices]
            active_coal_data = self.df.iloc[active_indices].copy()
            total = active_ratios.sum()
            
            if np.abs(total) < 1e-10:
                return 0.0
                
            # 计算物理模型预测值
            phy_value = self._calculate_advanced_property(prop_type, active_ratios)
            if phy_value is None:
                return np.dot(active_ratios, active_coal_data[prop_type.lower()].values) / total
                
            # 构建特征
            features = {
                # 物理模型预测值
                'phy_G': self._calculate_advanced_property('G_value', active_ratios),
                'phy_CSR': self._calculate_advanced_property('CSR', active_ratios),
                'phy_Y': self._calculate_advanced_property('Y_value', active_ratios),
                
                # 基础加权平均特征
                'G_base': np.dot(active_ratios, active_coal_data['G_value'].values) / total,
                'CSR_base': np.dot(active_ratios, active_coal_data['CSR'].values) / total,
                'Y_base': np.dot(active_ratios, active_coal_data['Y_value'].values) / total,
                
                # 煤质特征
                'ash': np.dot(active_ratios, active_coal_data['ash'].values) / total,
                'sulfur': np.dot(active_ratios, active_coal_data['sulfur'].values) / total,
                'volatile': np.dot(active_ratios, active_coal_data['volatile'].values) / total,
                'reflect_avg': np.dot(active_ratios, active_coal_data['reflect_avg'].values) / total,
                'V': np.dot(active_ratios, active_coal_data['V'].values) / total,
                'reflect_std': self._calculate_reflect_std(active_ratios, active_coal_data),
                
                # 配比特征
                'ratio_std': np.std(active_ratios),
                'ratio_max': np.max(active_ratios),
                'ratio_min': np.min(active_ratios),
                'ratio_range': np.max(active_ratios) - np.min(active_ratios),
                'num_coals': len(active_ratios),
                'entropy': -np.sum(active_ratios * np.log(active_ratios + 1e-10)),
                
                # 主煤特征
                'main_coal_ratio': np.max(active_ratios),
                'main_coal_G': active_coal_data.loc[active_coal_data.index[np.argmax(active_ratios)], 'G_value'],
                'main_coal_CSR': active_coal_data.loc[active_coal_data.index[np.argmax(active_ratios)], 'CSR'],
                'main_coal_Y': active_coal_data.loc[active_coal_data.index[np.argmax(active_ratios)], 'Y_value'],
                
                # 高级特征
                'plastic_temp': self._calculate_plastic_temp(active_ratios, active_coal_data),
                'weighted_tmax': np.dot(active_ratios, active_coal_data['Tmax'].values) / total,
                'weighted_cact': np.dot(active_ratios, active_coal_data['C_act'].values) / total,
                'weighted_re': np.dot(active_ratios, active_coal_data['Re'].values) / total,
                'weighted_f': np.dot(active_ratios, active_coal_data['F'].values) / total,
                'weighted_b': np.dot(active_ratios, active_coal_data['b'].values) / total,
                
                # 非线性组合特征
                'G_CSR_ratio': np.dot(active_ratios, active_coal_data['G_value'].values) / (np.dot(active_ratios, active_coal_data['CSR'].values) + 1e-10),
                'ash_vol_ratio': np.dot(active_ratios, active_coal_data['ash'].values) / (np.dot(active_ratios, active_coal_data['volatile'].values) + 1e-10),
                'V_vol_ratio': np.dot(active_ratios, active_coal_data['V'].values) / (np.dot(active_ratios, active_coal_data['volatile'].values) + 1e-10),
                'reflect_tmax_ratio': np.dot(active_ratios, active_coal_data['reflect_avg'].values) / (np.dot(active_ratios, active_coal_data['Tmax'].values) + 1e-10)
            }
            
            # 添加二阶交互特征
            props = ['ash', 'sulfur', 'volatile', 'reflect_avg', 'V']
            for i in range(len(props)):
                for j in range(i+1, len(props)):
                    name = f'{props[i]}_{props[j]}_inter'
                    features[name] = (np.dot(active_ratios, active_coal_data[props[i]].values) * 
                                    np.dot(active_ratios, active_coal_data[props[j]].values)) / (total ** 2)
            
            # 转换为DataFrame并确保特征顺序一致
            X_pred = pd.DataFrame([features])[self.feature_names[target]]
            
            # 标准化特征
            X_scaled = self.scalers[target].transform(X_pred)
            
            # 预测残差
            residual = self.ml_models[target].predict(X_scaled)[0]
            
            # 返回物理模型预测值加上残差
            return phy_value + residual
            
        except Exception as e:
            messagebox.showerror("错误", f"预测残差时出错: {str(e)}")
            return None

    def run_optimization(self):
        """主煤遍历+分组优化：每次以一个煤为主煤，筛选所有与主煤挥发分差值在max_diff内的煤种，分别做优化，选出最优解"""
        if not self._validate_inputs():
            return
        try:
            start_time = datetime.now()
            self.status_bar.config(text=" 正在执行分组优化... ")
            self.master.update()
            if self.df is None or self.df.empty:
                messagebox.showerror("数据错误", "煤种数据文件未正确加载或数据为空")
                return
            self._update_constraints()
            quantity = float(self.quantity_entry.get())
            max_diff = self.constraints.get('volatile_diff_max', None)
            if max_diff is None:
                messagebox.showerror("参数错误", "请设置最大挥发分差值")
                return
            best_solution = None
            best_cost = float('inf')
            best_metrics = None
            best_main_coal = None
            best_df = None
            total_main = len(self.df)
            for idx, (i, main_coal) in enumerate(self.df.iterrows()):
                main_volatile = main_coal['volatile']
                print(f"[主煤进度] {idx+1}/{total_main} 主煤: {main_coal['coal_name']} 挥发分: {main_volatile}")
                # 找到所有与主煤挥发分差值在max_diff内的煤
                valid_indices = self.df.index[abs(self.df['volatile'] - main_volatile) <= max_diff].tolist()
                if len(valid_indices) == 0:
                    continue
                sub_df = self.df.iloc[valid_indices].reset_index(drop=True)
                print(f"  子集煤种: {list(sub_df['coal_name'])}")
                print(f"  子集挥发分区间: {sub_df['volatile'].min()} ~ {sub_df['volatile'].max()}")
                # 递归剔除法优化，传递主煤名
                res = optimizer_core.find_best_volatile_solution(
                    sub_df,
                    quantity,
                    self.constraints,
                    self.parsed_individual_perc_constraints,
                    self.parsed_reflect_regions,
                    self,
                    max_diff,
                    depth=1,
                    main_coal_name=main_coal['coal_name'],
                    main_volatile=main_volatile
                )
                if res:
                    solution, metrics, cost, sub_df_result = res
                    print(f"    [主煤:{main_coal['coal_name']}] 可行解 成本: {cost:.2f}")
                    if cost < best_cost:
                        best_cost = cost
                        best_solution = solution
                        best_main_coal = main_coal['coal_name'] if 'coal_name' in main_coal else str(i)
                        best_df = sub_df_result
                        best_metrics = metrics
            if best_solution is not None:
                # 用最优子集df更新结果面板（不再修改self.df）
                self._update_result_panels(best_solution, best_df)
                self.status_bar.config(text=f" 计算完成，主煤：{best_main_coal}")
                print(f"[最终最优解] 主煤: {best_main_coal} 成本: {best_cost:.2f}")
            else:
                messagebox.showerror("优化失败", "所有主煤分组均无法找到可行解，请检查约束设置和煤种数据")
        except Exception as e:
            error_msg = f"分组优化过程出现错误：{str(e) if e else '未知错误'}"
            messagebox.showerror("计算错误", error_msg)
            if hasattr(self, 'logger'):
                self.logger.error(error_msg)
        finally:
            self.master.update()

    def _postprocess_solution(self, solution, use_integer, precision=1, df=None):
        """配比百分比严格取整，且和为100，吨数也为整数"""
        original_total = solution.sum()
        if use_integer:
            # 1. 计算原始百分比和小数部分
            ratio_percent_raw = solution / original_total * 100
            ratio_percent_int = np.floor(ratio_percent_raw).astype(int)
            remainder = ratio_percent_raw - ratio_percent_int
            # 2. 分配剩余到最大的小数部分，保证和为100
            diff = 100 - np.sum(ratio_percent_int)
            if diff > 0:
                idx = np.argsort(-remainder)[:diff]
                ratio_percent_int[idx] += 1
            elif diff < 0: # 如果取整后总和大于100，则从小数部分最小的开始减，直到和为100
                idx = np.argsort(remainder)[:abs(diff)]
                ratio_percent_int[idx] -=1

            # 确保所有配比都是非负的
            ratio_percent_int = np.maximum(0, ratio_percent_int)
            # 再次归一化以防万一，并确保和为100
            if ratio_percent_int.sum() != 100 and ratio_percent_int.sum() > 0 :
                 ratio_percent_int = np.round(ratio_percent_int / ratio_percent_int.sum() * 100).astype(int)
                 # 如果二次归一化后和仍不为100，则再次使用最大余数法
                 current_sum = ratio_percent_int.sum()
                 if current_sum != 100:
                    diff_after_re_norm = 100 - current_sum
                    if diff_after_re_norm > 0:
                        idx_re_norm = np.argsort(-remainder)[:diff_after_re_norm] # 使用原始的remainder
                        ratio_percent_int[idx_re_norm] += 1
                    elif diff_after_re_norm < 0:
                        idx_re_norm = np.argsort(remainder)[:abs(diff_after_re_norm)] # 使用原始的remainder
                        ratio_percent_int[idx_re_norm] -=1
            # 3. 还原为吨数并根据 self.rounding_precision 进行取整
            tons_from_int_percent = ratio_percent_int / 100 * original_total
            num_decimals_to_round = self.rounding_precision.get() # 0 for integer, 1 for 1 decimal place
            if num_decimals_to_round == 0:
                base_rounded = np.round(tons_from_int_percent).astype(int) # 取整到整数
            else:
                base_rounded = np.round(tons_from_int_percent, num_decimals_to_round)
        else:
            # 如果不启用取整，则直接对用量取整到两位小数
            base_rounded = np.round(solution, 2)
        # 校验解决方案（所有约束）
        violation = self._calculate_constraint_violation(base_rounded, original_total, df)
        if violation > 0:
            print("取整/后处理后解不满足所有约束，违规值：", violation)
            for prop in ['ash', 'sulfur', 'volatile', 'G_value', 'Y_value', 'CSR']:
                value = self._calculate_property(base_rounded, prop, df if df is not None else self.df)
                print(f"{prop}: {value}, 约束: {self.constraints[prop]}")
            reflect_std = self._calculate_reflect_std(base_rounded, df if df is not None else self.df)
            print(f"reflect_std_max: {reflect_std}, 约束: {self.constraints['reflect_std_max']}")
            if hasattr(self, 'parsed_individual_perc_constraints') and self.parsed_individual_perc_constraints:
                for c in self.parsed_individual_perc_constraints:
                    idxs = c['indices_in_df']
                    minp = c['min_perc']
                    maxp = c['max_perc']
                    total = base_rounded.sum()
                    perc = np.sum(base_rounded[idxs]) / (total + 1e-10)
                    print(f"多煤种占比: {perc}, 约束: [{minp}, {maxp}]")
            if 'volatile_diff_max' in self.constraints:
                used_idx = [i for i, v in enumerate(base_rounded) if v > 0.01]
                if used_idx:
                    used_v = [(df if df is not None else self.df).iloc[i]['volatile'] for i in used_idx]
                    vmin, vmax = min(used_v), max(used_v)
                    diff = vmax - vmin
                    print(f"实际挥发分区间: {vmin}~{vmax}, 差值: {diff}, 约束: {self.constraints['volatile_diff_max']}")
            raise Exception("取整/后处理后无可行解，所有约束必须严格满足！")
        # 如果基础取整解无效，尝试智能调整
        return self._smart_adjustment(
            solution, 
            base_rounded, # 传入的是已经按配比取整并还原的吨数
            original_total,
            precision=1, # 智能调整时，仍然基于整数吨位进行
            df=df
        )
    
    def _smart_adjustment(self, original, base_rounded, target_total, precision=1, df=None):
        """基于蒙特卡洛树搜索的智能调整算法（调整吨数，逼近目标总量）"""
        from collections import deque
        best_solution = base_rounded # 初始最优解是经过配比取整的解
        min_violation = self._calculate_constraint_violation(base_rounded, target_total, df)

        if min_violation <= precision * 0.01: # 如果初始解已经很好，直接返回
            return best_solution

        candidates = deque([base_rounded.copy()])
        visited = set()
        # 将solution转换为可哈希的元组
        visited.add(tuple(np.round(base_rounded).astype(int)))

        max_attempts = 1000
        # tolerance = precision * 0.01 # 允许的微小误差，这里precision是1（吨）
        # 对于吨数调整，总量差异小于0.5吨即可接受
        tolerance = 0.5 

        for _ in range(max_attempts):
            if not candidates:
                break
                
            current_solution_tons = candidates.popleft()
            
            # 评估当前解（基于吨数）
            current_violation = self._calculate_constraint_violation(current_solution_tons, target_total, df)
            
            if current_violation <= tolerance:
                return current_solution_tons # 找到满足总量约束的解
                
            if current_violation < min_violation:
                best_solution = current_solution_tons.copy()
                min_violation = current_violation
            
            for idx in range(len(current_solution_tons)):
                
                # 向上调整吨数
                new_up_tons = current_solution_tons.copy()
                new_up_tons[idx] += precision # precision是1吨
                new_up_tons_tuple = tuple(np.round(new_up_tons).astype(int))
                if new_up_tons_tuple not in visited:
                    candidates.append(new_up_tons)
                    visited.add(new_up_tons_tuple)
                
                # 向下调整吨数
                new_down_tons = current_solution_tons.copy()
                if new_down_tons[idx] >= precision:
                    new_down_tons[idx] -= precision # precision是1吨
                    new_down_tons_tuple = tuple(np.round(new_down_tons).astype(int))
                    if new_down_tons_tuple not in visited:
                        candidates.append(new_down_tons)
                        visited.add(new_down_tons_tuple)
                        
        # 无论是否完全满足约束，都返回最优近似解
        return best_solution

    def _calculate_constraint_violation(self, solution, target_total, df=None):
        if df is None:
            df = self.df
        return optimizer_core.calculate_constraint_violation(
            solution,
            target_total,
            self.constraints,
            df,
            getattr(self, 'parsed_individual_perc_constraints', None),
            self.parsed_reflect_regions,
        )

    def _validate_solution(self, solution, target_total, df=None):
        try:
            if not np.isclose(solution.sum(), target_total, rtol=0.001):
                return False
            return True
        except:
            return False

    def _update_result_panels(self, solution, df=None):
        # 清空现有内容
        for widget in self.ratio_frame.winfo_children():
            widget.destroy()
        for widget in self.metric_frame.winfo_children():
            widget.destroy()
        for widget in self.reflect_frame.winfo_children():
            widget.destroy()
        # 检查是否已存在反射率区间占比表tab，没有则添加
        if not hasattr(self, 'reflect_table_frame'):
            self.reflect_table_frame = ttk.Frame(self.result_notebook)
            self.result_notebook.add(self.reflect_table_frame, text="反射率区间占比表")
        else:
            for widget in self.reflect_table_frame.winfo_children():
                widget.destroy()
        # 更新各个面板
        self._build_ratio_table(self.ratio_frame, solution, df)
        self._build_metric_panel(self.metric_frame, solution, df)
        self._build_reflect_chart(self.reflect_frame, solution, df)
        self._build_reflect_table(self.reflect_table_frame, solution, df)
        # 切换到第一个标签页
        self.result_notebook.select(0)
        # 添加精度标记
        summary_frame = ttk.Frame(self.ratio_frame)
        summary_frame.pack(fill=tk.X, padx=5, pady=5)
        if self.integer_constraint.get():
            label_text = f"结果已按整数配比%取整"
            ttk.Label(summary_frame, text=label_text, 
                     foreground="#3498DB", font=("Arial", 9, "bold")).pack(side=tk.RIGHT)
        violation = self._calculate_constraint_violation(solution, solution.sum(), df)
        if violation < 0.5:
            status = "已按整数配比%取整，总量基本满足"
            color = "#27AE60"
        else:
            status = f"已按整数配比%取整，但总量存在偏差 ({violation:.2f}吨)"
            color = "#E67E22"
        ttk.Label(summary_frame, text=status, foreground=color).pack(side=tk.RIGHT, padx=10)

    def _build_reflect_table(self, parent, solution, df=None):
        if df is None:
            df = self.df
        if df is None or df.empty:
            return
        # 区间与列名
        regions = [
            '<0.5', '0.5-0.55', '0.55-0.6', '0.6-0.65', '0.65-0.7', '0.7-0.75', '0.75-0.8', '0.8-0.85',
            '0.85-0.9', '0.9-0.95', '0.95-1.0', '1.0-1.05', '1.05-1.1', '1.1-1.15', '1.15-1.2', '1.2-1.25',
            '1.25-1.3', '1.3-1.35', '1.35-1.4', '1.4-1.45', '1.45-1.5', '1.5-1.55', '1.55-1.6', '1.6-1.65',
            '1.65-1.7', '1.7-1.75', '1.75-1.8', '1.8-1.85', '1.85-1.9', '1.9-1.95', '1.95-2.0', '2.0-2.05',
            '2.05-2.1', '2.1-2.15', '2.15-2.2', '2.2-2.25', '2.25-2.3', '2.3-2.35', '2.35-2.4', '2.4-2.45',
            '2.45-2.5', '＞2.5'
        ]
        col_names = [region.replace('-', '_').replace('<', 'less_than_').replace('>', 'greater_than_') for region in regions]
        total = sum(solution)
        raw_props = []
        for col in col_names:
            prop = np.dot(solution, df[col]) / total if total != 0 else 0
            raw_props.append(prop)
        sum_props = sum(raw_props)
        if sum_props > 0:
            proportions = [p / sum_props * 100 for p in raw_props]
        else:
            proportions = [0 for _ in raw_props]
        # 只保留有占比的区间
        show_regions = []
        show_props = []
        for region, prop in zip(regions, proportions):
            if prop > 0:
                show_regions.append(region)
                show_props.append(prop)
        # 带横向滚动条的表格
        for widget in parent.winfo_children():
            widget.destroy()
        canvas = tk.Canvas(parent, highlightthickness=0)
        h_scroll = ttk.Scrollbar(parent, orient="horizontal", command=canvas.xview)
        canvas.configure(xscrollcommand=h_scroll.set)
        canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        table_frame = ttk.Frame(canvas)
        # 第一行：区间
        header_row = ttk.Frame(table_frame)
        header_row.pack(fill=tk.X)
        ttk.Label(header_row, text="反射率区间", width=12, anchor=tk.CENTER, background="#273746", foreground="white", font=("Microsoft YaHei", 10, "bold"), relief="solid", borderwidth=1, padding=(4,6)).pack(side=tk.LEFT)
        for region in show_regions:
            ttk.Label(header_row, text=region, width=10, anchor=tk.CENTER, background="#F8F9F9", font=("Microsoft YaHei", 10, "bold"), relief="solid", borderwidth=1, padding=(4,6)).pack(side=tk.LEFT)
        # 第二行：占比
        value_row = ttk.Frame(table_frame)
        value_row.pack(fill=tk.X)
        ttk.Label(value_row, text="占比(%)", width=12, anchor=tk.CENTER, background="#273746", foreground="white", font=("Microsoft YaHei", 10, "bold"), relief="solid", borderwidth=1, padding=(4,6)).pack(side=tk.LEFT)
        for prop in show_props:
            ttk.Label(value_row, text=f"{prop:.2f}", width=10, anchor=tk.CENTER, background="#F8F9F9", font=("Microsoft YaHei", 10), relief="solid", borderwidth=1, padding=(4,6)).pack(side=tk.LEFT)
        # 放入canvas
        table_frame.update_idletasks()
        canvas.create_window((0, 0), window=table_frame, anchor="nw")
        canvas.config(scrollregion=canvas.bbox("all"))

    def _validate_inputs(self):
        if self.df is None or self.df.empty:
            messagebox.showwarning("输入错误", "请先选择煤种数据文件")
            return False
        
        if not self.quantity_entry.get().strip():
            messagebox.showwarning("输入错误", "请输入配煤数量")
            return False
        
        try:
            quantity = float(self.quantity_entry.get())
            if quantity <= 0:
                raise ValueError("配煤数量必须大于0")
        except ValueError as e:
            messagebox.showwarning("输入错误", f"无效的配煤数量：{str(e)}")
            return False
        
        return True

    def _update_constraints(self):
        try:
            for key in self.entries:
                if key == 'reflect_std_max':
                    value = self.entries[key].get()
                    if not value:
                        raise ValueError("反射率标准差未输入")
                    value = float(value)
                    if value <= 0:
                        raise ValueError("反射率标准差必须大于0")
                    self.constraints[key] = value
                elif key == 'volatile_diff_max':
                    value = self.entries[key].get()
                    if not value:
                        raise ValueError("最大挥发分差值未输入")
                    value = float(value)
                    if value < 0:
                        raise ValueError("最大挥发分差值不能为负数")
                    self.constraints[key] = value
                else:
                    min_val = self.entries[key][0].get()
                    max_val = self.entries[key][1].get()
                    if not min_val or not max_val:
                        raise ValueError(f"{key} 约束值未输入")
                    min_val = float(min_val)
                    max_val = float(max_val)
                    if min_val > max_val:
                        raise ValueError(f"{key} 下限值不能大于上限值")
                    self.constraints[key] = (min_val, max_val)
        except ValueError as e:
            messagebox.showerror("参数错误", f"约束条件设置错误：\n{str(e)}")
            raise

        # 解析多煤种独立占比约束
        parsed = []
        for i in range(self.num_individual_perc_constraints):
            coals_str = self.ind_perc_coals_strs[i].get().strip()
            min_perc_str = self.ind_perc_min_strs[i].get().strip()
            max_perc_str = self.ind_perc_max_strs[i].get().strip()
            if not coals_str:
                continue
            current_constraint_coals_list = [coal.strip() for coal in coals_str.split(',') if coal.strip()]
            if not current_constraint_coals_list:
                continue
            try:
                current_min_perc = float(min_perc_str) if min_perc_str else 0.0
                current_max_perc = float(max_perc_str) if max_perc_str else 100.0
                if not (0 <= current_min_perc <= 100 and 0 <= current_max_perc <= 100):
                    continue
                if current_min_perc > current_max_perc:
                    continue
            except ValueError:
                continue
            current_indices_in_df = []
            for coal_name in current_constraint_coals_list:
                if coal_name in self.df['coal_name'].values:
                    idx = self.df.index[self.df['coal_name'] == coal_name].tolist()[0]
                    current_indices_in_df.append(idx)
            if not current_indices_in_df:
                continue
            parsed.append({
                'coals_list': [self.df['coal_name'].iloc[k] for k in current_indices_in_df],
                'indices_in_df': current_indices_in_df,
                'min_perc': current_min_perc / 100.0,
                'max_perc': current_max_perc / 100.0
            })
        self.parsed_individual_perc_constraints = parsed

    def _build_ratio_table(self, parent, solution, df=None):
        if df is None:
            df = self.df
        if df is None or df.empty:
            return
        # 创建表格框架
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        # 创建滚动条
        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        hsb = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        # 新表头
        columns = ('coal_name', 'ash', 'sulfur', 'volatile', 'G_value', 'csr', 'ratio_percent', 'unit_cost')
        tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show='headings',
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set
        )
        # 设置列标题
        tree.heading('coal_name', text='煤种名称', anchor=tk.W)
        tree.heading('ash', text='灰分(%)', anchor=tk.E)
        tree.heading('sulfur', text='硫分(%)', anchor=tk.E)
        tree.heading('volatile', text='挥发分(%)', anchor=tk.E)
        tree.heading('G_value', text='G值', anchor=tk.E)
        tree.heading('csr', text='CSR', anchor=tk.E)
        tree.heading('ratio_percent', text='配比%', anchor=tk.E)
        tree.heading('unit_cost', text='单煤成本(元/吨)', anchor=tk.E)
        # 设置列宽
        tree.column('coal_name', width=180, minwidth=120)
        tree.column('ash', width=80, minwidth=60, anchor=tk.E)
        tree.column('sulfur', width=80, minwidth=60, anchor=tk.E)
        tree.column('volatile', width=80, minwidth=60, anchor=tk.E)
        tree.column('G_value', width=80, minwidth=60, anchor=tk.E)
        tree.column('csr', width=80, minwidth=60, anchor=tk.E)
        tree.column('ratio_percent', width=80, minwidth=60, anchor=tk.E)
        tree.column('unit_cost', width=110, minwidth=90, anchor=tk.E)
        # 计算配比百分比
        total_quantity = sum(solution)
        if self.integer_constraint.get():
            # 取整配比%（和为100）
            ratio_percent_raw = np.array(solution) / total_quantity * 100
            ratio_percent_int = np.floor(ratio_percent_raw).astype(int)
            remainder = ratio_percent_raw - ratio_percent_int
            diff = 100 - np.sum(ratio_percent_int)
            if diff > 0:
                idx = np.argsort(-remainder)[:diff]
                ratio_percent_int[idx] += 1
            elif diff < 0:
                idx = np.argsort(remainder)[:abs(diff)]
                ratio_percent_int[idx] -= 1
            ratio_percent_display = ratio_percent_int
        else:
            ratio_percent_display = np.round(np.array(solution) / total_quantity * 100, 2)
        # 添加数据
        for idx, ratio in enumerate(solution):
            if ratio > 0.01:
                coal = df.iloc[idx]
                ratio_percent = ratio_percent_display[idx]
                unit_cost = coal['price'] if pd.notna(coal['price']) else 0
                # 取煤种的各项指标
                ash = coal['ash'] if pd.notna(coal['ash']) else ''
                sulfur = coal['sulfur'] if pd.notna(coal['sulfur']) else ''
                volatile = coal['volatile'] if pd.notna(coal['volatile']) else ''
                G_value = coal['G_value'] if pd.notna(coal['G_value']) else ''
                # 修正CSR字段的获取逻辑，优先小写csr
                if 'csr' in coal and pd.notna(coal['csr']):
                    csr = coal['csr']
                elif 'CSR' in coal and pd.notna(coal['CSR']):
                    csr = coal['CSR']
                else:
                    csr = ''
                tree.insert('', 'end', values=(
                    coal['coal_name'] if pd.notna(coal['coal_name']) else '未知煤种',
                    f"{ash}",
                    f"{sulfur}",
                    f"{volatile}",
                    f"{G_value}",
                    f"{csr}",
                    f"{ratio_percent}",
                    f"{unit_cost:.1f}"
                ))
        # 配置滚动条
        vsb.config(command=tree.yview)
        hsb.config(command=tree.xview)
        # 布局表格和滚动条
        tree.grid(row=0, column=0, sticky=tk.NSEW)
        vsb.grid(row=0, column=1, sticky=tk.NS)
        hsb.grid(row=1, column=0, sticky=tk.EW)
        tree_frame.grid_columnconfigure(0, weight=1)
        tree_frame.grid_rowconfigure(0, weight=1)
        # ====== 配煤总成本和单吨成本 ======
        # 只统计实际用到的煤
        total_cost = 0.0
        for idx, ratio in enumerate(solution):
            if ratio > 0.01:
                coal = df.iloc[idx]
                price = coal['price'] if pd.notna(coal['price']) else 0
                total_cost += ratio * price
        unit_cost = total_cost / total_quantity if total_quantity > 0 else 0
        cost_frame = ttk.Frame(parent)
        cost_frame.pack(fill=tk.X, padx=5, pady=(0, 8))
        ttk.Label(cost_frame, text=f"配煤总成本: {total_cost:,.2f} 元", font=("Microsoft YaHei", 11, "bold"), foreground="#C0392B").pack(side=tk.LEFT, padx=10)
        ttk.Label(cost_frame, text=f"单吨成本: {unit_cost:,.2f} 元/吨", font=("Microsoft YaHei", 11, "bold"), foreground="#2874A6").pack(side=tk.LEFT, padx=10)

    def _build_metric_panel(self, parent, solution, df=None):
        if df is None:
            df = self.df
        if df is None or df.empty:
            return
        # 恢复主界面竖排指标展示
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        metrics = {
            '灰分 (%)': ('ash', lambda x: np.dot(x, df['ash']) / x.sum()),
            '硫分 (%)': ('sulfur', lambda x: np.dot(x, df['sulfur']) / x.sum()),
            '挥发分 (%)': ('volatile', lambda x: np.dot(x, df['volatile']) / x.sum()),
            'G值': ('G_value', lambda x: self._calculate_property(x, 'G_value', df)),
            'Y值 (mm)': ('Y_value', lambda x: self._calculate_property(x, 'Y_value', df)),
            'CSR': ('CSR', lambda x: self._calculate_property(x, 'CSR', df)),
            '反射率标准差': ('reflect_std', lambda x: self._calculate_reflect_std(x, df)),
            '最大挥发分差值': ('volatile_diff', lambda x: (df['volatile'][x > 0.01].max() - df['volatile'][x > 0.01].min()) if np.any(x > 0.01) else 0)
        }
        for i, (name, (key, func)) in enumerate(metrics.items()):
            frame = ttk.Frame(main_frame)
            frame.pack(fill=tk.X, pady=3)
            ttk.Label(frame, text=name, width=14, anchor=tk.E).pack(side=tk.LEFT)
            value = func(solution)
            display_value = value if pd.notna(value) else 0
            ttk.Label(frame, text=f"{display_value:.4f}", width=10, font=('Microsoft YaHei', 9, 'bold')).pack(side=tk.LEFT)
            if key == 'reflect_std':
                constraint = f"≤ {self.constraints['reflect_std_max']:.4f}"
                color = '#2ECC71' if (display_value <= self.constraints['reflect_std_max']) else '#E74C3C'
            elif key == 'volatile_diff':
                if 'volatile_diff_max' in self.constraints:
                    constraint = f"≤ {self.constraints['volatile_diff_max']:.2f}"
                    color = '#2ECC71' if (display_value <= self.constraints['volatile_diff_max']) else '#E74C3C'
                else:
                    constraint = "-"
                    color = '#2C3E50'
            elif key != 'reflect_std':
                constraint = f"[{self.constraints[key][0]:.2f} - {self.constraints[key][1]:.2f}]"
                color = '#2ECC71' if (self.constraints[key][0] <= display_value <= self.constraints[key][1]) else '#E74C3C'
            else:
                constraint = ""
                color = '#2C3E50'
            ttk.Label(frame, text=constraint, foreground=color, font=('Microsoft YaHei', 9)).pack(side=tk.LEFT, padx=10)

    def _build_reflect_chart(self, parent, solution, df=None):
        if df is None:
            df = self.df
        if df is None or df.empty:
            return
        # 创建图表
        fig = plt.Figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)
        # 定义反射率区间
        regions = [
            '<0.5', '0.5-0.55', '0.55-0.6', '0.6-0.65', '0.65-0.7', '0.7-0.75', '0.75-0.8', '0.8-0.85',
            '0.85-0.9', '0.9-0.95', '0.95-1.0', '1.0-1.05', '1.05-1.1', '1.1-1.15', '1.15-1.2', '1.2-1.25',
            '1.25-1.3', '1.3-1.35', '1.35-1.4', '1.4-1.45', '1.45-1.5', '1.5-1.55', '1.55-1.6', '1.6-1.65',
            '1.65-1.7', '1.7-1.75', '1.75-1.8', '1.8-1.85', '1.85-1.9', '1.9-1.95', '1.95-2.0', '2.0-2.05',
            '2.05-2.1', '2.1-2.15', '2.15-2.2', '2.2-2.25', '2.25-2.3', '2.3-2.35', '2.35-2.4', '2.4-2.45',
            '2.45-2.5', '＞2.5'
        ]
        col_names = [region.replace('-', '_').replace('<', 'less_than_').replace('>', 'greater_than_') 
                    for region in regions]
        # 计算各区间占比（先不乘100）
        total = sum(solution)
        raw_props = []
        for col in col_names:
            prop = np.dot(solution, df[col]) / total if total != 0 else 0
            raw_props.append(prop)
        sum_props = sum(raw_props)
        if sum_props > 0:
            proportions = [p / sum_props * 100 for p in raw_props]
        else:
            proportions = [0 for _ in raw_props]
        # 检查是否有异常值
        if any(p > 100 or p < 0 for p in proportions):
            print('警告：反射率分布区间出现异常占比！', proportions)
        # 绘制柱状图
        x_pos = np.arange(len(regions))
        bars = ax.bar(x_pos, proportions, align='center', alpha=0.7)
        # 设置柱状图颜色
        for bar, prop in zip(bars, proportions):
            bar.set_color('#3498DB' if prop > 5 else '#BDC3C7')
        # 设置图表样式
        ax.set_title('镜质组反射率分布', fontsize=12, pad=15)
        ax.set_xlabel('反射率区间 (%)', fontsize=9)
        ax.set_ylabel('占比 (%)', fontsize=9)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(regions, rotation=45, ha='right', fontsize=7)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        # 调整布局
        fig.tight_layout()
        # 创建画布
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def calculate_confidence(self, features):
        volatility_effect = np.exp(-max(0, features['volatile']-25)/10)
        return 0.7 * volatility_effect + 0.3 * self.historical_accuracy

class MultiPlanDialog:
    def __init__(self, master, default_constraints, df, optimizer):
        self.top = tk.Toplevel(master)
        self.top.title("多方案参数设置")
        self.result = None
        self.df = df
        self.optimizer = optimizer
        self.default_constraints = default_constraints
        # 简化：只支持输入一个方案
        frame = ttk.Frame(self.top)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        ttk.Label(frame, text="配煤总量:").grid(row=0, column=0, sticky=tk.W)
        self.quantity_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.quantity_var, width=12).grid(row=0, column=1, sticky=tk.W)
        # 质量约束
        self.cons_vars = {}
        row = 1
        for key in ['ash', 'sulfur', 'volatile', 'G_value', 'Y_value', 'CSR']:
            ttk.Label(frame, text=key+":").grid(row=row, column=0, sticky=tk.W)
            vmin = tk.StringVar(value=str(default_constraints[key][0]))
            vmax = tk.StringVar(value=str(default_constraints[key][1]))
            ttk.Entry(frame, textvariable=vmin, width=8).grid(row=row, column=1, sticky=tk.W)
            ttk.Entry(frame, textvariable=vmax, width=8).grid(row=row, column=2, sticky=tk.W)
            self.cons_vars[key] = (vmin, vmax)
            row += 1
        ttk.Label(frame, text="反射率标准差:").grid(row=row, column=0, sticky=tk.W)
        reflect_var = tk.StringVar(value=str(default_constraints['reflect_std_max']))
        ttk.Entry(frame, textvariable=reflect_var, width=8).grid(row=row, column=1, sticky=tk.W)
        self.cons_vars['reflect_std_max'] = reflect_var
        row += 1
        # 多煤种占比约束
        ttk.Label(frame, text="多煤种独立占比约束(格式:煤名,最小%,最大%):").grid(row=row, column=0, sticky=tk.W)
        self.ind_perc_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.ind_perc_var, width=30).grid(row=row, column=1, columnspan=2, sticky=tk.W)
        row += 1
        ttk.Button(frame, text="确定", command=self._on_ok).grid(row=row, column=0, columnspan=3, pady=10)
    def _on_ok(self):
        try:
            quantity = float(self.quantity_var.get())
            cons = {}
            for key in ['ash', 'sulfur', 'volatile', 'G_value', 'Y_value', 'CSR']:
                vmin = float(self.cons_vars[key][0].get())
                vmax = float(self.cons_vars[key][1].get())
                cons[key] = (vmin, vmax)
            cons['reflect_std_max'] = float(self.cons_vars['reflect_std_max'].get())
            # 解析多煤种占比约束
            ind_perc = []
            txt = self.ind_perc_var.get().strip()
            if txt:
                parts = txt.split(',')
                if len(parts) == 3:
                    name, minp, maxp = parts
                    idxs = [i for i, n in enumerate(self.df['coal_name']) if n == name.strip()]
                    if idxs:
                        ind_perc.append({
                            'coals_list': [name.strip()],
                            'indices_in_df': idxs,
                            'min_perc': float(minp)/100.0,
                            'max_perc': float(maxp)/100.0
                        })
            plan = {'quantity': quantity, 'constraints': cons, 'individual_perc_constraints': ind_perc}
            self.result = [plan]
            self.top.destroy()
        except Exception as e:
            messagebox.showerror("输入错误", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = CoalBlendOptimizer(root)
    root.mainloop()