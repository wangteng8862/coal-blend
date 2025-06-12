import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime

# 引入核心算法模块（已剥离）
import optimizer_core as opt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class CoalBlendOptimizer:
    def __init__(self, master):
        self.master = master
        self.df = None
        self.constraints = {
            'G_value': (80, 88),
            'CSR': (60, 68),
            'Y_value': (15, 25),
            'ash': (8, 10),
            'sulfur': (0.5, 1.0),
            'volatile': (18, 28),
            'reflect_std_max': 0.3,
            'volatile_diff_max': 5
        }
        self.num_individual_perc_constraints = 12
        self.parsed_individual_perc_constraints = []
        self.integer_constraint = tk.BooleanVar(value=False)
        self.rounding_precision = tk.IntVar(value=0)
        self.current_page = 0
        self.constraints_per_page = 6

        # 初始化 UI 组件数据结构
        self.ind_perc_coals_strs = [tk.StringVar() for _ in range(self.num_individual_perc_constraints)]
        self.ind_perc_min_strs = [tk.StringVar() for _ in range(self.num_individual_perc_constraints)]
        self.ind_perc_max_strs = [tk.StringVar() for _ in range(self.num_individual_perc_constraints)]

        self._setup_style()
        self.build_ui()
        self._bind_shortcuts()
        self._bind_events()
        self.status_bar = tk.Label(self.master, text=" 就绪 ", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.file_status = None
    def _setup_style(self):
        style = ttk.Style()
        style.theme_use('clam')

        style.configure('.', background='#F8F9FA', foreground='#2C3E50', font=('Microsoft YaHei', 10))
        style.map('TButton',
                  foreground=[('active', '#FFFFFF'), ('disabled', '#A0A0A0')],
                  background=[('active', '#2C81BA'), ('!active', '#3498DB')])
        style.configure('Accent.TButton', background='#27AE60', foreground='white',
                        font=('Microsoft YaHei', 10, 'bold'))
        style.configure('StatusBar.TLabel', background='#F8F9FA', foreground='#7F8C8D',
                        font=('Microsoft YaHei', 9))

    def build_ui(self):
        self.master.title("智能配煤系统")
        self.master.geometry("1200x800")

        self._create_menu()
        self._create_toolbar()

        main_container = ttk.Frame(self.master)
        main_container.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)
        main_container.columnconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=2)
        main_container.rowconfigure(0, weight=1)

        left_panel = ttk.Frame(main_container)
        left_panel.grid(row=0, column=0, padx=8, pady=8, sticky='nsew')

        right_panel = ttk.Frame(main_container)
        right_panel.grid(row=0, column=1, padx=8, pady=8, sticky='nsew')

        self._build_left_panel(left_panel)
        self._build_right_panel(right_panel)

        self.status_bar = ttk.Label(self.master, text=" 就绪 ", style='StatusBar.TLabel')
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _create_menu(self):
        menubar = tk.Menu(self.master)
        self.master.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="选择煤种文件", command=self.load_coal_file, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.master.quit, accelerator="Alt+F4")

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="关于", command=self.show_about)

    def _create_toolbar(self):
        toolbar = ttk.Frame(self.master)
        toolbar.pack(fill=tk.X, padx=5, pady=2)

        ttk.Button(toolbar, text="选择煤种文件", command=self.load_coal_file, style='Accent.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        ttk.Button(toolbar, text="开始计算", command=self.run_optimization, style='Accent.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="重置参数", command=self.reset_parameters, style='Accent.TButton').pack(side=tk.LEFT, padx=2)

        self.file_status = ttk.Label(toolbar, text=" 等待文件选择 ", style='StatusBar.TLabel')
        self.file_status.pack(side=tk.RIGHT, padx=5)
        
    def run_optimization(self):
        if not self._validate_inputs():
            return
        try:
            self.status_bar.config(text=" 正在执行分组优化... ")
            self.master.update()

            self._update_constraints()
            quantity = float(self.quantity_entry.get())
            max_diff = self.constraints.get('volatile_diff_max', None)
            if max_diff is None:
                messagebox.showerror("参数错误", "请设置最大挥发分差值")
                return

            best_solution = None
            best_cost = float('inf')
            best_main_coal = None
            best_df = None
            best_metrics = None

            reflect_regions = opt.parse_reflect_regions()

            for idx, (_, main_coal) in enumerate(self.df.iterrows()):
                main_volatile = main_coal['volatile']
                valid_indices = self.df.index[abs(self.df['volatile'] - main_volatile) <= max_diff].tolist()
                if not valid_indices:
                    continue
                sub_df = self.df.iloc[valid_indices].reset_index(drop=True)

                res = opt.find_best_volatile_solution(
                    df=sub_df,
                    quantity=quantity,
                    constraints=self.constraints,
                    parsed_individual_constraints=self.parsed_individual_perc_constraints,
                    max_diff=max_diff,
                    reflect_regions=reflect_regions,
                    main_coal_name=main_coal['coal_name'],
                    main_volatile=main_volatile
                )
                if res:
                    solution, metrics, cost, result_df = res
                    if cost < best_cost:
                        best_cost = cost
                        best_solution = solution
                        best_main_coal = main_coal['coal_name']
                        best_df = result_df
                        best_metrics = metrics

            if best_solution is not None:
                self._update_result_panels(best_solution, best_df)
                self.status_bar.config(text=f" 计算完成，主煤：{best_main_coal}")
            else:
                messagebox.showerror("优化失败", "未找到可行解，请检查约束设置与煤种数据")

        except Exception as e:
            messagebox.showerror("错误", f"优化失败：{str(e)}")
        finally:
            self.master.update()
    def load_coal_file(self):
        try:
            filepath = filedialog.askopenfilename(filetypes=[("CSV文件", "*.csv"), ("Excel文件", "*.xlsx")])
            if not filepath:
                return

            self.file_status.config(text=" 正在加载文件... ")
            self.status_bar.config(text=" 正在读取文件... ")

            if filepath.endswith('.csv'):
                self.df = pd.read_csv(filepath, encoding='utf-8')
            else:
                self.df = pd.read_excel(filepath)

            if self.df is not None and not self.df.empty:
                self.df.fillna(0, inplace=True)
                self.file_status.config(text=f" 已加载：{filepath.split('/')[-1]} ")
                self.status_bar.config(text=" 文件加载成功 ")
            else:
                messagebox.showerror("文件错误", "加载的文件数据为空")
        except Exception as e:
            self.file_status.config(text=" 文件加载失败 ")
            self.status_bar.config(text=f" 错误：{str(e)} ")
            messagebox.showerror("文件错误", f"文件加载失败：{str(e)}")

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
            for key, entry in self.entries.items():
                if key == 'reflect_std_max':
                    self.constraints[key] = float(entry.get())
                elif key == 'volatile_diff_max':
                    self.constraints[key] = float(entry.get())
                else:
                    min_val = float(entry[0].get())
                    max_val = float(entry[1].get())
                    self.constraints[key] = (min_val, max_val)
        except Exception as e:
            messagebox.showerror("参数错误", f"约束条件错误：{str(e)}")
            raise

        parsed = []
        for i in range(self.num_individual_perc_constraints):
            names = self.ind_perc_coals_strs[i].get().strip()
            minp = self.ind_perc_min_strs[i].get().strip()
            maxp = self.ind_perc_max_strs[i].get().strip()
            if not names:
                continue
            coal_list = [c.strip() for c in names.split(',') if c.strip()]
            indices = []
            for name in coal_list:
                if name in self.df['coal_name'].values:
                    idx = self.df.index[self.df['coal_name'] == name].tolist()[0]
                    indices.append(idx)
            if indices:
                try:
                    parsed.append({
                        'coals_list': coal_list,
                        'indices_in_df': indices,
                        'min_perc': float(minp) / 100.0 if minp else 0.0,
                        'max_perc': float(maxp) / 100.0 if maxp else 1.0
                    })
                except:
                    continue
        self.parsed_individual_perc_constraints = parsed

    def reset_parameters(self):
        self.quantity_entry.delete(0, tk.END)
        for key in self.entries:
            if key == 'reflect_std_max' or key == 'volatile_diff_max':
                self.entries[key].delete(0, tk.END)
                self.entries[key].insert(0, str(self.constraints[key]))
            else:
                self.entries[key][0].delete(0, tk.END)
                self.entries[key][1].delete(0, tk.END)
                self.entries[key][0].insert(0, str(self.constraints[key][0]))
                self.entries[key][1].insert(0, str(self.constraints[key][1]))

        for i in range(self.num_individual_perc_constraints):
            self.ind_perc_coals_strs[i].set("")
            self.ind_perc_min_strs[i].set("")
            self.ind_perc_max_strs[i].set("")
        self.parsed_individual_perc_constraints = []
        self.current_page = 0
        self.status_bar.config(text=" 参数已重置 ")
    def show_about(self):
        about_text = "智能配煤系统\n版本：v1.0\n作者：Dragen Team"
        messagebox.showinfo("关于", about_text)

    def _bind_shortcuts(self):
        self.master.bind('<Control-o>', lambda e: self.load_coal_file())
        self.master.bind('<Return>', lambda e: self.run_optimization())

    def _bind_events(self):
        self.master.bind_all("<Button-1>", lambda e: self._handle_focus(e))

    def _handle_focus(self, event):
        try:
            if hasattr(event.widget, 'focus_set'):
                event.widget.focus_set()
        except:
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = CoalBlendOptimizer(root)
    root.mainloop()
