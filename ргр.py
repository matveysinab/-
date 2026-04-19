import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import random
import threading
from datetime import datetime

class GeneticAlgorithmGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Генетический алгоритм оптимизации")
        self.root.geometry("1500x850")
        
        # Настройка цветовой темы
        self.setup_theme()
        
        # Параметры алгоритма
        self.population_size = tk.IntVar(value=100)
        self.generations = tk.IntVar(value=50)
        self.mutation_rate = tk.DoubleVar(value=0.1)
        self.crossover_rate = tk.DoubleVar(value=0.8)
        self.elite_size = tk.IntVar(value=5)
        
        # Параметры задачи
        self.x_min = tk.DoubleVar(value=-5)
        self.x_max = tk.DoubleVar(value=5)
        self.y_min = tk.DoubleVar(value=-5)
        self.y_max = tk.DoubleVar(value=5)
        
        # Выбор функции
        self.function_var = tk.StringVar(value="Rastrigin")
        self.custom_function = tk.StringVar(value="x**2 + y**2")
        self.func_type = tk.StringVar(value="preset")
        
        # Параметры поиска
        self.search_type = tk.StringVar(value="min")  # min или max
        
        # Переменные для хранения данных
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.is_running = False
        self.optimization_thread = None
        
        self.setup_ui()
        self.show_history()
        self.show_hints()
        self.show_algorithm_description()
        self.show_applications()
    def setup_theme(self):
        """Настройка цветовой темы"""
        self.bg_color = "#1e1e2e"
        self.fg_color = "#cdd6f4"
        self.accent_color = "#89b4fa"
        self.success_color = "#a6e3a1"
        self.warning_color = "#f9e2af"
        self.error_color = "#f38ba8"
        self.surface_color = "#313244"
        self.info_color = "#74c7ec"
        
        self.root.configure(bg=self.bg_color)
        
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure("TFrame", background=self.bg_color)
        style.configure("TLabel", background=self.bg_color, foreground=self.fg_color, font=('Segoe UI', 10))
        style.configure("TLabelframe", background=self.bg_color, foreground=self.fg_color, font=('Segoe UI', 10, 'bold'))
        style.configure("TLabelframe.Label", background=self.bg_color, foreground=self.accent_color)
        
        style.configure("TButton", 
                       background=self.surface_color,
                       foreground=self.fg_color,
                       borderwidth=0,
                       font=('Segoe UI', 10))
        style.map("TButton",
                 background=[('active', self.accent_color)],
                 foreground=[('active', self.bg_color)])
        
        style.configure("Accent.TButton",
                       background=self.accent_color,
                       foreground=self.bg_color,
                       font=('Segoe UI', 10, 'bold'))
        
        style.configure("TRadiobutton", background=self.bg_color, foreground=self.fg_color)
        style.configure("TEntry", fieldbackground=self.surface_color, foreground=self.fg_color)
        style.configure("TScale", background=self.bg_color, troughcolor=self.surface_color)
        style.configure("TSpinbox", fieldbackground=self.surface_color, foreground=self.fg_color)
        style.configure("TCombobox", fieldbackground=self.surface_color, foreground=self.fg_color)
        
    def setup_ui(self):
        """Настройка интерфейса"""
        # Основной контейнер с вкладками
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Вкладка 1: Оптимизация
        self.optimization_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.optimization_tab, text="🧬 Оптимизация")
        
        # Вкладка 2: История метода
        self.history_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.history_tab, text="📖 История метода")
        
        # Вкладка 3: Подсказки
        self.hints_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.hints_tab, text="💡 Подсказки")
        
        # Настройка вкладки оптимизации
        self.setup_optimization_tab()
        # После существующих вкладок добавьте:

        # Вкладка 4: Описание алгоритма
        self.algorithm_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.algorithm_tab, text="📖 Описание алгоритма")

        # Вкладка 5: Сферы применения
        self.applications_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.applications_tab, text="🎯 Сферы применения")
        
    def setup_optimization_tab(self):
        """Настройка вкладки оптимизации"""
        # Создаем три панели
        left_panel = ttk.Frame(self.optimization_tab, width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        center_panel = ttk.Frame(self.optimization_tab)
        center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        right_panel = ttk.Frame(self.optimization_tab, width=350)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)
        
        # === ЛЕВАЯ ПАНЕЛЬ - Параметры ===
        # Секция функции
        func_frame = ttk.LabelFrame(left_panel, text="📝 Функция для оптимизации", padding=10)
        func_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Переключатель типа функции
        func_type_frame = ttk.Frame(func_frame)
        func_type_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Radiobutton(func_type_frame, text="Готовая функция", 
                       variable=self.func_type, value="preset",
                       command=self.toggle_function_input).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(func_type_frame, text="Своя функция", 
                       variable=self.func_type, value="custom",
                       command=self.toggle_function_input).pack(side=tk.LEFT)
        
        # Готовые функции
        self.preset_frame = ttk.Frame(func_frame)
        self.preset_frame.pack(fill=tk.X)
        
        ttk.Label(self.preset_frame, text="Выберите функцию:").pack(anchor=tk.W, pady=(0, 5))
        self.function_combo = ttk.Combobox(self.preset_frame, textvariable=self.function_var, 
                                       values=["Rastrigin", "Ackley", "Sphere", "Rosenbrock", 
                                              "Beale", "Himmelblau", "Matyas"],
                                       state="readonly")
        self.function_combo.pack(fill=tk.X)
        self.function_combo.bind('<<ComboboxSelected>>', lambda e: self.update_surface())
        
        # Пользовательская функция
        self.custom_frame = ttk.Frame(func_frame)
        
        ttk.Label(self.custom_frame, text="Введите f(x, y):").pack(anchor=tk.W, pady=(0, 5))
        self.custom_entry = ttk.Entry(self.custom_frame, textvariable=self.custom_function, width=35)
        self.custom_entry.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(self.custom_frame, text="🧪 Проверить", 
                  command=self.test_custom_function).pack(fill=tk.X)
        
        self.toggle_function_input()
        
        # Секция поиска (МИНИМУМ/МАКСИМУМ)
        search_frame = ttk.LabelFrame(left_panel, text="🎯 Направление поиска", padding=10)
        search_frame.pack(fill=tk.X, pady=(0, 10))
        
        search_type_frame = ttk.Frame(search_frame)
        search_type_frame.pack()
        
        ttk.Radiobutton(search_type_frame, text="🔽 Поиск минимума", 
                       variable=self.search_type, value="min").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(search_type_frame, text="🔼 Поиск максимума", 
                       variable=self.search_type, value="max").pack(side=tk.LEFT, padx=10)
        
        # Секция диапазонов
        range_frame = ttk.LabelFrame(left_panel, text="🎯 Диапазоны поиска", padding=10)
        range_frame.pack(fill=tk.X, pady=(0, 10))
        
        x_frame = ttk.Frame(range_frame)
        x_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(x_frame, text="x:").pack(side=tk.LEFT)
        ttk.Entry(x_frame, textvariable=self.x_min, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(x_frame, text="→").pack(side=tk.LEFT)
        ttk.Entry(x_frame, textvariable=self.x_max, width=8).pack(side=tk.LEFT, padx=5)
        
        y_frame = ttk.Frame(range_frame)
        y_frame.pack(fill=tk.X)
        ttk.Label(y_frame, text="y:").pack(side=tk.LEFT)
        ttk.Entry(y_frame, textvariable=self.y_min, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(y_frame, text="→").pack(side=tk.LEFT)
        ttk.Entry(y_frame, textvariable=self.y_max, width=8).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(range_frame, text="🔄 Обновить поверхность", 
                  command=self.update_surface).pack(fill=tk.X, pady=(10, 0))
        
        # Секция параметров алгоритма
        algo_frame = ttk.LabelFrame(left_panel, text="⚙️ Параметры алгоритма", padding=10)
        algo_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(algo_frame, text="Размер популяции:").pack(anchor=tk.W)
        ttk.Spinbox(algo_frame, from_=10, to=500, textvariable=self.population_size, width=15).pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(algo_frame, text="Количество поколений:").pack(anchor=tk.W)
        ttk.Spinbox(algo_frame, from_=10, to=500, textvariable=self.generations, width=15).pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(algo_frame, text="Вероятность мутации:").pack(anchor=tk.W)
        ttk.Scale(algo_frame, from_=0.0, to=0.5, variable=self.mutation_rate, orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(algo_frame, textvariable=self.mutation_rate).pack(anchor=tk.CENTER, pady=(0, 10))
        
        ttk.Label(algo_frame, text="Вероятность скрещивания:").pack(anchor=tk.W)
        ttk.Scale(algo_frame, from_=0.0, to=1.0, variable=self.crossover_rate, orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(algo_frame, textvariable=self.crossover_rate).pack(anchor=tk.CENTER, pady=(0, 10))
        
        ttk.Label(algo_frame, text="Размер элиты:").pack(anchor=tk.W)
        ttk.Spinbox(algo_frame, from_=1, to=20, textvariable=self.elite_size, width=15).pack(fill=tk.X)
        
        # Кнопки управления
        control_frame = ttk.Frame(left_panel)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.start_button = ttk.Button(control_frame, text="▶ ЗАПУСТИТЬ ОПТИМИЗАЦИЮ", 
                                       command=self.start_optimization, style="Accent.TButton")
        self.start_button.pack(fill=tk.X, pady=5)
        
        self.stop_button = ttk.Button(control_frame, text="■ ОСТАНОВИТЬ", 
                                      command=self.stop_optimization, state='disabled')
        self.stop_button.pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="🗑 ОЧИСТИТЬ ВСЁ", 
                  command=self.clear_all).pack(fill=tk.X, pady=5)
        
        # === ЦЕНТРАЛЬНАЯ ПАНЕЛЬ - Графики ===
        plt.style.use('dark_background')
        self.fig = Figure(figsize=(8, 6), dpi=100, facecolor=self.bg_color)
        self.ax1 = self.fig.add_subplot(211, facecolor=self.surface_color)
        self.ax2 = self.fig.add_subplot(212, facecolor=self.surface_color)
        
        self.canvas = FigureCanvasTkAgg(self.fig, center_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # === ПРАВАЯ ПАНЕЛЬ - Промежуточные шаги ===
        steps_frame = ttk.LabelFrame(right_panel, text="📋 ПРОМЕЖУТОЧНЫЕ ШАГИ", padding=10)
        steps_frame.pack(fill=tk.BOTH, expand=True)
        
        self.steps_text = scrolledtext.ScrolledText(steps_frame, height=20,
                                                     bg=self.surface_color,
                                                     fg=self.fg_color,
                                                     font=('Consolas', 9),
                                                     wrap=tk.WORD)
        self.steps_text.pack(fill=tk.BOTH, expand=True)
        
        # Настройка тегов
        self.steps_text.tag_config("success", foreground=self.success_color)
        self.steps_text.tag_config("warning", foreground=self.warning_color)
        self.steps_text.tag_config("error", foreground=self.error_color)
        self.steps_text.tag_config("info", foreground=self.info_color)
        self.steps_text.tag_config("bold", font=('Consolas', 9, 'bold'))
        
        # Начальная отрисовка
        self.update_surface()
    
    def toggle_function_input(self):
        """Переключение между готовой и пользовательской функцией"""
        if self.func_type.get() == "preset":
            self.preset_frame.pack(fill=tk.X)
            self.custom_frame.pack_forget()
        else:
            self.preset_frame.pack_forget()
            self.custom_frame.pack(fill=tk.X)
    
    def test_custom_function(self):
        """Тестирование пользовательской функции"""
        try:
            test_x = 1.0
            test_y = 1.0
            result = self.evaluate_custom_function(test_x, test_y)
            self.add_step(f"✓ Функция корректна! f(1,1) = {result:.6f}", "success")
        except Exception as e:
            self.add_step(f"✗ Ошибка: {e}", "error")
    
    def evaluate_custom_function(self, x, y):
        """Вычисление пользовательской функции"""
        try:
            namespace = {
                'x': x, 'y': y,
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
                'abs': np.abs, 'pi': np.pi, 'e': np.e
            }
            expr = self.custom_function.get()
            result = eval(expr, {"__builtins__": {}}, namespace)
            return float(result)
        except Exception as e:
            raise ValueError(f"Ошибка: {e}")
    
    def objective_function(self, x, y):
        """Целевая функция с учетом минимума/максимума"""
        if self.func_type.get() == "custom":
            try:
                value = self.evaluate_custom_function(x, y)
            except:
                return float('inf') if self.search_type.get() == "min" else float('-inf')
        else:
            func = self.function_var.get()
            if func == "Rastrigin":
                value = (x**2 - 10*np.cos(2*np.pi*x)) + (y**2 - 10*np.cos(2*np.pi*y)) + 20
            elif func == "Ackley":
                value = -20*np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2))) - \
                       np.exp(0.5*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))) + np.e + 20
            elif func == "Sphere":
                value = x**2 + y**2
            elif func == "Rosenbrock":
                value = (1-x)**2 + 100*(y - x**2)**2
            elif func == "Beale":
                value = (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
            elif func == "Himmelblau":
                value = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
            elif func == "Matyas":
                value = 0.26*(x**2 + y**2) - 0.48*x*y
            else:
                value = x**2 + y**2
        
        # Преобразование для поиска максимума
        if self.search_type.get() == "max":
            return -value
        return value
    
    def get_actual_value(self, fitness):
        """Получение реального значения функции (для отображения)"""
        if self.search_type.get() == "max":
            return -fitness
        return fitness
    
    def add_step(self, message, tag=None):
        """Добавление шага в лог промежуточных результатов"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}\n"
        self.steps_text.insert(tk.END, formatted)
        if tag:
            line_start = self.steps_text.index("end-2l linestart")
            line_end = self.steps_text.index("end-2l lineend")
            self.steps_text.tag_add(tag, line_start, line_end)
        self.steps_text.see(tk.END)
    
    def update_surface(self):
        """Обновление поверхности функции"""
        self.ax1.clear()
        
        try:
            x = np.linspace(self.x_min.get(), self.x_max.get(), 100)
            y = np.linspace(self.y_min.get(), self.y_max.get(), 100)
            X, Y = np.meshgrid(x, y)
            
            Z = np.zeros_like(X)
            for i in range(len(x)):
                for j in range(len(y)):
                    val = self.objective_function(X[i, j], Y[i, j])
                    Z[i, j] = self.get_actual_value(val)
            
            contour = self.ax1.contourf(X, Y, Z, levels=50, cmap='plasma', alpha=0.8)
            self.fig.colorbar(contour, ax=self.ax1, label='Значение функции')
            
            search_text = "Минимум" if self.search_type.get() == "min" else "Максимум"
            func_name = self.function_var.get() if self.func_type.get() == "preset" else "Пользовательская"
            self.ax1.set_title(f'Поверхность функции: {func_name} (поиск {search_text})', color=self.fg_color)
            self.ax1.set_xlabel('x', color=self.fg_color)
            self.ax1.set_ylabel('y', color=self.fg_color)
            self.ax1.tick_params(colors=self.fg_color)
            self.ax1.grid(True, alpha=0.2)
            
            self.canvas.draw()
        except Exception as e:
            self.add_step(f"Ошибка при построении: {e}", "error")
    
    def initialize_population(self):
        """Создание начальной популяции"""
        population = []
        for i in range(self.population_size.get()):
            individual = {
                'x': random.uniform(self.x_min.get(), self.x_max.get()),
                'y': random.uniform(self.y_min.get(), self.y_max.get()),
                'id': i
            }
            individual['fitness'] = self.objective_function(individual['x'], individual['y'])
            population.append(individual)
        return population
    
    def selection(self, population):
        """Турнирный отбор"""
        tournament_size = 3
        parents = []
        for _ in range(len(population)):
            tournament = random.sample(population, tournament_size)
            winner = min(tournament, key=lambda x: x['fitness'])
            parents.append(winner.copy())
        return parents
    
    def crossover(self, parent1, parent2):
        """Скрещивание"""
        if random.random() < self.crossover_rate.get():
            alpha = random.random()
            child1 = {
                'x': alpha * parent1['x'] + (1 - alpha) * parent2['x'],
                'y': alpha * parent1['y'] + (1 - alpha) * parent2['y']
            }
            child2 = {
                'x': (1 - alpha) * parent1['x'] + alpha * parent2['x'],
                'y': (1 - alpha) * parent1['y'] + alpha * parent2['y']
            }
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def mutate(self, individual):
        """Мутация"""
        if random.random() < self.mutation_rate.get():
            individual['x'] += random.gauss(0, 0.5)
            individual['y'] += random.gauss(0, 0.5)
            
            individual['x'] = max(self.x_min.get(), min(self.x_max.get(), individual['x']))
            individual['y'] = max(self.y_min.get(), min(self.y_max.get(), individual['y']))
        return individual
    
    def evolve_population(self, population):
        """Эволюция одного поколения"""
        population.sort(key=lambda x: x['fitness'])
        elites = population[:self.elite_size.get()]
        parents = self.selection(population)
        
        new_population = []
        new_population.extend(elites)
        
        while len(new_population) < self.population_size.get():
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = self.crossover(parent1, parent2)
            
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            child1['fitness'] = self.objective_function(child1['x'], child1['y'])
            child2['fitness'] = self.objective_function(child2['x'], child2['y'])
            
            new_population.append(child1)
            if len(new_population) < self.population_size.get():
                new_population.append(child2)
        
        return new_population
    
    def update_visualization(self, generation, population, best, avg):
        """Обновление графиков"""
        # Обновление поверхности
        self.ax1.clear()
        x = np.linspace(self.x_min.get(), self.x_max.get(), 100)
        y = np.linspace(self.y_min.get(), self.y_max.get(), 100)
        X, Y = np.meshgrid(x, y)
        
        Z = np.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                val = self.objective_function(X[i, j], Y[i, j])
                Z[i, j] = self.get_actual_value(val)
        
        self.ax1.contourf(X, Y, Z, levels=50, cmap='plasma', alpha=0.7)
        actual_value = self.get_actual_value(best['fitness'])
        self.ax1.scatter(best['x'], best['y'], color='red', s=200, marker='*', 
                        edgecolors='white', linewidth=2, zorder=5)
        
        search_text = "Минимум" if self.search_type.get() == "min" else "Максимум"
        self.ax1.set_title(f'{search_text}: x={best["x"]:.4f}, y={best["y"]:.4f}, f={actual_value:.6f}', 
                          color=self.fg_color)
        self.ax1.set_xlabel('x', color=self.fg_color)
        self.ax1.set_ylabel('y', color=self.fg_color)
        self.ax1.tick_params(colors=self.fg_color)
        
        # Обновление графика сходимости
        self.ax2.clear()
        generations = list(range(generation + 1))
        
        best_values = [self.get_actual_value(v) for v in self.best_fitness_history]
        avg_values = [self.get_actual_value(v) for v in self.avg_fitness_history]
        
        self.ax2.plot(generations, best_values, '#89b4fa', linewidth=2.5, label='Лучшее значение')
        self.ax2.plot(generations, avg_values, '#f9e2af', linewidth=2, linestyle='--', label='Среднее значение')
        self.ax2.set_xlabel('Поколение', color=self.fg_color)
        self.ax2.set_ylabel('Значение функции', color=self.fg_color)
        self.ax2.set_title('Сходимость алгоритма', color=self.fg_color)
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.2)
        self.ax2.tick_params(colors=self.fg_color)
        
        self.canvas.draw()
    
    def start_optimization(self):
        """Запуск оптимизации"""
        # Проверяем, не запущен ли уже алгоритм
        if self.is_running:
            messagebox.showwarning("Предупреждение", "Оптимизация уже запущена!")
            return
        
        self.is_running = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        
        # Очистка промежуточных шагов
        self.steps_text.delete(1.0, tk.END)
        
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.ax2.clear()
        
        # Запуск в отдельном потоке
        self.optimization_thread = threading.Thread(target=self.optimization_loop)
        self.optimization_thread.daemon = True
        self.optimization_thread.start()
    
    def stop_optimization(self):
        """Остановка оптимизации"""
        if self.is_running:
            self.is_running = False
            self.add_step("⏸ Оптимизация остановлена пользователем", "warning")
    
    def optimization_loop(self):
        """Основной цикл оптимизации"""
        self.add_step("="*60, "info")
        self.add_step("🧬 ЗАПУСК ГЕНЕТИЧЕСКОГО АЛГОРИТМА", "bold")
        
        search_text = "МИНИМУМА" if self.search_type.get() == "min" else "МАКСИМУМА"
        self.add_step(f"🎯 Направление поиска: {search_text}", "info")
        
        func_name = self.function_var.get() if self.func_type.get() == "preset" else "Пользовательская функция"
        self.add_step(f"📊 Функция: {func_name}", "info")
        
        if self.func_type.get() == "custom":
            self.add_step(f"📝 Выражение: {self.custom_function.get()}", "info")
        
        self.add_step(f"👥 Размер популяции: {self.population_size.get()}", "info")
        self.add_step(f"🔄 Количество поколений: {self.generations.get()}", "info")
        self.add_step(f"🧬 Вероятность мутации: {self.mutation_rate.get():.2f}", "info")
        self.add_step(f"💑 Вероятность скрещивания: {self.crossover_rate.get():.2f}", "info")
        self.add_step(f"⭐ Размер элиты: {self.elite_size.get()}", "info")
        self.add_step("="*60, "info")
        
        # Инициализация
        population = self.initialize_population()
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
        self.add_step("📌 Начальная популяция создана", "success")
        
        for generation in range(self.generations.get()):
            if not self.is_running:
                break
            
            # Эволюция
            population = self.evolve_population(population)
            
            # Сбор статистики
            if self.search_type.get() == "min":
                best_individual = min(population, key=lambda x: x['fitness'])
            else:
                best_individual = max(population, key=lambda x: x['fitness'])
            
            avg_fitness = np.mean([ind['fitness'] for ind in population])
            
            self.best_fitness_history.append(best_individual['fitness'])
            self.avg_fitness_history.append(avg_fitness)
            
            actual_value = self.get_actual_value(best_individual['fitness'])
            
            # Добавление промежуточного шага
            if generation % 5 == 0 or generation == self.generations.get() - 1:
                progress = (generation + 1) / self.generations.get() * 100
                self.add_step(
                    f"Поколение {generation+1:3d} [{progress:5.1f}%] | "
                    f"Лучшее = {actual_value:10.6f} | "
                    f"x={best_individual['x']:7.4f}, y={best_individual['y']:7.4f}"
                )
            
            # Обновление графиков
            self.root.after(0, self.update_visualization, generation, population, 
                           best_individual, avg_fitness)
            self.root.update()
            self.root.after(20)
        
        # Всегда вызываем завершение
        self.root.after(0, self.optimization_finished)
    
    def optimization_finished(self):
        """Завершение оптимизации"""
        self.is_running = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
        if self.best_fitness_history:
            if self.search_type.get() == "min":
                best_value = min(self.best_fitness_history)
                best_index = self.best_fitness_history.index(best_value)
            else:
                best_value = max(self.best_fitness_history)
                best_index = self.best_fitness_history.index(best_value)
            
            actual_value = self.get_actual_value(best_value)
            
            self.add_step("="*60, "info")
            self.add_step("✓ ОПТИМИЗАЦИЯ УСПЕШНО ЗАВЕРШЕНА!", "success")
            
            search_text = "минимум" if self.search_type.get() == "min" else "максимум"
            self.add_step(f"🏆 Найден {search_text}: {actual_value:.8f}", "success")
            self.add_step(f"📍 Найден в поколении: {best_index + 1}", "info")
            self.add_step("="*60, "info")
        
        # Очищаем поток
        self.optimization_thread = None
    
    def clear_all(self):
        """Очистка"""
        if self.is_running:
            messagebox.showwarning("Предупреждение", "Сначала остановите оптимизацию!")
            return
        
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.update_surface()
        self.ax2.clear()
        self.canvas.draw()
        self.steps_text.delete(1.0, tk.END)
        self.add_step("🧹 Все данные очищены", "info")
    
    def show_history(self):
        """Показать историю метода на вкладке"""
        history_text = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ИСТОРИЯ ГЕНЕТИЧЕСКОГО АЛГОРИТМА                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

📅 1950-е годы:
   • Алекс Фрейзер (Alex Fraser) - первые идеи использования эволюции для 
     оптимизации
   • Идея: моделировать естественный отбор для решения задач

📅 1960-е годы:
   • Джон Холланд (John Holland) в Мичиганском университете разрабатывает 
     теоретические основы генетических алгоритмов
   • Вводит понятия: хромосома, мутация, кроссинговер, селекция
   • Публикует работу "Adaptation in Natural and Artificial Systems" (1975)

📅 1970-е годы:
   • Кеннет Де Йонг (Kenneth De Jong) проводит первые эксперименты
   • Вводит понятие "функция приспособленности" (fitness function)
   • Разрабатывает методы сравнения ГА

📅 1980-е годы:
   • Джон Голдберг (David Goldberg) популяризирует ГА в инженерии
   • Публикует книгу "Genetic Algorithms in Search, Optimization, and Machine Learning"
   • Первые успешные промышленные применения

📅 1990-е годы:
   • Развитие гибридных алгоритмов (меметические алгоритмы)
   • Применение в машинном обучении
   • Создание NSGA-II для многокритериальной оптимизации

📅 2000-е годы - настоящее время:
   • Интеграция с глубоким обучением
   • Применение в биоинформатике, робототехнике, финансах
   • Развитие распределенных и параллельных ГА
"""
        
        history_frame = ttk.Frame(self.history_tab)
        history_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        history_text_widget = scrolledtext.ScrolledText(history_frame, 
                                                        bg=self.surface_color,
                                                        fg=self.fg_color,
                                                        font=('Consolas', 10),
                                                        wrap=tk.WORD)
        history_text_widget.pack(fill=tk.BOTH, expand=True)
        history_text_widget.insert(tk.END, history_text)
        history_text_widget.config(state=tk.DISABLED)
    
    def show_hints(self):
        """Показать подсказки на вкладке"""
        hints_text = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                          ПОДСКАЗКИ ПО НАСТРОЙКЕ                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 ВЫБОР ФУНКЦИИ:
────────────────────────────────────────────────────────────────────────────────
   • Rastrigin   - много локальных минимумов (сложная)
   • Ackley      - имеет один глобальный минимум в центре
   • Sphere      - простая парабола (легкая)
   • Rosenbrock  - овражная функция (сложная)

🎯 ПОИСК МИНИМУМА vs МАКСИМУМА:
────────────────────────────────────────────────────────────────────────────────
   • Поиск минимума  - алгоритм ищет наименьшее значение функции
   • Поиск максимума - алгоритм ищет наибольшее значение функции

⚙️ ПАРАМЕТРЫ АЛГОРИТМА:
────────────────────────────────────────────────────────────────────────────────

1. РАЗМЕР ПОПУЛЯЦИИ (10-500):
   • Маленький (< 50)    - быстро, но может застрять в локальном оптимуме
   • Средний (50-200)    - оптимальный баланс
   • Большой (> 200)     - медленно, но лучше исследует пространство

2. ВЕРОЯТНОСТЬ МУТАЦИИ (0.0-0.5):
   • 0.01-0.05  - небольшие изменения, быстрая сходимость
   • 0.05-0.1   - стандартный диапазон
   • 0.1-0.3    - активный поиск

3. ВЕРОЯТНОСТЬ СКРЕЩИВАНИЯ (0.0-1.0):
   • 0.5-0.8    - классический диапазон
   • >0.9       - активный обмен информацией

4. РАЗМЕР ЭЛИТЫ (1-20):
   • 1-3        - классический вариант
   • 5-10       - более стабильная сходимость
"""
        
        hints_frame = ttk.Frame(self.hints_tab)
        hints_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        hints_text_widget = scrolledtext.ScrolledText(hints_frame, 
                                                      bg=self.surface_color,
                                                      fg=self.fg_color,
                                                      font=('Consolas', 10),
                                                      wrap=tk.WORD)
        hints_text_widget.pack(fill=tk.BOTH, expand=True)
        hints_text_widget.insert(tk.END, hints_text)
        hints_text_widget.config(state=tk.DISABLED)
    def show_algorithm_description(self):
        """Показать описание алгоритма на вкладке"""
        description_text = """
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                      АЛГОРИТМ РАБОТЫ ГЕНЕТИЧЕСКОГО АЛГОРИТМА                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  ОСНОВНАЯ ИДЕЯ                                                                                                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  Генетический алгоритм (ГА) — это метод оптимизации, основанный на механизмах естественного отбора и наследственности.│
│  Алгоритм имитирует эволюционный процесс: популяция решений эволюционирует от поколения к поколению, становясь лучше.  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  ШАГ 1: ИНИЦИАЛИЗАЦИЯ ПОПУЛЯЦИИ                                                                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  • Создаётся начальная популяция из N случайных особей (потенциальных решений)                                        │
│  • Каждая особь представляется вектором координат (x, y) в пространстве поиска                                        │
│  • Для каждой особи вычисляется значение функции приспособленности (fitness)                                          │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  ШАГ 2: ОЦЕНКА ПРИСПОСОБЛЕННОСТИ (FITNESS)                                                                            │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  • Функция приспособленности — это целевая функция, которую мы оптимизируем                                           │
│  • При поиске МИНИМУМА: лучшие особи имеют наименьшее значение fitness                                               │
│  • При поиске МАКСИМУМА: алгоритм преобразует задачу (fitness = -f(x)) и ищет минимум                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  ШАГ 3: СЕЛЕКЦИЯ (ОТБОР РОДИТЕЛЕЙ) - ТУРНИРНЫЙ ОТБОР                                                                  │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  1. Случайно выбирается группа особей (турнир) размером k (обычно k = 3)                                              │
│  2. Из этой группы выбирается лучшая особь (с наименьшим значением fitness)                                           │
│  3. Выбранная особь становится родителем                                                                              │
│  4. Процесс повторяется, пока не набрано нужное количество родителей                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  ШАГ 4: КРОССИНГОВЕР (СКРЕЩИВАНИЕ) - АРИФМЕТИЧЕСКИЙ                                                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  Для двух родителей P1 и P2 создаются два потомка C1 и C2:                                                            │
│      C1 = α · P1 + (1-α) · P2    C2 = (1-α) · P1 + α · P2, где α — случайное число в интервале [0, 1]                │
│  Кроссинговер выполняется с вероятностью P_crossover (обычно 0.6-0.9)                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  ШАГ 5: МУТАЦИЯ - ГАУССОВСКАЯ                                                                                         │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│      x' = x + N(0, σ)    y' = y + N(0, σ)                                                                             │
│  где N(0, σ) — случайное значение из нормального распределения с нулевым средним и σ = 0.5                           │
│  • Мутация выполняется с вероятностью P_mutation (обычно 0.01-0.1)                                                    │
│  • Поддерживает генетическое разнообразие и помогает избежать преждевременной сходимости                              │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  ШАГ 6: ЭЛИТИЗМ                                                                                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  • Лучшие E особей текущего поколения автоматически переходят в следующее поколение без изменений                     │
│  • Гарантирует, что лучшее найденное решение не будет потеряно                                                        │
│  • Размер элиты (E) обычно составляет 1-5 особей                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  ВИЗУАЛЬНОЕ ПРЕДСТАВЛЕНИЕ РАБОТЫ АЛГОРИТМА                                                                            │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  Верхний график: Контурный график поверхности целевой функции, красная звезда ★ — текущее лучшее решение             │
│  Нижний график: Синяя линия — лучшее значение, жёлтая пунктирная — среднее значение в популяции                       │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
        
        # Создаем фрейм для текста
        desc_frame = ttk.Frame(self.algorithm_tab)
        desc_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Текстовое поле с прокруткой
        desc_text = scrolledtext.ScrolledText(desc_frame, 
                                              bg=self.surface_color,
                                              fg=self.fg_color,
                                              font=('Consolas', 10),
                                              wrap=tk.WORD)
        desc_text.pack(fill=tk.BOTH, expand=True)
        desc_text.insert(tk.END, description_text)
        desc_text.config(state=tk.DISABLED)

    def show_applications(self):
        """Показать сферы применения на вкладке"""
        applications_text = """
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                         СФЕРЫ ПРИМЕНЕНИЯ ГЕНЕТИЧЕСКИХ АЛГОРИТМОВ                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  1. ИНЖЕНЕРНОЕ ПРОЕКТИРОВАНИЕ                                                                                         │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  • Оптимизация формы аэродинамических поверхностей (крылья самолётов, лопасти турбин)                                 │
│  • Проектирование электронных схем и антенн                                                                           │
│  • Настройка параметров автоматических систем управления                                                              │
│  • Оптимизация топологии конструкций (максимальная прочность при минимальном весе)                                    │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  2. МАШИННОЕ ОБУЧЕНИЕ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ                                                                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  • Настройка гиперпараметров нейронных сетей (количество слоёв, нейронов, скорость обучения)                         │
│  • Отбор признаков (feature selection) — выбор наиболее информативных признаков                                       │
│  • Нейроэволюция — эволюционное создание архитектур нейросетей                                                        │
│  • Обучение с подкреплением (reinforcement learning)                                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  3. РОБОТОТЕХНИКА                                                                                                     │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  • Обучение роботов методом проб и ошибок                                                                             │
│  • Оптимизация траекторий движения манипуляторов                                                                      │
│  • Настройка параметров сенсорных систем                                                                              │
│  • Планирование маршрутов мобильных роботов                                                                           │
│  • Эволюция управляющих программ для автономных роботов                                                               │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  4. ЭКОНОМИКА И ФИНАНСЫ                                                                                                │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  • Оптимизация инвестиционных портфелей (максимизация доходности при заданном риске)                                  │
│  • Калибровка экономических моделей                                                                                   │
│  • Прогнозирование временных рядов (курсы валют, цены акций)                                                           │
│  • Оптимизация логистических цепочек                                                                                  │
│  • Задачи маршрутизации транспорта (задача коммивояжёра)                                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  5. БИОИНФОРМАТИКА И МЕДИЦИНА                                                                                         │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  • Выравнивание последовательностей ДНК                                                                               │
│  • Предсказание структуры белков (фолдинг)                                                                            │
│  • Анализ генетических данных и поиск мутаций                                                                         │
│  • Оптимизация дозировки лекарств                                                                                     │
│  • Планирование лучевой терапии в онкологии                                                                           │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  6. ХИМИЯ И МАТЕРИАЛОВЕДЕНИЕ                                                                                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  • Оптимизация состава сплавов для достижения заданных свойств                                                        │
│  • Поиск оптимальных условий химических реакций (температура, давление, катализаторы)                                 │
│  • Проектирование новых материалов с заданными свойствами                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  7. ЭНЕРГЕТИКА                                                                                                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  • Оптимизация режимов работы электростанций                                                                          │
│  • Оптимизация распределения нагрузки в энергосетях                                                                   │
│  • Размещение возобновляемых источников энергии (ветряки, солнечные панели)                                           │
│  • Прогнозирование энергопотребления                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  8. ТРАНСПОРТ И ЛОГИСТИКА                                                                                             │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  • Задача коммивояжёра (поиск кратчайшего маршрута)                                                                   │
│  • Маршрутизация транспортных средств (VRP)                                                                           │
│  • Оптимизация расписаний (авиакомпании, поезда, автобусы)                                                            │
│  • Планирование доставки (последняя миля)                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  ПРЕИМУЩЕСТВА ГЕНЕТИЧЕСКИХ АЛГОРИТМОВ                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  ✓ Не требуют вычисления производных (работают с любыми функциями)                                                    │
│  ✓ Способны находить глобальный оптимум в многоэкстремальных задачах                                                  │
│  ✓ Устойчивы к шумам и разрывам в целевой функции                                                                     │
│  ✓ Легко распараллеливаются (каждая особь вычисляется независимо)                                                     │
│  ✓ Просты в реализации и понимании                                                                                    │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
        
        # Создаем фрейм для текста
        app_frame = ttk.Frame(self.applications_tab)
        app_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Текстовое поле с прокруткой
        app_text = scrolledtext.ScrolledText(app_frame, 
                                             bg=self.surface_color,
                                             fg=self.fg_color,
                                             font=('Consolas', 10),
                                             wrap=tk.WORD)
        app_text.pack(fill=tk.BOTH, expand=True)
        app_text.insert(tk.END, applications_text)
        app_text.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = GeneticAlgorithmGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()    
