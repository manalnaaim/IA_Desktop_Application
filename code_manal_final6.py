import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, silhouette_score
import pandas as pd
import seaborn as sns
from PIL import Image, ImageTk
import os
import pickle
from sklearn.metrics.pairwise import euclidean_distances

class manalapp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced IA Platform")
        self.root.geometry("1400x900")
        
        # Style configuration
        self.style = {
            'bg': '#F0FFF0',  # Changed to match the example
            'fg': '#2E8B57',
            'button_bg': '#77DD77',  # Changed to match the example
            'button_active': '#3CB371',  # Changed to match the example
            'highlight': '#2E8B57',
            'text_bg': '#FFFFFF',
            'graph_bg': '#F5F5F5',
            'font': ('Arial', 10),
            'title_font': ('Arial', 14, 'bold')
        }
        
        # Data and model storage
        self.current_dataset = None
        self.models = {
            'lin_reg': None,
            'random_forest': None,
            'kmeans': None,
            'arima': None
        }
        
        # Default parameters
        self.model_params = {
            'lin_reg': {'test_size': 0.2, 'random_state': 42},
            'random_forest': {
                'n_estimators': 100, 
                'max_depth': None, 
                'is_classification': True,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt'
            },
            'kmeans': {
                'n_clusters': 3, 
                'random_state': 42,
                'min_samples': 5,
                'max_iter': 300
            },
            'arima': {'order': (2,1,1), 'steps': 10}
        }
        
        # Initialize app with home page like in the example
        self.create_home_page()
    
    def create_home_page(self):
        """Create the welcoming home page matching the example style"""
        self.home_frame = tk.Frame(self.root, bg=self.style['bg'])
        self.home_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title - styled like the example
        tk.Label(self.home_frame, 
                text="Interface des Algorithmes d'IA", 
                font=("Arial", 20, "bold"),
                bg=self.style['bg'],
                fg=self.style['highlight']).pack(pady=40)
        
        # Description (optional, not in example but could be added)
        
        # Buttons - styled exactly like the example
        enter_btn = tk.Button(self.home_frame, 
                            text="Entrer", 
                            command=self.show_main_interface,
                            bg=self.style['button_bg'],
                            fg="white",
                            activebackground=self.style['button_active'],
                            font=('Arial', 14),
                            padx=20,
                            pady=10,
                            relief=tk.RAISED,
                            borderwidth=3)
        enter_btn.pack(pady=20)
        
        quit_btn = tk.Button(self.home_frame, 
                           text="Quitter", 
                           command=self.root.destroy,
                           bg="#C1E1C1",
                           fg="#2E8B57",
                           activebackground="#A7C7A7",
                           font=('Arial', 12),
                           padx=15,
                           pady=5,
                           relief=tk.RAISED,
                           borderwidth=2)
        quit_btn.pack(pady=10)
    
    def show_main_interface(self):
        """Show the main interface after home page"""
        self.home_frame.destroy()
        self.create_main_interface()
        self.generate_sample_data()
    
    def create_main_interface(self):
        """Create the main application interface"""
        # Main layout
        self.main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=self.style['bg'])
        self.main_pane.pack(fill=tk.BOTH, expand=True)
        
        # Control panel on left
        self.control_frame = tk.Frame(self.main_pane, bg=self.style['bg'], width=400)
        self.main_pane.add(self.control_frame)
        
        # Results panel on right
        self.result_frame = tk.Frame(self.main_pane, bg=self.style['bg'])
        self.main_pane.add(self.result_frame)
        
        # Create components
        self.create_control_panel()
        self.create_result_panel()
    
    def create_control_panel(self):
        """Create the left control panel"""
        control_notebook = ttk.Notebook(self.control_frame)
        control_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Data tab
        data_tab = tk.Frame(control_notebook, bg=self.style['bg'])
        self.create_data_controls(data_tab)
        control_notebook.add(data_tab, text="Data")
        
        # Models tab
        model_tab = tk.Frame(control_notebook, bg=self.style['bg'])
        self.create_model_controls(model_tab)
        control_notebook.add(model_tab, text="Models")
        
        # Settings tab
        settings_tab = tk.Frame(control_notebook, bg=self.style['bg'])
        self.create_settings_controls(settings_tab)
        control_notebook.add(settings_tab, text="Settings")
        
        # Manual Input tab
        manual_tab = tk.Frame(control_notebook, bg=self.style['bg'])
        self.create_manual_controls(manual_tab)
        control_notebook.add(manual_tab, text="Manual Input")
    
    def create_manual_controls(self, parent):
        """Create controls for manual input"""
        tk.Label(parent, 
                text="Manual Data Input", 
                font=self.style['title_font'],
                bg=self.style['bg'],
                fg=self.style['highlight']).pack(pady=10, anchor='w')
        
        # X Value input
        x_frame = tk.Frame(parent, bg=self.style['bg'])
        x_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(x_frame, 
                text="X Value:", 
                bg=self.style['bg'],
                fg=self.style['fg']).pack(side=tk.LEFT)
        
        self.x_input = tk.Entry(x_frame)
        self.x_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.x_input.insert(0, "50")
        
        # Y Value input
        y_frame = tk.Frame(parent, bg=self.style['bg'])
        y_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(y_frame, 
                text="Y Value:", 
                bg=self.style['bg'],
                fg=self.style['fg']).pack(side=tk.LEFT)
        
        self.y_input = tk.Entry(y_frame)
        self.y_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.y_input.insert(0, "20")
        
        # Range sliders
        range_frame = tk.LabelFrame(parent, 
                                  text="Data Range Settings",
                                  bg=self.style['bg'],
                                  fg=self.style['highlight'],
                                  font=self.style['font'])
        range_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Min X slider
        tk.Label(range_frame, 
                text="Min X:", 
                bg=self.style['bg'],
                fg=self.style['fg']).pack(anchor='w')
        
        self.min_x_slider = tk.Scale(range_frame, 
                                   from_=0, to=100, 
                                   orient=tk.HORIZONTAL,
                                   bg=self.style['bg'],
                                   fg=self.style['fg'],
                                   highlightbackground=self.style['bg'])
        self.min_x_slider.set(0)
        self.min_x_slider.pack(fill=tk.X, padx=5, pady=2)
        
        # Max X slider
        tk.Label(range_frame, 
                text="Max X:", 
                bg=self.style['bg'],
                fg=self.style['fg']).pack(anchor='w')
        
        self.max_x_slider = tk.Scale(range_frame, 
                                   from_=0, to=100, 
                                   orient=tk.HORIZONTAL,
                                   bg=self.style['bg'],
                                   fg=self.style['fg'],
                                   highlightbackground=self.style['bg'])
        self.max_x_slider.set(100)
        self.max_x_slider.pack(fill=tk.X, padx=5, pady=2)
        
        # Min Y slider
        tk.Label(range_frame, 
                text="Min Y:", 
                bg=self.style['bg'],
                fg=self.style['fg']).pack(anchor='w')
        
        self.min_y_slider = tk.Scale(range_frame, 
                                   from_=0, to=100, 
                                   orient=tk.HORIZONTAL,
                                   bg=self.style['bg'],
                                   fg=self.style['fg'],
                                   highlightbackground=self.style['bg'])
        self.min_y_slider.set(0)
        self.min_y_slider.pack(fill=tk.X, padx=5, pady=2)
        
        # Max Y slider
        tk.Label(range_frame, 
                text="Max Y:", 
                bg=self.style['bg'],
                fg=self.style['fg']).pack(anchor='w')
        
        self.max_y_slider = tk.Scale(range_frame, 
                                   from_=0, to=100, 
                                   orient=tk.HORIZONTAL,
                                   bg=self.style['bg'],
                                   fg=self.style['fg'],
                                   highlightbackground=self.style['bg'])
        self.max_y_slider.set(100)
        self.max_y_slider.pack(fill=tk.X, padx=5, pady=2)
        
        # Update button - styled like the example
        update_btn = tk.Button(parent,
                             text="Update All Models",
                             command=self.update_all_models,
                             bg=self.style['button_bg'],
                             fg="white",
                             font=('Arial', 10),
                             padx=15,
                             pady=5,
                             relief=tk.RAISED,
                             borderwidth=2)
        update_btn.pack(fill=tk.X, padx=5, pady=10)
    
    def update_all_models(self):
        """Update all models with manual input values"""
        try:
            # Get manual input values
            x_val = float(self.x_input.get())
            y_val = float(self.y_input.get())
            
            # Get range values
            min_x = self.min_x_slider.get()
            max_x = self.max_x_slider.get()
            min_y = self.min_y_slider.get()
            max_y = self.max_y_slider.get()
            
            # Update data with new point (for visualization)
            if self.current_dataset is not None:
                new_data = pd.DataFrame({
                    'X': [x_val],
                    'Y': [y_val],
                    'Power': [x_val],
                    'Weight': [y_val],
                    'Aerodynamics': [(x_val + y_val)/2],
                    'Consumption': [0.3*x_val + 0.1*y_val - 0.2*((x_val + y_val)/2)]
                })
                
                # Filter data based on range
                filtered_data = self.current_dataset[
                    (self.current_dataset['Power'] >= min_x) & 
                    (self.current_dataset['Power'] <= max_x) &
                    (self.current_dataset['Weight'] >= min_y) & 
                    (self.current_dataset['Weight'] <= max_y)
                ]
                
                # Combine with new point
                self.current_dataset = pd.concat([filtered_data, new_data], ignore_index=True)
                
                # Update variable menus
                self.update_variable_menus()
                
                # Run all models
                self.run_linear_regression(manual_x=x_val, manual_y=y_val)
                self.run_random_forest(manual_x=x_val, manual_y=y_val)
                self.run_kmeans(manual_x=x_val, manual_y=y_val)
                self.run_arima(manual_x=x_val)
                
                messagebox.showinfo("Success", "All models updated with new data point!")
            else:
                messagebox.showerror("Error", "No dataset available")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for X and Y values")
    
    def create_data_controls(self, parent):
        """Create data management controls"""
        tk.Label(parent, 
                text="Data Management", 
                font=self.style['title_font'],
                bg=self.style['bg'],
                fg=self.style['highlight']).pack(pady=10, anchor='w')
        
        # Load data button - styled like the example
        load_btn = tk.Button(parent,
                            text="Load CSV File",
                            command=self.load_dataset,
                            bg=self.style['button_bg'],
                            fg="white",
                            activebackground=self.style['button_active'],
                            font=('Arial', 10),
                            padx=15,
                            pady=5,
                            relief=tk.RAISED,
                            borderwidth=2)
        load_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Generate data button - styled like the example
        gen_btn = tk.Button(parent,
                           text="Generate Sample Data",
                           command=self.generate_sample_data,
                           bg="#77DD77",
                           fg="white",
                           activebackground="#3CB371",
                           font=('Arial', 10),
                           padx=15,
                           pady=5,
                           relief=tk.RAISED,
                           borderwidth=2)
        gen_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Save data button - styled like the example
        save_btn = tk.Button(parent,
                           text="Save Current Data",
                           command=self.save_current_data,
                           bg="#C1E1C1",
                           fg="#2E8B57",
                           activebackground="#A7C7A7",
                           font=('Arial', 10),
                           padx=15,
                           pady=5,
                           relief=tk.RAISED,
                           borderwidth=2)
        save_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Data info display
        self.data_info = tk.Label(parent,
                                 text="No data loaded",
                                 bg=self.style['bg'],
                                 fg=self.style['fg'],
                                 font=self.style['font'],
                                 wraplength=330,
                                 justify=tk.LEFT)
        self.data_info.pack(pady=10, padx=5, fill=tk.X)
        
        # Variable selection
        self.var_frame = tk.LabelFrame(parent, 
                                     text="Variable Selection",
                                     bg=self.style['bg'],
                                     fg=self.style['highlight'],
                                     font=self.style['font'])
        self.var_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.x_var = tk.StringVar()
        self.y_var = tk.StringVar()
        
        tk.Label(self.var_frame, 
                text="X Variable:", 
                bg=self.style['bg'],
                fg=self.style['fg']).pack(anchor='w')
        self.x_menu = ttk.Combobox(self.var_frame, textvariable=self.x_var, state='readonly')
        self.x_menu.pack(fill=tk.X, padx=5, pady=2)
        
        tk.Label(self.var_frame, 
                text="Y Variable:", 
                bg=self.style['bg'],
                fg=self.style['fg']).pack(anchor='w')
        self.y_menu = ttk.Combobox(self.var_frame, textvariable=self.y_var, state='readonly')
        self.y_menu.pack(fill=tk.X, padx=5, pady=2)
        
        # Data preprocessing options
        preprocess_frame = tk.LabelFrame(parent,
                                       text="Data Preprocessing",
                                       bg=self.style['bg'],
                                       fg=self.style['highlight'],
                                       font=self.style['font'])
        preprocess_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.normalize_var = tk.BooleanVar(value=False)
        tk.Checkbutton(preprocess_frame,
                      text="Normalize Data",
                      variable=self.normalize_var,
                      bg=self.style['bg'],
                      fg=self.style['fg']).pack(anchor='w')
    
    def create_model_controls(self, parent):
        """Create model configuration controls"""
        tk.Label(parent, 
                text="Model Configuration", 
                font=self.style['title_font'],
                bg=self.style['bg'],
                fg=self.style['highlight']).pack(pady=10, anchor='w')
        
        # Model notebook
        model_notebook = ttk.Notebook(parent)
        model_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Linear Regression
        reg_frame = tk.Frame(model_notebook, bg=self.style['bg'])
        self.create_regression_controls(reg_frame)
        model_notebook.add(reg_frame, text="Regression")
        
        # Random Forest
        rf_frame = tk.Frame(model_notebook, bg=self.style['bg'])
        self.create_randomforest_controls(rf_frame)
        model_notebook.add(rf_frame, text="Random Forest")
        
        # K-Means
        kmeans_frame = tk.Frame(model_notebook, bg=self.style['bg'])
        self.create_kmeans_controls(kmeans_frame)
        model_notebook.add(kmeans_frame, text="Clustering")
        
        # ARIMA
        arima_frame = tk.Frame(model_notebook, bg=self.style['bg'])
        self.create_arima_controls(arima_frame)
        model_notebook.add(arima_frame, text="Time Series")
    
    def create_regression_controls(self, parent):
        """Create linear regression controls"""
        tk.Label(parent, 
                text="Linear Regression", 
                font=("Arial", 12),
                bg=self.style['bg'],
                fg=self.style['highlight']).pack(pady=5, anchor='w')
        
        # Test size
        tk.Label(parent, 
                text="Test Size (%):", 
                bg=self.style['bg'],
                fg=self.style['fg']).pack(anchor='w')
        
        test_size = tk.Scale(parent, 
                           from_=10, to=50, 
                           orient=tk.HORIZONTAL,
                           bg=self.style['bg'],
                           fg=self.style['fg'],
                           highlightbackground=self.style['bg'])
        test_size.set(self.model_params['lin_reg']['test_size'] * 100)
        test_size.pack(fill=tk.X, padx=5, pady=2)
        self.model_params['lin_reg']['test_size_scale'] = test_size
        
        # Cross-validation
        cv_frame = tk.Frame(parent, bg=self.style['bg'])
        cv_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.cv_var = tk.BooleanVar(value=False)
        tk.Checkbutton(cv_frame,
                      text="Use Cross-Validation",
                      variable=self.cv_var,
                      bg=self.style['bg'],
                      fg=self.style['fg']).pack(side=tk.LEFT)
        
        self.cv_folds = tk.IntVar(value=5)
        tk.Spinbox(cv_frame, from_=2, to=10, textvariable=self.cv_folds, width=3).pack(side=tk.RIGHT)
        tk.Label(cv_frame, text="Folds:", bg=self.style['bg'], fg=self.style['fg']).pack(side=tk.RIGHT)
        
        # Algorithm comparison for CV
        self.cv_compare_var = tk.BooleanVar(value=False)
        tk.Checkbutton(parent,
                     text="Compare with Random Forest in CV",
                     variable=self.cv_compare_var,
                     bg=self.style['bg'],
                     fg=self.style['fg']).pack(anchor='w', pady=5)
        

        # Min X pour la régression
        tk.Label(parent, text="Min X pour la régression:", bg=self.style['bg'], fg=self.style['fg']).pack(anchor='w')
        min_x_reg = tk.Scale(parent, from_=0, to=100, orient=tk.HORIZONTAL, bg=self.style['bg'], fg=self.style['fg'], highlightbackground=self.style['bg'])
        min_x_reg.set(0)
        min_x_reg.pack(fill=tk.X, padx=5, pady=2)
        self.model_params['lin_reg']['min_x_reg'] = min_x_reg

        # Max X pour la régression
        tk.Label(parent, text="Max X pour la régression:", bg=self.style['bg'], fg=self.style['fg']).pack(anchor='w')
        max_x_reg = tk.Scale(parent, from_=0, to=100, orient=tk.HORIZONTAL, bg=self.style['bg'], fg=self.style['fg'], highlightbackground=self.style['bg'])
        max_x_reg.set(100)
        max_x_reg.pack(fill=tk.X, padx=5, pady=2)
        self.model_params['lin_reg']['max_x_reg'] = max_x_reg

        # Run button - styled like the example
        run_btn = tk.Button(parent,
                           text="Run Linear Regression",
                           command=self.run_linear_regression,
                           bg=self.style['button_bg'],
                           fg="white",
                           activebackground=self.style['button_active'],
                           font=('Arial', 10),
                           padx=15,
                           pady=5,
                           relief=tk.RAISED,
                           borderwidth=2)
        run_btn.pack(fill=tk.X, padx=5, pady=10)
        
        # Save model button - styled like the example
        save_model_btn = tk.Button(parent,
                                 text="Save Model",
                                 command=lambda: self.save_model('lin_reg'),
                                 bg="#77DD77",
                                 fg="white",
                                 activebackground="#3CB371",
                                 font=('Arial', 10),
                                 padx=15,
                                 pady=5,
                                 relief=tk.RAISED,
                                 borderwidth=2)
        save_model_btn.pack(fill=tk.X, padx=5, pady=5)
    
    def create_randomforest_controls(self, parent):
        """Create random forest controls"""
        tk.Label(parent, 
                text="Random Forest", 
                font=("Arial", 12),
                bg=self.style['bg'],
                fg=self.style['highlight']).pack(pady=5, anchor='w')
        
        # Model type
        tk.Label(parent, 
                text="Model Type:", 
                bg=self.style['bg'],
                fg=self.style['fg']).pack(anchor='w')
        
        self.rf_type = tk.StringVar(value="classification")
        tk.Radiobutton(parent, text="Classification", variable=self.rf_type, 
                      value="classification", bg=self.style['bg']).pack(anchor='w')
        tk.Radiobutton(parent, text="Regression", variable=self.rf_type, 
                      value="regression", bg=self.style['bg']).pack(anchor='w')
        
        # Number of trees
        tk.Label(parent, 
                text="Number of Trees:", 
                bg=self.style['bg'],
                fg=self.style['fg']).pack(anchor='w')
        
        n_estimators = tk.Scale(parent, 
                              from_=10, to=200, 
                              orient=tk.HORIZONTAL,
                              bg=self.style['bg'],
                              fg=self.style['fg'],
                              highlightbackground=self.style['bg'])
        n_estimators.set(self.model_params['random_forest']['n_estimators'])
        n_estimators.pack(fill=tk.X, padx=5, pady=2)
        self.model_params['random_forest']['n_estimators_scale'] = n_estimators
        
        # Max depth
        tk.Label(parent, 
                text="Max Depth (0=unlimited):", 
                bg=self.style['bg'],
                fg=self.style['fg']).pack(anchor='w')
        
        max_depth = tk.Scale(parent, 
                           from_=0, to=20, 
                           orient=tk.HORIZONTAL,
                           bg=self.style['bg'],
                           fg=self.style['fg'],
                           highlightbackground=self.style['bg'])
        max_depth.set(0)
        max_depth.pack(fill=tk.X, padx=5, pady=2)
        self.model_params['random_forest']['max_depth_scale'] = max_depth
        
        # Min samples split
        tk.Label(parent, 
                text="Min Samples Split:", 
                bg=self.style['bg'],
                fg=self.style['fg']).pack(anchor='w')
        
        min_samples_split = tk.Scale(parent, 
                                   from_=2, to=10, 
                                   orient=tk.HORIZONTAL,
                                   bg=self.style['bg'],
                                   fg=self.style['fg'],
                                   highlightbackground=self.style['bg'])
        min_samples_split.set(self.model_params['random_forest']['min_samples_split'])
        min_samples_split.pack(fill=tk.X, padx=5, pady=2)
        self.model_params['random_forest']['min_samples_split_scale'] = min_samples_split
        
        # Min samples leaf
        tk.Label(parent, 
                text="Min Samples Leaf:", 
                bg=self.style['bg'],
                fg=self.style['fg']).pack(anchor='w')
        
        min_samples_leaf = tk.Scale(parent, 
                                  from_=1, to=10, 
                                  orient=tk.HORIZONTAL,
                                  bg=self.style['bg'],
                                  fg=self.style['fg'],
                                  highlightbackground=self.style['bg'])
        min_samples_leaf.set(self.model_params['random_forest']['min_samples_leaf'])
        min_samples_leaf.pack(fill=tk.X, padx=5, pady=2)
        self.model_params['random_forest']['min_samples_leaf_scale'] = min_samples_leaf
        
        # Max features
        tk.Label(parent, 
                text="Max Features:", 
                bg=self.style['bg'],
                fg=self.style['fg']).pack(anchor='w')
        
        max_features = ttk.Combobox(parent, 
                                  values=['sqrt', 'log2', None, 'auto'],
                                  state='readonly')
        max_features.set(self.model_params['random_forest']['max_features'])
        max_features.pack(fill=tk.X, padx=5, pady=2)
        self.model_params['random_forest']['max_features_combo'] = max_features
        

        # Min X pour la régression
        tk.Label(parent, text="Min X pour la régression:", bg=self.style['bg'], fg=self.style['fg']).pack(anchor='w')
        min_x_reg = tk.Scale(parent, from_=0, to=100, orient=tk.HORIZONTAL, bg=self.style['bg'], fg=self.style['fg'], highlightbackground=self.style['bg'])
        min_x_reg.set(0)
        min_x_reg.pack(fill=tk.X, padx=5, pady=2)
        self.model_params['lin_reg']['min_x_reg'] = min_x_reg

        # Max X pour la régression
        tk.Label(parent, text="Max X pour la régression:", bg=self.style['bg'], fg=self.style['fg']).pack(anchor='w')
        max_x_reg = tk.Scale(parent, from_=0, to=100, orient=tk.HORIZONTAL, bg=self.style['bg'], fg=self.style['fg'], highlightbackground=self.style['bg'])
        max_x_reg.set(100)
        max_x_reg.pack(fill=tk.X, padx=5, pady=2)
        self.model_params['lin_reg']['max_x_reg'] = max_x_reg

        # Run button - styled like the example
        run_btn = tk.Button(parent,
                           text="Run Random Forest",
                           command=self.run_random_forest,
                           bg=self.style['button_bg'],
                           fg="white",
                           activebackground=self.style['button_active'],
                           font=('Arial', 10),
                           padx=15,
                           pady=5,
                           relief=tk.RAISED,
                           borderwidth=2)
        run_btn.pack(fill=tk.X, padx=5, pady=10)
        
        # Save model button - styled like the example
        save_model_btn = tk.Button(parent,
                                 text="Save Model",
                                 command=lambda: self.save_model('random_forest'),
                                 bg="#77DD77",
                                 fg="white",
                                 activebackground="#3CB371",
                                 font=('Arial', 10),
                                 padx=15,
                                 pady=5,
                                 relief=tk.RAISED,
                                 borderwidth=2)
        save_model_btn.pack(fill=tk.X, padx=5, pady=5)
    
    def create_kmeans_controls(self, parent):
        """Create K-Means controls"""
        tk.Label(parent, 
                text="K-Means Clustering", 
                font=("Arial", 12),
                bg=self.style['bg'],
                fg=self.style['highlight']).pack(pady=5, anchor='w')
        
        # Number of clusters
        tk.Label(parent, 
                text="Number of Clusters:", 
                bg=self.style['bg'],
                fg=self.style['fg']).pack(anchor='w')
        
        n_clusters = tk.Scale(parent, 
                            from_=2, to=10, 
                            orient=tk.HORIZONTAL,
                            bg=self.style['bg'],
                            fg=self.style['fg'],
                            highlightbackground=self.style['bg'])
        n_clusters.set(self.model_params['kmeans']['n_clusters'])
        n_clusters.pack(fill=tk.X, padx=5, pady=2)
        self.model_params['kmeans']['n_clusters_scale'] = n_clusters
        
        # Min samples per cluster
        tk.Label(parent, 
                text="Min Samples per Cluster:", 
                bg=self.style['bg'],
                fg=self.style['fg']).pack(anchor='w')
        
        min_samples = tk.Scale(parent, 
                             from_=1, to=20, 
                             orient=tk.HORIZONTAL,
                             bg=self.style['bg'],
                             fg=self.style['fg'],
                             highlightbackground=self.style['bg'])
        min_samples.set(self.model_params['kmeans']['min_samples'])
        min_samples.pack(fill=tk.X, padx=5, pady=2)
        self.model_params['kmeans']['min_samples_scale'] = min_samples
        
        # Max iterations
        tk.Label(parent, 
                text="Max Iterations:", 
                bg=self.style['bg'],
                fg=self.style['fg']).pack(anchor='w')
        
        max_iter = tk.Scale(parent, 
                          from_=100, to=500, 
                          orient=tk.HORIZONTAL,
                          bg=self.style['bg'],
                          fg=self.style['fg'],
                          highlightbackground=self.style['bg'])
        max_iter.set(self.model_params['kmeans']['max_iter'])
        max_iter.pack(fill=tk.X, padx=5, pady=2)
        self.model_params['kmeans']['max_iter_scale'] = max_iter
        

        # Min X pour la régression
        tk.Label(parent, text="Min X pour la régression:", bg=self.style['bg'], fg=self.style['fg']).pack(anchor='w')
        min_x_reg = tk.Scale(parent, from_=0, to=100, orient=tk.HORIZONTAL, bg=self.style['bg'], fg=self.style['fg'], highlightbackground=self.style['bg'])
        min_x_reg.set(0)
        min_x_reg.pack(fill=tk.X, padx=5, pady=2)
        self.model_params['lin_reg']['min_x_reg'] = min_x_reg

        # Max X pour la régression
        tk.Label(parent, text="Max X pour la régression:", bg=self.style['bg'], fg=self.style['fg']).pack(anchor='w')
        max_x_reg = tk.Scale(parent, from_=0, to=100, orient=tk.HORIZONTAL, bg=self.style['bg'], fg=self.style['fg'], highlightbackground=self.style['bg'])
        max_x_reg.set(100)
        max_x_reg.pack(fill=tk.X, padx=5, pady=2)
        self.model_params['lin_reg']['max_x_reg'] = max_x_reg

        # Run button - styled like the example
        run_btn = tk.Button(parent,
                           text="Run K-Means",
                           command=self.run_kmeans,
                           bg=self.style['button_bg'],
                           fg="white",
                           activebackground=self.style['button_active'],
                           font=('Arial', 10),
                           padx=15,
                           pady=5,
                           relief=tk.RAISED,
                           borderwidth=2)
        run_btn.pack(fill=tk.X, padx=5, pady=10)
        
        # Save model button - styled like the example
        save_model_btn = tk.Button(parent,
                                 text="Save Model",
                                 command=lambda: self.save_model('kmeans'),
                                 bg="#77DD77",
                                 fg="white",
                                 activebackground="#3CB371",
                                 font=('Arial', 10),
                                 padx=15,
                                 pady=5,
                                 relief=tk.RAISED,
                                 borderwidth=2)
        save_model_btn.pack(fill=tk.X, padx=5, pady=5)
    
    def create_arima_controls(self, parent):
        """Create ARIMA controls"""
        tk.Label(parent, 
                text="ARIMA Model", 
                font=("Arial", 12),
                bg=self.style['bg'],
                fg=self.style['highlight']).pack(pady=5, anchor='w')
        
        # ARIMA parameters
        params_frame = tk.Frame(parent, bg=self.style['bg'])
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(params_frame, 
                text="ARIMA Order (p,d,q):", 
                bg=self.style['bg'],
                fg=self.style['fg']).grid(row=0, column=0, sticky='w')
        
        p_var = tk.IntVar(value=self.model_params['arima']['order'][0])
        d_var = tk.IntVar(value=self.model_params['arima']['order'][1])
        q_var = tk.IntVar(value=self.model_params['arima']['order'][2])
        
        tk.Spinbox(params_frame, from_=0, to=5, textvariable=p_var, width=3).grid(row=0, column=1, padx=2)
        tk.Spinbox(params_frame, from_=0, to=2, textvariable=d_var, width=3).grid(row=0, column=2, padx=2)
        tk.Spinbox(params_frame, from_=0, to=5, textvariable=q_var, width=3).grid(row=0, column=3, padx=2)
        
        self.model_params['arima']['p_var'] = p_var
        self.model_params['arima']['d_var'] = d_var
        self.model_params['arima']['q_var'] = q_var
        
        # Forecast steps
        tk.Label(parent, 
                text="Forecast Steps:", 
                bg=self.style['bg'],
                fg=self.style['fg']).pack(anchor='w')
        
        steps = tk.Scale(parent, 
                       from_=5, to=50, 
                       orient=tk.HORIZONTAL,
                       bg=self.style['bg'],
                       fg=self.style['fg'],
                       highlightbackground=self.style['bg'])
        steps.set(self.model_params['arima']['steps'])
        steps.pack(fill=tk.X, padx=5, pady=2)
        self.model_params['arima']['steps_scale'] = steps
        

        # Min X pour la régression
        tk.Label(parent, text="Min X pour la régression:", bg=self.style['bg'], fg=self.style['fg']).pack(anchor='w')
        min_x_reg = tk.Scale(parent, from_=0, to=100, orient=tk.HORIZONTAL, bg=self.style['bg'], fg=self.style['fg'], highlightbackground=self.style['bg'])
        min_x_reg.set(0)
        min_x_reg.pack(fill=tk.X, padx=5, pady=2)
        self.model_params['lin_reg']['min_x_reg'] = min_x_reg

        # Max X pour la régression
        tk.Label(parent, text="Max X pour la régression:", bg=self.style['bg'], fg=self.style['fg']).pack(anchor='w')
        max_x_reg = tk.Scale(parent, from_=0, to=100, orient=tk.HORIZONTAL, bg=self.style['bg'], fg=self.style['fg'], highlightbackground=self.style['bg'])
        max_x_reg.set(100)
        max_x_reg.pack(fill=tk.X, padx=5, pady=2)
        self.model_params['lin_reg']['max_x_reg'] = max_x_reg

        # Run button - styled like the example
        run_btn = tk.Button(parent,
                           text="Run ARIMA",
                           command=self.run_arima,
                           bg=self.style['button_bg'],
                           fg="white",
                           activebackground=self.style['button_active'],
                           font=('Arial', 10),
                           padx=15,
                           pady=5,
                           relief=tk.RAISED,
                           borderwidth=2)
        run_btn.pack(fill=tk.X, padx=5, pady=10)
        
        # Save model button - styled like the example
        save_model_btn = tk.Button(parent,
                                 text="Save Model",
                                 command=lambda: self.save_model('arima'),
                                 bg="#77DD77",
                                 fg="white",
                                 activebackground="#3CB371",
                                 font=('Arial', 10),
                                 padx=15,
                                 pady=5,
                                 relief=tk.RAISED,
                                 borderwidth=2)
        save_model_btn.pack(fill=tk.X, padx=5, pady=5)
    
    def create_settings_controls(self, parent):
        """Create settings controls"""
        tk.Label(parent, 
                text="Application Settings", 
                font=self.style['title_font'],
                bg=self.style['bg'],
                fg=self.style['highlight']).pack(pady=10, anchor='w')
        
        # Random seed
        tk.Label(parent, 
                text="Random Seed:", 
                bg=self.style['bg'],
                fg=self.style['fg']).pack(anchor='w')
        
        random_seed = tk.Entry(parent)
        random_seed.insert(0, str(self.model_params['lin_reg']['random_state']))
        random_seed.pack(fill=tk.X, padx=5, pady=2)
        self.model_params['random_seed_entry'] = random_seed
        
        # Apply settings button - styled like the example
        apply_btn = tk.Button(parent,
                            text="Apply Settings",
                            command=self.apply_settings,
                            bg=self.style['button_bg'],
                            fg="white",
                            activebackground=self.style['button_active'],
                            font=('Arial', 10),
                            padx=15,
                            pady=5,
                            relief=tk.RAISED,
                            borderwidth=2)
        apply_btn.pack(fill=tk.X, padx=5, pady=10)
        
        # Load model button - styled like the example
        load_model_btn = tk.Button(parent,
                                 text="Load Model",
                                 command=self.load_model,
                                 bg="#77DD77",
                                 fg="white",
                                 activebackground="#3CB371",
                                 font=('Arial', 10),
                                 padx=15,
                                 pady=5,
                                 relief=tk.RAISED,
                                 borderwidth=2)
        load_model_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Home button - styled like the example
        home_btn = tk.Button(parent,
                           text="Retour à l'accueil",
                           command=self.return_to_home,
                           bg="#C1E1C1",
                           fg="#2E8B57",
                           activebackground="#A7C7A7",
                           font=('Arial', 10),
                           padx=10,
                           pady=3,
                           relief=tk.RAISED,
                           borderwidth=2)
        home_btn.pack(fill=tk.X, padx=5, pady=20)
    
    def create_result_panel(self):
        """Create the right results panel"""
        # Notebook for results
        self.result_notebook = ttk.Notebook(self.result_frame)
        self.result_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Results tab
        self.results_tab = tk.Frame(self.result_notebook, bg=self.style['bg'])
        self.create_results_tab(self.results_tab)
        self.result_notebook.add(self.results_tab, text="Results")
        
        # Visualization tab
        self.viz_tab = tk.Frame(self.result_notebook, bg=self.style['bg'])
        self.result_notebook.add(self.viz_tab, text="Visualization")
        
        # Data tab
        self.data_tab = tk.Frame(self.result_notebook, bg=self.style['bg'])
        self.create_data_tab(self.data_tab)
        self.result_notebook.add(self.data_tab, text="Data")
        
        # All Models tab
        self.all_models_tab = tk.Frame(self.result_notebook, bg=self.style['bg'])
        self.result_notebook.add(self.all_models_tab, text="All Models")
    
    def create_results_tab(self, parent):
        """Create results text output"""
        # Text output
        self.text_output = scrolledtext.ScrolledText(parent, 
                                                   wrap=tk.WORD,
                                                   font=self.style['font'],
                                                   bg=self.style['text_bg'],
                                                   fg=self.style['fg'])
        self.text_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Export button - styled like the example
        export_btn = tk.Button(parent,
                             text="Export Results",
                             command=self.export_results,
                             bg=self.style['button_bg'],
                             fg="white",
                             activebackground=self.style['button_active'],
                             font=('Arial', 10),
                             padx=15,
                             pady=5,
                             relief=tk.RAISED,
                             borderwidth=2)
        export_btn.pack(fill=tk.X, padx=5, pady=5)
    
    def create_data_tab(self, parent):
        """Create data display tab"""
        # Table for data display
        self.data_table = ttk.Treeview(parent)
        self.data_table.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.data_table.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.data_table.configure(yscrollcommand=scrollbar.set)
        
        # Export button - styled like the example
        export_btn = tk.Button(parent,
                             text="Export Data",
                             command=self.export_data,
                             bg=self.style['button_bg'],
                             fg="white",
                             activebackground=self.style['button_active'],
                             font=('Arial', 10),
                             padx=15,
                             pady=5,
                             relief=tk.RAISED,
                             borderwidth=2)
        export_btn.pack(fill=tk.X, padx=5, pady=5)
    
    # ======================
    # DATA MANAGEMENT METHODS
    # ======================
    
    def load_dataset(self):
        """Load dataset from CSV file"""
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            try:
                self.current_dataset = pd.read_csv(file_path)
                self.update_data_info()
                self.update_variable_menus()
                self.display_data_table()
                messagebox.showinfo("Success", "Data loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
    
    def save_current_data(self):
        """Save current dataset to CSV"""
        if self.current_dataset is None:
            messagebox.showerror("Error", "No data to save")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                               filetypes=[("CSV files", "*.csv"),
                                                          ("All files", "*.*")])
        if file_path:
            try:
                self.current_dataset.to_csv(file_path, index=False)
                messagebox.showinfo("Success", "Data saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save data:\n{str(e)}")
    
    def generate_sample_data(self):
        """Generate sample data for demonstration"""
        np.random.seed(int(self.model_params['random_seed_entry'].get()))
        
        # Regression data
        size = 200
        self.lin_reg_X = np.random.rand(size, 3) * 100
        self.lin_reg_y = (0.3 * self.lin_reg_X[:,0] + 
                        0.1 * self.lin_reg_X[:,1] - 
                        0.2 * self.lin_reg_X[:,2] + 
                        np.random.randn(size) * 5)
        
        # Create DataFrame for display
        self.current_dataset = pd.DataFrame({
            'X': self.lin_reg_X[:,0],
            'Y': self.lin_reg_X[:,1],
            'Power': self.lin_reg_X[:,0],
            'Weight': self.lin_reg_X[:,1],
            'Aerodynamics': self.lin_reg_X[:,2],
            'Consumption': self.lin_reg_y
        })
        
        self.update_data_info()
        self.update_variable_menus()
        self.display_data_table()
        messagebox.showinfo("Success", "Sample data generated successfully!")
    
    def update_data_info(self):
        """Update data information display"""
        if self.current_dataset is not None:
            info = f"Loaded Data:\n- Rows: {len(self.current_dataset)}\n- Columns: {len(self.current_dataset.columns)}\n"
            info += f"- Columns: {', '.join(self.current_dataset.columns)}"
            self.data_info.config(text=info)
    
    def update_variable_menus(self):
        """Update variable selection menus"""
        if self.current_dataset is not None:
            columns = list(self.current_dataset.columns)
            self.x_menu['values'] = columns
            self.y_menu['values'] = columns
            if len(columns) > 0:
                self.x_var.set('X')
                if len(columns) > 1:
                    self.y_var.set('Y')
    
    def display_data_table(self):
        """Display data in table view"""
        if self.current_dataset is not None:
            # Clear existing columns
            for col in self.data_table.get_children():
                self.data_table.delete(col)
            self.data_table["columns"] = list(self.current_dataset.columns)
            
            # Configure columns
            for col in self.current_dataset.columns:
                self.data_table.heading(col, text=col)
                self.data_table.column(col, width=100)
            
            # Add data
            for i, row in self.current_dataset.iterrows():
                self.data_table.insert("", "end", values=list(row))
    
    def apply_settings(self):
        """Apply global settings"""
        try:
            seed = int(self.model_params['random_seed_entry'].get())
            np.random.seed(seed)
            self.model_params['lin_reg']['random_state'] = seed
            self.model_params['random_forest']['random_state'] = seed
            self.model_params['kmeans']['random_state'] = seed
            messagebox.showinfo("Success", "Settings applied successfully!")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for random seed")
    
    # =================
    # MODEL RUN METHODS
    # =================
    
    def clear_output(self):
        """Clear text and visualization outputs"""
        self.text_output.delete(1.0, tk.END)
        self.clear_viz_tab()
    
    def clear_viz_tab(self):
        """Clear visualization tab"""
        for widget in self.viz_tab.winfo_children():
            widget.destroy()
    
    def run_linear_regression(self, manual_x=None, manual_y=None):
        """Run linear regression model"""
        self.clear_output()
        
        if self.current_dataset is None:
            messagebox.showerror("Error", "Please load or generate data first")
            return
        
        try:
            # Get parameters
            test_size = self.model_params['lin_reg']['test_size_scale'].get() / 100
            random_state = self.model_params['lin_reg']['random_state']
            use_cv = self.cv_var.get()
            cv_folds = self.cv_folds.get()
            compare_with_rf = self.cv_compare_var.get()
            
            # Select variables
            X = self.current_dataset[[self.x_var.get()]].values
            y = self.current_dataset[self.y_var.get()].values
            
            
            # Filtrage selon les bornes min/max X définies par l'utilisateur
            min_x = self.model_params['lin_reg']['min_x_reg'].get()
            max_x = self.model_params['lin_reg']['max_x_reg'].get()
            if min_x > max_x:
                messagebox.showerror("Erreur", "Min X ne peut pas être supérieur à Max X.")
                return
            mask = (X[:, 0] >= min_x) & (X[:, 0] <= max_x)
            X = X[mask]
            y = y[mask]
            if len(X) == 0:
                messagebox.showerror("Erreur", "Aucune donnée dans l'intervalle choisi pour X.")
                return

# Normalize if selected
            if self.normalize_var.get():
                X = (X - X.mean()) / X.std()
                y = (y - y.mean()) / y.std()
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state)
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = model.score(X_test, y_test)
            
            # Cross-validation if selected
            cv_results = None
            rf_cv_results = None
            
            if use_cv:
                kf = KFold(n_splits=cv_folds)
                cv_scores = []
                
                for train_idx, test_idx in kf.split(X):
                    X_train_cv, X_test_cv = X[train_idx], X[test_idx]
                    y_train_cv, y_test_cv = y[train_idx], y[test_idx]
                    
                    model_cv = LinearRegression()
                    model_cv.fit(X_train_cv, y_train_cv)
                    y_pred_cv = model_cv.predict(X_test_cv)
                    cv_scores.append(mean_squared_error(y_test_cv, y_pred_cv))
                
                cv_results = {
                    'mean_mse': np.mean(cv_scores),
                    'std_mse': np.std(cv_scores),
                    'scores': cv_scores
                }
                
                # Compare with Random Forest if selected
                if compare_with_rf:
                    rf_scores = []
                    for train_idx, test_idx in kf.split(X):
                        X_train_cv, X_test_cv = X[train_idx], X[test_idx]
                        y_train_cv, y_test_cv = y[train_idx], y[test_idx]
                        
                        rf_model = RandomForestRegressor(
                            n_estimators=self.model_params['random_forest']['n_estimators_scale'].get(),
                            max_depth=self.model_params['random_forest']['max_depth_scale'].get(),
                            random_state=random_state
                        )
                        rf_model.fit(X_train_cv, y_train_cv)
                        y_pred_cv = rf_model.predict(X_test_cv)
                        rf_scores.append(mean_squared_error(y_test_cv, y_pred_cv))
                    
                    rf_cv_results = {
                        'mean_mse': np.mean(rf_scores),
                        'std_mse': np.std(rf_scores),
                        'scores': rf_scores
                    }
            
            # Display results
            self.text_output.insert(tk.END, "=== LINEAR REGRESSION ===\n\n")
            self.text_output.insert(tk.END, f"X Variable: {self.x_var.get()}\n")
            self.text_output.insert(tk.END, f"Y Variable: {self.y_var.get()}\n\n")
            
            self.text_output.insert(tk.END, "=== MODEL PARAMETERS ===\n")
            self.text_output.insert(tk.END, f"Test Size: {test_size:.0%}\n")
            self.text_output.insert(tk.END, f"Random State: {random_state}\n")
            if use_cv:
                self.text_output.insert(tk.END, f"Cross-Validation Folds: {cv_folds}\n")
            
            self.text_output.insert(tk.END, "\n=== MODEL RESULTS ===\n")
            self.text_output.insert(tk.END, f"Coefficient: {model.coef_[0]:.4f}\n")
            self.text_output.insert(tk.END, f"Intercept: {model.intercept_:.4f}\n")
            self.text_output.insert(tk.END, f"MSE: {mse:.4f}\n")
            self.text_output.insert(tk.END, f"R²: {r2:.4f}\n")
            
            # Ajout des marges de variation
            self.text_output.insert(tk.END, "\n=== VARIATION MARGINS ===\n")
            self.text_output.insert(tk.END, "Input Variation:\n")
            self.text_output.insert(tk.END, f"- Standard deviation: {X.std():.4f}\n")
            self.text_output.insert(tk.END, f"- Range: [{X.min():.4f}, {X.max():.4f}]\n")
            
            self.text_output.insert(tk.END, "\nOutput Variation:\n")
            self.text_output.insert(tk.END, f"- Standard deviation: {y.std():.4f}\n")
            self.text_output.insert(tk.END, f"- Range: [{y.min():.4f}, {y.max():.4f}]\n")
            
            # Add data statistics
            self.text_output.insert(tk.END, "\n=== DATA STATISTICS ===\n")
            self.text_output.insert(tk.END, f"Mean of X: {X.mean():.4f}\n")
            self.text_output.insert(tk.END, f"Std of X: {X.std():.4f}\n")
            self.text_output.insert(tk.END, f"Mean of Y: {y.mean():.4f}\n")
            self.text_output.insert(tk.END, f"Std of Y: {y.std():.4f}\n")
            
            # Add variation analysis
            self.text_output.insert(tk.END, "\n=== VARIATION ANALYSIS ===\n")
            self.text_output.insert(tk.END, f"Covariance: {np.cov(X.flatten(), y)[0,1]:.4f}\n")
            self.text_output.insert(tk.END, f"Correlation: {np.corrcoef(X.flatten(), y)[0,1]:.4f}\n")
            
            if use_cv:
                self.text_output.insert(tk.END, "\n=== CROSS-VALIDATION RESULTS ===\n\n")
                self.text_output.insert(tk.END, f"Mean MSE ({cv_folds}-fold CV): {cv_results['mean_mse']:.4f}\n")
                self.text_output.insert(tk.END, f"Std MSE: {cv_results['std_mse']:.4f}\n")
                self.text_output.insert(tk.END, f"Individual fold scores: {[f'{x:.4f}' for x in cv_results['scores']]}\n")
                
                if compare_with_rf:
                    self.text_output.insert(tk.END, "\n=== RANDOM FOREST CV COMPARISON ===\n")
                    self.text_output.insert(tk.END, f"Mean MSE ({cv_folds}-fold CV): {rf_cv_results['mean_mse']:.4f}\n")
                    self.text_output.insert(tk.END, f"Std MSE: {rf_cv_results['std_mse']:.4f}\n")
                    self.text_output.insert(tk.END, f"Individual fold scores: {[f'{x:.4f}' for x in rf_cv_results['scores']]}\n")
                    self.text_output.insert(tk.END, f"Difference (Linear - RF): {cv_results['mean_mse'] - rf_cv_results['mean_mse']:.4f}\n")
            
            # Store model
            self.models['lin_reg'] = model
            
            # Visualization
            self.show_linear_regression_plot(X_test, y_test, y_pred, use_cv, cv_results, manual_x, manual_y, rf_cv_results)
            
            # Update all models visualization
            self.show_all_models_visualization(manual_x, manual_y)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error in linear regression:\n{str(e)}")
    
    def show_linear_regression_plot(self, X_test, y_test, y_pred, use_cv=False, cv_results=None, manual_x=None, manual_y=None, rf_cv_results=None):
        """Visualize linear regression results"""
        self.clear_viz_tab()
        
        # Determine the number of subplots needed
        num_plots = 2  # Always have regression and residual plots
        if use_cv:
            num_plots += 1
            if rf_cv_results is not None:
                num_plots += 1
        
        # Create figure with appropriate number of subplots
        if num_plots == 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax3, ax4 = None, None
        elif num_plots == 3:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
            ax4 = None
        else:  # num_plots == 4
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Regression plot
        ax1.scatter(X_test, y_test, alpha=0.5, color='#3CB371', label='Actual')
        ax1.plot(X_test, y_pred, color='#2E8B57', linewidth=2, label='Predicted')
        
        # Highlight manual input point if provided
        if manual_x is not None and manual_y is not None:
            ax1.scatter([manual_x], [manual_y], color='red', s=100, label='Manual Input')
            manual_pred = self.models['lin_reg'].predict([[manual_x]])[0]
            ax1.scatter([manual_x], [manual_pred], color='blue', s=100, label='Prediction')
        
        ax1.set_xlabel(self.x_var.get(), color='#2E8B57')
        ax1.set_ylabel(self.y_var.get(), color='#2E8B57')
        ax1.set_title('Linear Regression', color='#2E8B57')
        ax1.legend()
        
        # Residual plot
        residuals = y_test - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5, color='#3CB371')
        ax2.axhline(y=0, color='#2E8B57', linestyle='--')
        ax2.set_xlabel('Predictions', color='#2E8B57')
        ax2.set_ylabel('Residuals', color='#2E8B57')
        ax2.set_title('Residual Analysis', color='#2E8B57')
        
        # CV plot if applicable
        if use_cv and ax3 is not None:
            ax3.bar(range(1, len(cv_results['scores'])+1), cv_results['scores'], color='#77DD77')
            ax3.axhline(cv_results['mean_mse'], color='#2E8B57', linestyle='--', label='Mean MSE')
            ax3.set_xlabel('Fold', color='#2E8B57')
            ax3.set_ylabel('MSE', color='#2E8B57')
            ax3.set_title('Linear Regression CV Scores', color='#2E8B57')
            ax3.legend()
            
            # RF CV comparison plot if applicable
            if rf_cv_results is not None and ax4 is not None:
                ax4.bar(range(1, len(rf_cv_results['scores'])+1), rf_cv_results['scores'], color='#98FB98')
                ax4.axhline(rf_cv_results['mean_mse'], color='#228B22', linestyle='--', label='Mean MSE')
                ax4.set_xlabel('Fold', color='#2E8B57')
                ax4.set_ylabel('MSE', color='#2E8B57')
                ax4.set_title('Random Forest CV Scores', color='#2E8B57')
                ax4.legend()
        
        # Style
        for ax in [ax for ax in [ax1, ax2, ax3, ax4] if ax is not None]:
            ax.tick_params(colors='#2E8B57')
            for spine in ax.spines.values():
                spine.set_edgecolor('#3CB371')
        
        plt.tight_layout()
        
        # Display in interface
        canvas = FigureCanvasTkAgg(fig, master=self.viz_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Toolbar
        toolbar = NavigationToolbar2Tk(canvas, self.viz_tab)
        toolbar.update()
        canvas._tkcanvas.pack(fill=tk.BOTH, expand=True)
    
    def run_random_forest(self, manual_x=None, manual_y=None):
        """Run random forest model"""
        self.clear_output()
        
        if self.current_dataset is None:
            messagebox.showerror("Error", "Please load or generate data first")
            return
        
        try:
            # Get parameters
            n_estimators = self.model_params['random_forest']['n_estimators_scale'].get()
            max_depth = self.model_params['random_forest']['max_depth_scale'].get()
            max_depth = None if max_depth == 0 else max_depth
            min_samples_split = self.model_params['random_forest']['min_samples_split_scale'].get()
            min_samples_leaf = self.model_params['random_forest']['min_samples_leaf_scale'].get()
            max_features = self.model_params['random_forest']['max_features_combo'].get()
            random_state = self.model_params['lin_reg']['random_state']
            is_classification = self.rf_type.get() == "classification"
            
            # Select variables
            X = self.current_dataset[[self.x_var.get()]].values
            y = self.current_dataset[self.y_var.get()].values
            
            
            # Filtrage selon les bornes min/max X définies par l'utilisateur
            min_x = self.model_params['lin_reg']['min_x_reg'].get()
            max_x = self.model_params['lin_reg']['max_x_reg'].get()
            if min_x > max_x:
                messagebox.showerror("Erreur", "Min X ne peut pas être supérieur à Max X.")
                return
            mask = (X[:, 0] >= min_x) & (X[:, 0] <= max_x)
            X = X[mask]
            y = y[mask]
            if len(X) == 0:
                messagebox.showerror("Erreur", "Aucune donnée dans l'intervalle choisi pour X.")
                return

# For classification, convert y to integer labels if needed
            if is_classification:
                y = y.astype(int)
            
            # Normalize if selected
            if self.normalize_var.get():
                X = (X - X.mean()) / X.std()
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_state)
            
            # Train model
            if is_classification:
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=random_state)
            else:
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=random_state)
            
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            if is_classification:
                accuracy = accuracy_score(y_test, y_pred)
                conf_matrix = confusion_matrix(y_test, y_pred)
            else:
                mse = mean_squared_error(y_test, y_pred)
            
            # Display results
            self.text_output.insert(tk.END, "=== RANDOM FOREST ===\n\n")
            self.text_output.insert(tk.END, f"Model Type: {'Classification' if is_classification else 'Regression'}\n")
            self.text_output.insert(tk.END, f"X Variable: {self.x_var.get()}\n")
            self.text_output.insert(tk.END, f"Y Variable: {self.y_var.get()}\n\n")
            
            self.text_output.insert(tk.END, "=== MODEL PARAMETERS ===\n")
            self.text_output.insert(tk.END, f"Number of Trees: {n_estimators}\n")
            self.text_output.insert(tk.END, f"Max Depth: {'Unlimited' if max_depth is None else max_depth}\n")
            self.text_output.insert(tk.END, f"Min Samples Split: {min_samples_split}\n")
            self.text_output.insert(tk.END, f"Min Samples Leaf: {min_samples_leaf}\n")
            self.text_output.insert(tk.END, f"Max Features: {max_features}\n")
            self.text_output.insert(tk.END, f"Random State: {random_state}\n")
            
            self.text_output.insert(tk.END, "\n=== MODEL RESULTS ===\n")
            if is_classification:
                self.text_output.insert(tk.END, f"Accuracy: {accuracy:.2%}\n\n")
                self.text_output.insert(tk.END, "Confusion Matrix:\n")
                self.text_output.insert(tk.END, str(conf_matrix) + "\n")
            else:
                self.text_output.insert(tk.END, f"MSE: {mse:.4f}\n\n")
            
            self.text_output.insert(tk.END, "Feature Importances:\n")
            for name, importance in zip([self.x_var.get()], model.feature_importances_):
                self.text_output.insert(tk.END, f"- {name}: {importance:.4f}\n")
            
            # Add data statistics
            self.text_output.insert(tk.END, "\n=== DATA STATISTICS ===\n")
            self.text_output.insert(tk.END, f"Mean of X: {X.mean():.4f}\n")
            self.text_output.insert(tk.END, f"Std of X: {X.std():.4f}\n")
            
            if not is_classification:
                self.text_output.insert(tk.END, f"Mean of Y: {y.mean():.4f}\n")
                self.text_output.insert(tk.END, f"Std of Y: {y.std():.4f}\n")
                
                # Add prediction analysis
                residuals = y_test - y_pred
                self.text_output.insert(tk.END, "\n=== PREDICTION ANALYSIS ===\n")
                self.text_output.insert(tk.END, f"Mean residual: {residuals.mean():.4f}\n")
                self.text_output.insert(tk.END, f"Std of residuals: {residuals.std():.4f}\n")
            
            # Store model
            self.models['random_forest'] = model
            
            # Visualization
            self.show_random_forest_plots(model, X_test, y_test, y_pred, is_classification, manual_x, manual_y)
            
            # Update all models visualization
            self.show_all_models_visualization(manual_x, manual_y)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error in Random Forest:\n{str(e)}")
    
    def show_random_forest_plots(self, model, X_test, y_test, y_pred, is_classification, manual_x=None, manual_y=None):
        """Visualize random forest results"""
        self.clear_viz_tab()
        
        if is_classification:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        else:
            fig, ax1 = plt.subplots(figsize=(8, 5))
            ax2 = None
        
        # Feature importance
        features = [self.x_var.get()]
        importances = model.feature_importances_
        
        ax1.barh(range(len(features)), importances, color='#3CB371')
        ax1.set_yticks(range(len(features)))
        ax1.set_yticklabels(features, color='#2E8B57')
        ax1.set_xlabel('Importance', color='#2E8B57')
        ax1.set_title('Feature Importance', color='#2E8B57')
        
        # For regression, show actual vs predicted
        if not is_classification:
            ax1.clear()
            ax1.scatter(X_test, y_test, alpha=0.5, color='#3CB371', label='Actual')
            
            # Sort values for line plot
            sorted_idx = np.argsort(X_test.flatten())
            X_sorted = X_test[sorted_idx]
            y_pred_sorted = y_pred[sorted_idx]
            ax1.plot(X_sorted, y_pred_sorted, color='#2E8B57', linewidth=2, label='Predicted')
            
            # Highlight manual input point if provided
            if manual_x is not None and manual_y is not None:
                ax1.scatter([manual_x], [manual_y], color='red', s=100, label='Manual Input')
                if hasattr(model, 'predict'):
                    manual_pred = model.predict([[manual_x]])[0]
                    ax1.scatter([manual_x], [manual_pred], color='blue', s=100, label='Prediction')
            
            ax1.set_xlabel(self.x_var.get(), color='#2E8B57')
            ax1.set_ylabel(self.y_var.get(), color='#2E8B57')
            ax1.set_title('Actual vs Predicted', color='#2E8B57')
            ax1.legend()
        
        # Confusion matrix for classification
        if is_classification:
            conf_matrix = confusion_matrix(y_test, y_pred)
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', ax=ax2)
            ax2.set_xlabel('Predicted', color='#2E8B57')
            ax2.set_ylabel('Actual', color='#2E8B57')
            ax2.set_title('Confusion Matrix', color='#2E8B57')
        
        # Style
        for ax in [ax1, ax2] if ax2 else [ax1]:
            ax.tick_params(colors='#2E8B57')
            for spine in ax.spines.values():
                spine.set_edgecolor('#3CB371')
        
        # Display in interface
        canvas = FigureCanvasTkAgg(fig, master=self.viz_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Toolbar
        toolbar = NavigationToolbar2Tk(canvas, self.viz_tab)
        toolbar.update()
        canvas._tkcanvas.pack(fill=tk.BOTH, expand=True)
    
    def run_kmeans(self, manual_x=None, manual_y=None):
        """Run K-Means clustering"""
        self.clear_output()
        
        if self.current_dataset is None:
            messagebox.showerror("Error", "Please load or generate data first")
            return
        
        try:
            # Get parameters
            n_clusters = self.model_params['kmeans']['n_clusters_scale'].get()
            min_samples = self.model_params['kmeans']['min_samples_scale'].get()
            max_iter = self.model_params['kmeans']['max_iter_scale'].get()
            random_state = self.model_params['lin_reg']['random_state']
            
            # Select variables
            X = self.current_dataset[[self.x_var.get()]].values
            
            # Normalize if selected
            if self.normalize_var.get():
                X = (X - X.mean()) / X.std()
            
            # Train model
            model = KMeans(
                n_clusters=n_clusters, 
                random_state=random_state,
                max_iter=max_iter
            )
            model.fit(X)
            labels = model.predict(X)
            
            # Calculate silhouette score
            if n_clusters > 1:
                silhouette = silhouette_score(X, labels)
            else:
                silhouette = None
            
            # Display results
            self.text_output.insert(tk.END, "=== K-MEANS CLUSTERING ===\n\n")
            self.text_output.insert(tk.END, f"Variable used: {self.x_var.get()}\n")
            
            self.text_output.insert(tk.END, "\n=== MODEL PARAMETERS ===\n")
            self.text_output.insert(tk.END, f"Number of clusters: {n_clusters}\n")
            self.text_output.insert(tk.END, f"Min Samples per Cluster: {min_samples}\n")
            self.text_output.insert(tk.END, f"Max Iterations: {max_iter}\n")
            self.text_output.insert(tk.END, f"Random State: {random_state}\n")
            
            self.text_output.insert(tk.END, "\n=== MODEL RESULTS ===\n")
            self.text_output.insert(tk.END, "Cluster centers:\n")
            for i, center in enumerate(model.cluster_centers_):
                self.text_output.insert(tk.END, f"Cluster {i}: {center[0]:.2f}\n")
            
            # Add data statistics
            self.text_output.insert(tk.END, "\n=== DATA STATISTICS ===\n")
            self.text_output.insert(tk.END, f"Mean: {X.mean():.4f}\n")
            self.text_output.insert(tk.END, f"Std: {X.std():.4f}\n")
            self.text_output.insert(tk.END, f"Min: {X.min():.4f}\n")
            self.text_output.insert(tk.END, f"Max: {X.max():.4f}\n")
            
            if silhouette is not None:
                self.text_output.insert(tk.END, "\n=== CLUSTER QUALITY ===\n")
                self.text_output.insert(tk.END, f"Silhouette Score: {silhouette:.4f}\n")
                self.text_output.insert(tk.END, "Interpretation:\n")
                if silhouette > 0.7:
                    self.text_output.insert(tk.END, "- Strong structure\n")
                elif silhouette > 0.5:
                    self.text_output.insert(tk.END, "- Reasonable structure\n")
                elif silhouette > 0.25:
                    self.text_output.insert(tk.END, "- Weak structure\n")
                else:
                    self.text_output.insert(tk.END, "- No substantial structure\n")
                
                # Add inter-cluster distances
                if len(model.cluster_centers_) > 1:
                    distances = euclidean_distances(model.cluster_centers_)
                    self.text_output.insert(tk.END, "\nInter-cluster distances:\n")
                    for i in range(len(distances)):
                        for j in range(i+1, len(distances)):
                            self.text_output.insert(tk.END, f"Cluster {i} to {j}: {distances[i,j]:.4f}\n")
            
            # Store model
            self.models['kmeans'] = model
            
            # Visualization
            self.show_kmeans_plot(X, labels, model.cluster_centers_, manual_x, manual_y)
            
            # Update all models visualization
            self.show_all_models_visualization(manual_x, manual_y)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error in K-Means:\n{str(e)}")
    
    def show_kmeans_plot(self, X, labels, centers, manual_x=None, manual_y=None):
        """Visualize K-Means results"""
        self.clear_viz_tab()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Create color palette - using greens as in old code
        colors = ['#C1E1C1', '#77DD77', '#98FB98']  # Palette verte pastel
        
        # Plot points with 'x' markers as in old code
        for i in range(len(centers)):
            ax.scatter(X[labels == i], np.zeros_like(X[labels == i]), 
                      marker='x', color=colors[i % len(colors)], 
                      label=f'Cluster {i}', alpha=0.8)
        
        # Plot centers with 'X' marker as in old code
        ax.scatter(centers, np.zeros_like(centers), 
                  marker='X', s=200, c='#2E8B57', 
                  linewidths=2, label='Centroides')
        
        # Highlight manual input point if provided
        if manual_x is not None:
            manual_point = np.array([[manual_x]])
            if self.normalize_var.get():
                # Need to normalize the manual point the same way
                mean = self.current_dataset[self.x_var.get()].mean()
                std = self.current_dataset[self.x_var.get()].std()
                manual_point = (manual_point - mean) / std
            
            cluster = self.models['kmeans'].predict(manual_point)[0]
            ax.scatter(manual_point, [0], color='red', s=100, 
                      marker='o', label='Manual Input')
            ax.scatter(centers[cluster], [0], color='blue', s=100, 
                      marker='o', label='Cluster Center')
        
        ax.set_xlabel(self.x_var.get(), color='#2E8B57')
        ax.set_title('K-Means Clustering', color='#2E8B57')
        ax.legend()
        ax.tick_params(colors='#2E8B57')
        
        for spine in ax.spines.values():
            spine.set_edgecolor('#3CB371')
        
        # Display in interface
        canvas = FigureCanvasTkAgg(fig, master=self.viz_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Toolbar
        toolbar = NavigationToolbar2Tk(canvas, self.viz_tab)
        toolbar.update()
        canvas._tkcanvas.pack(fill=tk.BOTH, expand=True)
    
    def run_arima(self, manual_x=None):
        """Run ARIMA model"""
        self.clear_output()
        
        if self.current_dataset is None:
            messagebox.showerror("Error", "Please load or generate data first")
            return
        
        try:
            # Get parameters
            p = self.model_params['arima']['p_var'].get()
            d = self.model_params['arima']['d_var'].get()
            q = self.model_params['arima']['q_var'].get()
            steps = self.model_params['arima']['steps_scale'].get()
            
            # Select variable
            data = self.current_dataset[self.x_var.get()].values
            
            # Normalize if selected
            if self.normalize_var.get():
                data = (data - data.mean()) / data.std()
            
            # Train model
            model = ARIMA(data, order=(p,d,q))
            results = model.fit()
            forecast = results.forecast(steps=steps)
            
            # Display results
            self.text_output.insert(tk.END, "=== ARIMA MODEL ===\n\n")
            self.text_output.insert(tk.END, f"Variable used: {self.x_var.get()}\n")
            
            self.text_output.insert(tk.END, "\n=== MODEL PARAMETERS ===\n")
            self.text_output.insert(tk.END, f"Parameters (p,d,q): ({p},{d},{q})\n")
            self.text_output.insert(tk.END, f"Forecast steps: {steps}\n")
            
            self.text_output.insert(tk.END, "\n=== MODEL RESULTS ===\n")
            self.text_output.insert(tk.END, "Forecast:\n")
            for i, val in enumerate(forecast, 1):
                self.text_output.insert(tk.END, f"Step {i}: {val:.2f}\n")
            
            # Add data statistics
            self.text_output.insert(tk.END, "\n=== DATA STATISTICS ===\n")
            self.text_output.insert(tk.END, f"Mean: {data.mean():.4f}\n")
            self.text_output.insert(tk.END, f"Std: {data.std():.4f}\n")
            self.text_output.insert(tk.END, f"Min: {data.min():.4f}\n")
            self.text_output.insert(tk.END, f"Max: {data.max():.4f}\n")
            
            # Add model summary
            self.text_output.insert(tk.END, "\n=== MODEL SUMMARY ===\n")
            self.text_output.insert(tk.END, str(results.summary()))
            
            # Store model
            self.models['arima'] = results
            
            # Visualization
            self.show_arima_plot(data, forecast, manual_x)
            
            # Update all models visualization
            self.show_all_models_visualization(manual_x)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error in ARIMA:\n{str(e)}")

    
    def show_arima_plot(self, data, forecast, manual_x=None):
        """Visualize ARIMA results"""
        self.clear_viz_tab()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Historical data
        ax.plot(data, color='#3CB371', label='Historical Data')
        
        # Forecast
        forecast_index = range(len(data), len(data)+len(forecast))
        ax.plot(forecast_index, forecast, color='#2E8B57', linestyle='--', label='Forecast')
        
        # Highlight manual input point if provided
        if manual_x is not None:
            ax.axvline(x=len(data)-1, color='red', linestyle=':', label='Manual Input Time')
        
        ax.set_xlabel('Time', color='#2E8B57')
        ax.set_ylabel(self.x_var.get(), color='#2E8B57')
        ax.set_title('ARIMA Forecast', color='#2E8B57')
        ax.legend()
        ax.tick_params(colors='#2E8B57')
        
        for spine in ax.spines.values():
            spine.set_edgecolor('#3CB371')
        
        # Display in interface
        canvas = FigureCanvasTkAgg(fig, master=self.viz_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Toolbar
        toolbar = NavigationToolbar2Tk(canvas, self.viz_tab)
        toolbar.update()
        canvas._tkcanvas.pack(fill=tk.BOTH, expand=True)
    
    def show_all_models_visualization(self, manual_x=None, manual_y=None):
        """Show visualization of all models together"""
        # Clear the all models tab
        for widget in self.all_models_tab.winfo_children():
            widget.destroy()
        
        # Create a figure with subplots for each model
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('All Models Visualization', color='#2E8B57', fontsize=16)
        
        # Linear Regression plot
        if self.models['lin_reg'] is not None:
            ax = axes[0, 0]
            X = self.current_dataset[[self.x_var.get()]].values
            y = self.current_dataset[self.y_var.get()].values
            
            
            # Filtrage selon les bornes min/max X définies par l'utilisateur
            min_x = self.model_params['lin_reg']['min_x_reg'].get()
            max_x = self.model_params['lin_reg']['max_x_reg'].get()
            if min_x > max_x:
                messagebox.showerror("Erreur", "Min X ne peut pas être supérieur à Max X.")
                return
            mask = (X[:, 0] >= min_x) & (X[:, 0] <= max_x)
            X = X[mask]
            y = y[mask]
            if len(X) == 0:
                messagebox.showerror("Erreur", "Aucune donnée dans l'intervalle choisi pour X.")
                return

# Predictions for the entire range
            x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_pred = self.models['lin_reg'].predict(x_range)
            
            ax.scatter(X, y, alpha=0.5, color='#3CB371', label='Data')
            ax.plot(x_range, y_pred, color='#2E8B57', linewidth=2, label='Regression')
            
            # Highlight manual input point if provided
            if manual_x is not None and manual_y is not None:
                ax.scatter([manual_x], [manual_y], color='red', s=100, label='Manual Input')
                manual_pred = self.models['lin_reg'].predict([[manual_x]])[0]
                ax.scatter([manual_x], [manual_pred], color='blue', s=100, label='Prediction')
            
            ax.set_title('Linear Regression', color='#2E8B57')
            ax.legend()
        
        # Random Forest plot
        if self.models['random_forest'] is not None:
            ax = axes[0, 1]
            X = self.current_dataset[[self.x_var.get()]].values
            y = self.current_dataset[self.y_var.get()].values
            
            
            # Filtrage selon les bornes min/max X définies par l'utilisateur
            min_x = self.model_params['lin_reg']['min_x_reg'].get()
            max_x = self.model_params['lin_reg']['max_x_reg'].get()
            if min_x > max_x:
                messagebox.showerror("Erreur", "Min X ne peut pas être supérieur à Max X.")
                return
            mask = (X[:, 0] >= min_x) & (X[:, 0] <= max_x)
            X = X[mask]
            y = y[mask]
            if len(X) == 0:
                messagebox.showerror("Erreur", "Aucune donnée dans l'intervalle choisi pour X.")
                return

# Predictions for the entire range
            x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_pred = self.models['random_forest'].predict(x_range)
            
            ax.scatter(X, y, alpha=0.5, color='#3CB371', label='Data')
            ax.plot(x_range, y_pred, color='#2E8B57', linewidth=2, label='Random Forest')
            
            # Highlight manual input point if provided
            if manual_x is not None and manual_y is not None:
                ax.scatter([manual_x], [manual_y], color='red', s=100, label='Manual Input')
                manual_pred = self.models['random_forest'].predict([[manual_x]])[0]
                ax.scatter([manual_x], [manual_pred], color='blue', s=100, label='Prediction')
            
            ax.set_title('Random Forest', color='#2E8B57')
            ax.legend()
        
        # K-Means plot
        if self.models['kmeans'] is not None:
            ax = axes[1, 0]
            X = self.current_dataset[[self.x_var.get()]].values
            labels = self.models['kmeans'].predict(X)
            centers = self.models['kmeans'].cluster_centers_
            
            # Create color palette - same as old code
            colors = ['#C1E1C1', '#77DD77', '#98FB98']
            
            # Plot points with 'x' markers
            for i in range(len(centers)):
                ax.scatter(X[labels == i], np.zeros_like(X[labels == i]), 
                          marker='x', color=colors[i % len(colors)], 
                          label=f'Cluster {i}', alpha=0.8)
            
            # Plot centers with 'X' markers
            ax.scatter(centers, np.zeros_like(centers), 
                      marker='X', s=200, c='#2E8B57', 
                      linewidths=2, label='Centroides')
            
            # Highlight manual input point if provided
            if manual_x is not None:
                manual_point = np.array([[manual_x]])
                if self.normalize_var.get():
                    mean = self.current_dataset[self.x_var.get()].mean()
                    std = self.current_dataset[self.x_var.get()].std()
                    manual_point = (manual_point - mean) / std
                
                cluster = self.models['kmeans'].predict(manual_point)[0]
                ax.scatter(manual_point, [0], color='red', s=100, 
                          marker='o', label='Manual Input')
                ax.scatter(centers[cluster], [0], color='blue', s=100, 
                          marker='o', label='Cluster Center')
            
            ax.set_title('K-Means Clustering', color='#2E8B57')
            ax.legend()
        
        # ARIMA plot
        if self.models['arima'] is not None:
            ax = axes[1, 1]
            data = self.current_dataset[self.x_var.get()].values
            forecast = self.models['arima'].forecast(steps=self.model_params['arima']['steps_scale'].get())
            
            ax.plot(data, color='#3CB371', label='Historical Data')
            forecast_index = range(len(data), len(data)+len(forecast))
            ax.plot(forecast_index, forecast, color='#2E8B57', linestyle='--', label='Forecast')
            
            # Highlight manual input point if provided
            if manual_x is not None:
                ax.axvline(x=len(data)-1, color='red', linestyle=':', label='Manual Input Time')
            
            ax.set_title('ARIMA Forecast', color='#2E8B57')
            ax.legend()
        
        # Style all axes
        for ax in axes.flat:
            if ax:
                ax.tick_params(colors='#2E8B57')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#3CB371')
        
        plt.tight_layout()
        
        # Display in interface
        canvas = FigureCanvasTkAgg(fig, master=self.all_models_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Toolbar
        toolbar = NavigationToolbar2Tk(canvas, self.all_models_tab)
        toolbar.update()
        canvas._tkcanvas.pack(fill=tk.BOTH, expand=True)
    
    # ===================
    # UTILITY METHODS
    # ===================
    
    def export_results(self):
        """Export text results to file"""
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                               filetypes=[("Text files", "*.txt"),
                                                          ("All files", "*.*")])
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.text_output.get(1.0, tk.END))
                messagebox.showinfo("Success", "Results exported successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Error exporting results:\n{str(e)}")
    
    def export_data(self):
        """Export current dataset to file"""
        if self.current_dataset is None:
            messagebox.showerror("Error", "No data to export")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                               filetypes=[("CSV files", "*.csv"),
                                                          ("All files", "*.*")])
        if file_path:
            try:
                self.current_dataset.to_csv(file_path, index=False)
                messagebox.showinfo("Success", "Data exported successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Error exporting data:\n{str(e)}")
    
    def save_model(self, model_type):
        """Save trained model to file"""
        if self.models[model_type] is None:
            messagebox.showerror("Error", f"No {model_type} model trained yet")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".pkl",
                                               filetypes=[("Pickle files", "*.pkl"),
                                                          ("All files", "*.*")])
        if file_path:
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(self.models[model_type], f)
                messagebox.showinfo("Success", f"{model_type} model saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving model:\n{str(e)}")
    
    def load_model(self):
        """Load a trained model from file"""
        file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl"),
                                                        ("All files", "*.*")])
        if file_path:
            try:
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
                
                # Determine model type
                model_type = None
                if isinstance(model, LinearRegression):
                    model_type = 'lin_reg'
                elif isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
                    model_type = 'random_forest'
                elif isinstance(model, KMeans):
                    model_type = 'kmeans'
                elif hasattr(model, 'forecast'):  # ARIMA results
                    model_type = 'arima'
                
                if model_type:
                    self.models[model_type] = model
                    messagebox.showinfo("Success", f"{model_type} model loaded successfully!")
                else:
                    messagebox.showerror("Error", "Unknown model type")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading model:\n{str(e)}")
    
    def return_to_home(self):
        """Return to home page"""
        for widget in self.root.winfo_children():
            widget.destroy()
        self.create_home_page()

if __name__ == "__main__":
    root = tk.Tk()
    app = manalapp(root)
    root.mainloop()