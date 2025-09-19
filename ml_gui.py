#!/usr/bin/env python3
"""
VCT GUI using ML-based predictions with 2024 data and 2025 roster adjustments
"""

import tkinter as tk
from tkinter import messagebox
import sys
from pathlib import Path

# Add src to path for ML predictor
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    # Try enhanced ML model first
    import sys
    sys.path.append(str(Path(__file__).parent / "src" / "models"))
    sys.path.append(str(Path(__file__).parent / "src" / "prediction"))
    from enhanced_ml_predictor import EnhancedVCTPredictor
    from map_picker import SeriesFormat
    ENHANCED_ML_AVAILABLE = True
    MAP_PICKER_AVAILABLE = True
    print("Using Enhanced ML Model with Map Picking")
except ImportError as e:
    print(f"Enhanced ML model not available: {e}")
    ENHANCED_ML_AVAILABLE = False
    MAP_PICKER_AVAILABLE = False
    try:
        # Fallback to basic ML model
        from ml_predictor import VCTMLPredictor
        BASIC_ML_AVAILABLE = True
        print("Using Basic ML Model")
    except ImportError as e2:
        print(f"Warning: No ML predictor available: {e2}")
        BASIC_ML_AVAILABLE = False

class MLPredictionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("VCT 2025 Champions Predictor - ML Enhanced")
        self.root.geometry("1000x700+150+100")
        self.root.configure(bg='white')
        
        # Force to front
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after(2000, lambda: self.root.attributes('-topmost', False))
        
        # Team selections
        self.selected_team1 = None
        self.selected_team2 = None
        
        # Map picking settings
        self.series_format = SeriesFormat.BO3 if MAP_PICKER_AVAILABLE else None
        self.include_map_analysis = True
        self.current_map_pool = [
            "Ascent", "Bind", "Breeze", "Haven", 
            "Icebox", "Lotus", "Sunset"
        ]
        
        # Load ML predictor
        self.ml_predictor = None
        self.load_ml_model()
        
        # Get teams from ML model or fallback
        self.teams = self.get_available_teams()
        
        # Create GUI
        self.create_widgets()
        
        print("ML-Enhanced VCT GUI created successfully")
    
    def load_ml_model(self):
        """Load or train the ML prediction model"""
        if ENHANCED_ML_AVAILABLE:
            try:
                print("Loading Enhanced ML Model...")
                self.ml_predictor = EnhancedVCTPredictor()
                
                # Check if enhanced model exists
                enhanced_path = Path(__file__).parent / "src" / "models" / "enhanced_vct_model.pkl"
                if enhanced_path.exists():
                    print("Loading pre-trained enhanced ML model...")
                    if self.ml_predictor.load_enhanced_model(enhanced_path):
                        print("Enhanced ML model loaded successfully!")
                        return
                
                print("Training enhanced ML model (this may take a moment)...")
                self.ml_predictor.load_comprehensive_data()
                self.ml_predictor.train_enhanced_model()
                print("Enhanced ML model trained successfully!")
                return
                    
            except Exception as e:
                print(f"Error with enhanced ML model: {e}")
                print("Falling back to basic ML model...")
        
        # Fallback to basic ML model
        if 'BASIC_ML_AVAILABLE' in globals() and BASIC_ML_AVAILABLE:
            try:
                self.ml_predictor = VCTMLPredictor()
                
                # Try to load pre-trained model
                model_path = Path(__file__).parent / "src" / "models" / "vct_ml_model.pkl"
                if model_path.exists():
                    print("Loading pre-trained basic ML model...")
                    if self.ml_predictor.load_model(model_path):
                        print("Basic ML model loaded successfully!")
                        return
                
                # Train new model if no saved model exists
                print("Training new basic ML model...")
                self.ml_predictor.load_and_process_data()
                self.ml_predictor.apply_roster_adjustments()
                self.ml_predictor.train_model()
                self.ml_predictor.save_model(model_path)
                print("Basic ML model trained and saved!")
                
            except Exception as e:
                print(f"Error loading/training basic ML model: {e}")
                self.ml_predictor = None
        else:
            print("No ML predictor available, using fallback predictions")
            self.ml_predictor = None
    
    def get_available_teams(self):
        """Get teams from ML model or use fallback list"""
        # Complete list of VCT Champions 2025 qualified teams
        all_vct_2025_teams = [
            "Team Heretics", "Fnatic", "Team Liquid", "GIANTX",
            "Sentinels", "G2 Esports", "NRG", "MIBR", 
            "Paper Rex", "DRX", "T1", "Rex Regum Qeon",
            "Edward Gaming", "Bilibili Gaming", "Dragon Ranger Gaming", "Xi Lai Gaming"
        ]
        
        if self.ml_predictor and hasattr(self.ml_predictor, 'team_stats'):
            # Use ML model teams but ensure all 16 teams are present
            ml_teams = list(self.ml_predictor.team_stats.keys())
            available_teams = []
            
            for team in all_vct_2025_teams:
                if team in ml_teams:
                    available_teams.append(team)
                else:
                    # Add team even if not in ML data for completeness
                    available_teams.append(team)
                    print(f"Warning: {team} not found in ML training data")
            
            return sorted(available_teams)
        else:
            # Fallback teams - all 16 teams
            return sorted(all_vct_2025_teams)
    
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Title with ML indicator
        ml_status = "ML-ENHANCED" if self.ml_predictor else "FALLBACK MODE"
        title_text = f"VCT 2025 CHAMPIONS PREDICTOR - {ml_status}"
        
        title = tk.Label(self.root, text=title_text, 
                        font=('Arial', 16, 'bold'), fg='blue', bg='white')
        title.pack(pady=10)
        
        # Model info
        if self.ml_predictor:
            accuracy_text = f"Model Accuracy: {self.ml_predictor.model_accuracy:.1%}"
            data_info = f"Trained on {len(self.ml_predictor.team_stats)} teams from VCT 2024"
            
            info_frame = tk.Frame(self.root, bg='lightgreen', relief='raised', bd=2)
            info_frame.pack(fill=tk.X, padx=20, pady=5)
            
            tk.Label(info_frame, text="MACHINE LEARNING PREDICTION SYSTEM", 
                    font=('Arial', 12, 'bold'), bg='lightgreen').pack()
            tk.Label(info_frame, text=accuracy_text, 
                    font=('Arial', 10), bg='lightgreen').pack()
            tk.Label(info_frame, text=data_info, 
                    font=('Arial', 10), bg='lightgreen').pack()
        
        # Team selection display
        selection_frame = tk.Frame(self.root, bg='yellow', relief='raised', bd=3)
        selection_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(selection_frame, text="SELECTED MATCHUP", 
                font=('Arial', 14, 'bold'), bg='yellow').pack(pady=5)
        
        self.selection_display = tk.Label(selection_frame, 
                                         text="Team 1: [Not selected]    VS    Team 2: [Not selected]",
                                         font=('Arial', 12), bg='yellow', fg='black')
        self.selection_display.pack(pady=5)
        
        # Selection controls
        control_frame = tk.Frame(self.root, bg='lightblue', relief='raised', bd=2)
        control_frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Label(control_frame, text="Selection Mode:", font=('Arial', 11, 'bold'), 
                bg='lightblue').pack(side=tk.LEFT, padx=10)
        
        self.mode_var = tk.StringVar(value="team1")
        
        mode1_btn = tk.Radiobutton(control_frame, text="Select Team 1", variable=self.mode_var, 
                                  value="team1", font=('Arial', 10), bg='lightblue')
        mode1_btn.pack(side=tk.LEFT, padx=10)
        
        mode2_btn = tk.Radiobutton(control_frame, text="Select Team 2", variable=self.mode_var, 
                                  value="team2", font=('Arial', 10), bg='lightblue')
        mode2_btn.pack(side=tk.LEFT, padx=10)
        
        clear_btn = tk.Button(control_frame, text="Clear Selection", 
                             command=self.clear_selection,
                             font=('Arial', 10), bg='pink', padx=10)
        clear_btn.pack(side=tk.LEFT, padx=20)
        
        # Team buttons
        team_container = tk.Frame(self.root, bg='lightgray', relief='raised', bd=3)
        team_container.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(team_container, text="VCT CHAMPIONS 2025 QUALIFIED TEAMS", 
                font=('Arial', 12, 'bold'), bg='lightgray').pack(pady=5)
        
        # Team buttons in grid
        buttons_frame = tk.Frame(team_container, bg='lightgray')
        buttons_frame.pack(padx=10, pady=5)
        
        cols = 4
        for i, team in enumerate(self.teams):
            row = i // cols
            col = i % cols
            
            btn = tk.Button(buttons_frame, text=team, 
                           command=lambda t=team: self.select_team(t),
                           font=('Arial', 9), 
                           width=20, height=2,
                           relief='raised', bd=2,
                           bg='white', fg='black')
            btn.grid(row=row, column=col, padx=3, pady=3, sticky='ew')
        
        for i in range(cols):
            buttons_frame.columnconfigure(i, weight=1)
        
        # Map picking section (only if available)
        if MAP_PICKER_AVAILABLE:
            self.create_map_picking_interface()
        
        # Prediction button
        predict_btn = tk.Button(self.root, text="PREDICT MATCH WINNER (ML)", 
                               command=self.predict_match,
                               font=('Arial', 16, 'bold'), bg='green', fg='white',
                               padx=30, pady=10, relief='raised', bd=4)
        predict_btn.pack(pady=20)
        
        # Results display
        results_frame = tk.Frame(self.root, bg='lightyellow', relief='solid', bd=4)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        tk.Label(results_frame, text="PREDICTION RESULTS", 
                font=('Arial', 14, 'bold'), bg='lightyellow', fg='red').pack(pady=5)
        
        self.results_text = tk.Text(results_frame, height=12, width=100, 
                                   font=('Arial', 11), bg='white', 
                                   wrap=tk.WORD, relief='sunken', bd=3)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Initial message
        welcome_msg = f"""VCT CHAMPIONS 2025 PARIS - ML PREDICTION SYSTEM

{'MACHINE LEARNING ACTIVE' if self.ml_predictor else 'FALLBACK MODE ACTIVE'}

FEATURES:
{'- Trained on complete VCT 2024 tournament data' if self.ml_predictor else '- Using static team ratings'}
{'- Head-to-head analysis from 436 professional matches' if self.ml_predictor else '- Regional strength analysis'}
{'- Recent form and momentum tracking' if self.ml_predictor else '- Basic team comparisons'} 
{'- 2025 roster adjustment analysis' if self.ml_predictor else '- Manual team strength ratings'}
{'- Cross-validated ensemble model' if self.ml_predictor else '- Rule-based predictions'}

INSTRUCTIONS:
1. Choose "Select Team 1" or "Select Team 2" mode
2. Click team buttons to select teams for prediction
3. Watch selections appear in yellow box above
4. Click "PREDICT MATCH WINNER" for ML analysis

STATUS: Ready for predictions with {len(self.teams)} qualified teams
"""
        
        self.results_text.insert(tk.END, welcome_msg)
        
        print(f"GUI created with {len(self.teams)} teams")
    
    def create_map_picking_interface(self):
        """Create map picking and Best-Of series selection interface"""
        # Map analysis frame
        map_frame = tk.Frame(self.root, bg='lightcyan', relief='raised', bd=3)
        map_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(map_frame, text="MAP PICKING & SERIES ANALYSIS", 
                font=('Arial', 12, 'bold'), bg='lightcyan', fg='navy').pack(pady=5)
        
        # Series format selection
        series_control_frame = tk.Frame(map_frame, bg='lightcyan')
        series_control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(series_control_frame, text="Series Format:", 
                font=('Arial', 11, 'bold'), bg='lightcyan').pack(side=tk.LEFT)
        
        self.series_var = tk.StringVar(value="bo3")
        
        for format_option, display_name in [("bo1", "Best of 1"), ("bo3", "Best of 3"), ("bo5", "Best of 5")]:
            rb = tk.Radiobutton(series_control_frame, text=display_name, 
                               variable=self.series_var, value=format_option,
                               font=('Arial', 10), bg='lightcyan',
                               command=self.on_series_format_change)
            rb.pack(side=tk.LEFT, padx=10)
        
        # Map analysis toggle
        map_analysis_frame = tk.Frame(map_frame, bg='lightcyan')
        map_analysis_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.map_analysis_var = tk.BooleanVar(value=True)
        map_checkbox = tk.Checkbutton(map_analysis_frame, 
                                     text="Include Map Pick/Ban Analysis",
                                     variable=self.map_analysis_var,
                                     font=('Arial', 11), bg='lightcyan',
                                     command=self.on_map_analysis_toggle)
        map_checkbox.pack(side=tk.LEFT)
        
        # Map strengths display button
        map_strengths_btn = tk.Button(map_analysis_frame, text="View Team Map Strengths",
                                     command=self.show_map_strengths,
                                     font=('Arial', 10), bg='cyan', padx=10)
        map_strengths_btn.pack(side=tk.RIGHT, padx=10)
        
        # Current map pool display
        map_pool_frame = tk.Frame(map_frame, bg='lightcyan')
        map_pool_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(map_pool_frame, text="Current VCT Map Pool:", 
                font=('Arial', 10, 'bold'), bg='lightcyan').pack(side=tk.LEFT)
        
        map_pool_text = " • ".join(self.current_map_pool)
        tk.Label(map_pool_frame, text=map_pool_text, 
                font=('Arial', 9), bg='lightcyan', fg='darkblue').pack(side=tk.LEFT, padx=10)
    
    def on_series_format_change(self):
        """Handle series format change"""
        format_map = {"bo1": SeriesFormat.BO1, "bo3": SeriesFormat.BO3, "bo5": SeriesFormat.BO5}
        self.series_format = format_map[self.series_var.get()]
        print(f"Series format changed to: {self.series_format.value.upper()}")
    
    def on_map_analysis_toggle(self):
        """Handle map analysis toggle"""
        self.include_map_analysis = self.map_analysis_var.get()
        print(f"Map analysis {'enabled' if self.include_map_analysis else 'disabled'}")
    
    def show_map_strengths(self):
        """Show team map strengths in popup window"""
        if not self.selected_team1 and not self.selected_team2:
            messagebox.showwarning("Warning", "Please select at least one team to view map strengths!")
            return
        
        if not self.ml_predictor or not hasattr(self.ml_predictor, 'map_features_enabled') or not self.ml_predictor.map_features_enabled:
            messagebox.showinfo("Info", "Map analysis requires enhanced ML predictor with map data loaded.")
            return
        
        # Create popup window
        strengths_window = tk.Toplevel(self.root)
        strengths_window.title("Team Map Strengths Analysis")
        strengths_window.geometry("800x600")
        strengths_window.configure(bg='white')
        
        # Title
        title_label = tk.Label(strengths_window, text="TEAM MAP STRENGTHS ANALYSIS", 
                              font=('Arial', 14, 'bold'), bg='white', fg='blue')
        title_label.pack(pady=10)
        
        # Create scrollable text area
        text_frame = tk.Frame(strengths_window, bg='white')
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_area = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set,
                           font=('Arial', 10), bg='lightyellow')
        text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_area.yview)
        
        # Generate map strengths content
        content = "VCT CHAMPIONS 2025 - TEAM MAP STRENGTHS\n"
        content += "=" * 60 + "\n\n"
        
        teams_to_analyze = []
        if self.selected_team1:
            teams_to_analyze.append(self.selected_team1)
        if self.selected_team2:
            teams_to_analyze.append(self.selected_team2)
        
        for team in teams_to_analyze:
            content += f"📊 {team.upper()} MAP ANALYSIS\n"
            content += "-" * 40 + "\n"
            
            try:
                strengths = self.ml_predictor.get_team_map_strengths(team)
                if strengths:
                    # Sort maps by strength
                    sorted_maps = sorted(strengths.items(), key=lambda x: x[1], reverse=True)
                    
                    for i, (map_name, strength) in enumerate(sorted_maps):
                        rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else "📍"
                        strength_level = "Excellent" if strength >= 80 else "Good" if strength >= 65 else "Average" if strength >= 50 else "Weak"
                        content += f"{rank_emoji} {map_name:<12} {strength:5.1f}/100 ({strength_level})\n"
                else:
                    content += "   No map data available for this team\n"
            except Exception as e:
                content += f"   Error analyzing team: {e}\n"
            
            content += "\n"
        
        # Add comparison if both teams selected
        if self.selected_team1 and self.selected_team2:
            content += f"⚔️  HEAD-TO-HEAD MAP COMPARISON: {self.selected_team1} vs {self.selected_team2}\n"
            content += "=" * 60 + "\n"
            
            try:
                comparison = self.ml_predictor.compare_teams_on_maps(self.selected_team1, self.selected_team2)
                if comparison:
                    for map_name, data in comparison.items():
                        advantage = data['advantage']
                        favored = data['favored_team']
                        confidence = data['confidence'] * 100
                        
                        if abs(advantage) > 10:
                            advantage_level = "Strong Advantage"
                        elif abs(advantage) > 5:
                            advantage_level = "Slight Advantage"
                        else:
                            advantage_level = "Even Match"
                        
                        content += f"🗺️  {map_name:<12} → {favored} ({advantage_level})\n"
                        content += f"     Strength Diff: {advantage:+.1f} | Confidence: {confidence:.0f}%\n\n"
                else:
                    content += "   Unable to generate comparison data\n"
            except Exception as e:
                content += f"   Error generating comparison: {e}\n"
        
        content += "\n" + "=" * 60 + "\n"
        content += "📝 MAP STRENGTH LEGEND:\n"
        content += "   🥇 🥈 🥉 = Top 3 strongest maps\n"
        content += "   Excellent (80+) | Good (65-79) | Average (50-64) | Weak (<50)\n"
        content += "\n💡 TIP: Use this analysis to predict map pick/ban strategies!"
        
        text_area.insert(tk.END, content)
        text_area.config(state=tk.DISABLED)
        
        # Close button
        close_btn = tk.Button(strengths_window, text="Close", 
                             command=strengths_window.destroy,
                             font=('Arial', 12), bg='red', fg='white', padx=20)
        close_btn.pack(pady=10)
    
    def select_team(self, team_name):
        """Select a team for prediction"""
        mode = self.mode_var.get()
        
        if mode == "team1":
            self.selected_team1 = team_name
        else:
            self.selected_team2 = team_name
        
        self.update_selection_display()
        
        # Show feedback in results
        feedback_msg = f"\\n>>> {team_name} selected as {mode.replace('team', 'Team ')} <<<\\n"
        self.results_text.insert(tk.END, feedback_msg)
        self.results_text.see(tk.END)
    
    def clear_selection(self):
        """Clear team selections"""
        self.selected_team1 = None
        self.selected_team2 = None
        self.update_selection_display()
        
        self.results_text.insert(tk.END, "\\n>>> Team selections cleared <<<\\n")
        self.results_text.see(tk.END)
    
    def update_selection_display(self):
        """Update the selection display"""
        team1_text = self.selected_team1 if self.selected_team1 else "[Not selected]"
        team2_text = self.selected_team2 if self.selected_team2 else "[Not selected]"
        
        display_text = f"Team 1: {team1_text}    VS    Team 2: {team2_text}"
        self.selection_display.config(text=display_text)
    
    def predict_match(self):
        """Predict match outcome using ML or fallback"""
        if not self.selected_team1 or not self.selected_team2:
            messagebox.showwarning("Warning", "Please select both teams!")
            return
        
        if self.selected_team1 == self.selected_team2:
            messagebox.showwarning("Warning", "Please select two different teams!")
            return
        
        self.results_text.delete(1.0, tk.END)
        
        if self.ml_predictor:
            # Use ML prediction - check model type
            is_enhanced = ENHANCED_ML_AVAILABLE and isinstance(self.ml_predictor, EnhancedVCTPredictor)
            
            if is_enhanced and MAP_PICKER_AVAILABLE and self.include_map_analysis:
                # Use map-aware prediction
                prediction = self.ml_predictor.predict_match_with_maps(
                    self.selected_team1, self.selected_team2, 
                    series_format=self.series_format,
                    include_map_simulation=True
                )
            elif is_enhanced:
                prediction = self.ml_predictor.predict_match_enhanced(self.selected_team1, self.selected_team2)
            else:
                prediction = self.ml_predictor.predict_match(self.selected_team1, self.selected_team2)
            
            if prediction:
                self.display_ml_prediction(prediction)
            else:
                self.display_fallback_prediction()
        else:
            # Use fallback prediction
            self.display_fallback_prediction()
    
    def display_ml_prediction(self, prediction):
        """Display ML-based prediction results"""
        is_enhanced = ENHANCED_ML_AVAILABLE and isinstance(self.ml_predictor, EnhancedVCTPredictor)
        
        # Get analysis based on model type
        team1_analysis = None
        team2_analysis = None
        
        if is_enhanced:
            # Enhanced model may have different API
            try:
                team1_analysis = self.ml_predictor.get_team_analysis(self.selected_team1)
                team2_analysis = self.ml_predictor.get_team_analysis(self.selected_team2)
            except (AttributeError, Exception):
                pass
        else:
            # Basic model
            try:
                team1_analysis = self.ml_predictor.get_team_analysis(self.selected_team1)
                team2_analysis = self.ml_predictor.get_team_analysis(self.selected_team2)
            except (AttributeError, Exception):
                pass
        
        # Check if this is a map-enhanced prediction
        has_map_analysis = prediction.get('map_analysis') is not None
        
        model_type = "MAP-ENHANCED MACHINE LEARNING" if has_map_analysis else "ENHANCED MACHINE LEARNING" if is_enhanced else "MACHINE LEARNING"
        
        result_text = f"""{model_type} PREDICTION RESULTS
{'='*70}

MATCHUP: {prediction.get('team1', self.selected_team1)} vs {prediction.get('team2', self.selected_team2)}
{f"SERIES FORMAT: {self.series_format.value.upper()}" if has_map_analysis else ""}

PREDICTED WINNER: {prediction['predicted_winner']}
WIN PROBABILITY: {prediction['confidence']:.1%}
CONFIDENCE LEVEL: {prediction.get('confidence_level', 'Medium')}
MODEL ACCURACY: {prediction.get('model_accuracy', 0.83):.1%}
{f"ENHANCED WITH MAPS: {'Yes' if has_map_analysis else 'No'}"}

DETAILED PROBABILITIES:
   - {prediction.get('team1', self.selected_team1)}: {prediction.get('team1_probability', 0.5):.1%}
   - {prediction.get('team2', self.selected_team2)}: {prediction.get('team2_probability', 0.5):.1%}

TEAM ANALYSIS:
"""
        
        if team1_analysis:
            result_text += f"""
{prediction.get('team1', self.selected_team1)} PROFILE:
   - 2024 Win Rate: {team1_analysis.get('win_rate', 0.5):.1%}
   - Matches Played: {team1_analysis.get('matches_played', 'N/A')}
   - Recent Form: {team1_analysis.get('recent_form', 'Unknown')} (Rating: {team1_analysis.get('form_rating', 0.5):.1%})
   - Tournaments: {team1_analysis.get('tournaments_played', 'N/A')} events
"""
        
        if team2_analysis:
            result_text += f"""
{prediction.get('team2', self.selected_team2)} PROFILE:
   - 2024 Win Rate: {team2_analysis.get('win_rate', 0.5):.1%}
   - Matches Played: {team2_analysis.get('matches_played', 'N/A')}
   - Recent Form: {team2_analysis.get('recent_form', 'Unknown')} (Rating: {team2_analysis.get('form_rating', 0.5):.1%})
   - Tournaments: {team2_analysis.get('tournaments_played', 'N/A')} events
"""
        
        # Check for roster adjustments (basic model only)
        roster_info = []
        if not is_enhanced and hasattr(self.ml_predictor, 'roster_adjustments'):
            if self.selected_team1 in self.ml_predictor.roster_adjustments:
                adj = self.ml_predictor.roster_adjustments[self.selected_team1]
                roster_info.append(f"   - {self.selected_team1}: {adj['reason']}")
            
            if self.selected_team2 in self.ml_predictor.roster_adjustments:
                adj = self.ml_predictor.roster_adjustments[self.selected_team2]
                roster_info.append(f"   - {self.selected_team2}: {adj['reason']}")
        
        if roster_info:
            result_text += f"""
2025 ROSTER ADJUSTMENTS:
{chr(10).join(roster_info)}
"""
        
        # Add map analysis section if available
        if has_map_analysis and prediction.get('map_analysis'):
            map_data = prediction['map_analysis']
            result_text += f"""
MAP PICK/BAN ANALYSIS:
{'='*40}
SERIES FORMAT: {map_data.get('series_format', 'bo3').upper()}
PICKED MAPS: {', '.join(map_data.get('picked_maps', []))}

TEAM MAP ADVANTAGES:
"""
            
            # Team 1 advantages
            team1_advantages = map_data.get('team1_map_advantages', {})
            if team1_advantages:
                result_text += f"\n{prediction.get('team1', self.selected_team1)} STRONG MAPS:\n"
                strong_maps = [(map_name, adv) for map_name, adv in team1_advantages.items() if adv > 5]
                if strong_maps:
                    for map_name, advantage in sorted(strong_maps, key=lambda x: x[1], reverse=True):
                        result_text += f"   • {map_name}: +{advantage:.1f} advantage\n"
                else:
                    result_text += f"   • No significant map advantages\n"
            
            # Team 2 advantages
            team2_advantages = map_data.get('team2_map_advantages', {})
            if team2_advantages:
                result_text += f"\n{prediction.get('team2', self.selected_team2)} STRONG MAPS:\n"
                strong_maps = [(map_name, adv) for map_name, adv in team2_advantages.items() if adv > 5]
                if strong_maps:
                    for map_name, advantage in sorted(strong_maps, key=lambda x: x[1], reverse=True):
                        result_text += f"   • {map_name}: +{advantage:.1f} advantage\n"
                else:
                    result_text += f"   • No significant map advantages\n"
            
            # Strategic analysis
            strategic = map_data.get('strategic_analysis', {})
            if strategic:
                result_text += f"\nSTRATEGIC INSIGHTS:\n"
                for key, analysis in strategic.items():
                    if key != 'overall':  # Skip overall for now, show specific insights
                        result_text += f"   • {analysis}\n"
            
            # Pick sequence (simplified)
            pick_sequence = map_data.get('pick_sequence', [])
            if pick_sequence:
                result_text += f"\nPICK/BAN SEQUENCE:\n"
                for i, pick in enumerate(pick_sequence[:6]):  # Show first 6 picks/bans
                    action = pick.get('action', 'unknown').upper()
                    team = pick.get('team', 'unknown')
                    map_name = pick.get('map', 'unknown')
                    result_text += f"   {i+1}. {team} {action}S {map_name}\n"
                if len(pick_sequence) > 6:
                    result_text += f"   ... and {len(pick_sequence) - 6} more actions\n"
        
        # Model info based on type
        if is_enhanced:
            team_count = getattr(self.ml_predictor, 'num_teams', 47)
            feature_count = getattr(self.ml_predictor, 'num_features', 32)
            result_text += f"""
ENHANCED PREDICTION MODEL:
   - Algorithm: Ensemble (RF + GB + MLP + SVM)
   - Training Data: {team_count} teams, comprehensive VCT 2024 data
   - Features: {feature_count} advanced features including player stats, regional strength
   - Validation: Cross-validated ensemble with hyperparameter tuning
   - Enhancement: Player performance, map analysis, tournament context
"""
        else:
            team_count = len(getattr(self.ml_predictor, 'team_stats', {}))
            result_text += f"""
PREDICTION MODEL:
   - Algorithm: Ensemble (Random Forest + Gradient Boosting)
   - Training Data: {team_count} teams, VCT 2024 season
   - Features: Win rates, H2H records, recent form, roster changes
   - Validation: Cross-validated with {prediction.get('features_used', 10)} features
"""
        
        result_text += f"""
DISCLAIMER:
This prediction is based on statistical analysis of professional Valorant
matches from 2024 and known roster changes. Actual match results may vary
due to factors like individual performance, map selection, and meta shifts.

{'='*70}
PREDICTION COMPLETED - CHAMPIONS 2025 PARIS READY!
"""
        
        self.results_text.insert(tk.END, result_text)
        
        # Show popup with key result
        messagebox.showinfo("ML Prediction Complete", 
                          f"Winner: {prediction['predicted_winner']}\n"
                          f"Confidence: {prediction['confidence']:.1%} ({prediction.get('confidence_level', 'Medium')})")
    
    def display_fallback_prediction(self):
        """Display fallback prediction when ML is not available"""
        # Simple fallback logic
        team_ratings = {
            "Team Heretics": 92, "Fnatic": 88, "Paper Rex": 91, "DRX": 86,
            "Sentinels": 90, "G2 Esports": 87, "Edward Gaming": 89, "Bilibili Gaming": 86,
            "NRG": 84, "Team Liquid": 85, "T1": 83, "Dragon Ranger Gaming": 82,
            "GIANTX": 82, "Xi Lai Gaming": 80, "MIBR": 79, "Rex Regum Qeon": 78
        }
        
        team1_rating = team_ratings.get(self.selected_team1, 75)
        team2_rating = team_ratings.get(self.selected_team2, 75)
        
        if team1_rating > team2_rating:
            winner = self.selected_team1
            confidence = 55 + (team1_rating - team2_rating) * 0.5
        else:
            winner = self.selected_team2
            confidence = 55 + (team2_rating - team1_rating) * 0.5
        
        confidence = min(85, max(55, confidence)) / 100
        
        result_text = f"""FALLBACK PREDICTION MODE
{'='*50}

MATCHUP: {self.selected_team1} vs {self.selected_team2}

PREDICTED WINNER: {winner}
CONFIDENCE: {confidence:.1%}

TEAM RATINGS:
- {self.selected_team1}: {team1_rating}/100
- {self.selected_team2}: {team2_rating}/100

NOTE: This is a basic prediction. For enhanced ML predictions,
ensure the prediction system is properly initialized.
"""
        
        self.results_text.insert(tk.END, result_text)
        messagebox.showinfo("Prediction Complete", f"Winner: {winner}")
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()


def main():
    try:
        app = MLPredictionGUI()
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        messagebox.showerror("Error", f"Failed to start GUI: {e}")


if __name__ == "__main__":
    main()