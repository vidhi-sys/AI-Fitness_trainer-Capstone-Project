# --- fitness_demo_idle.py ---
"""
AI Fitness Trainer — Capstone Project
Real-Time Computer Vision Exercise Recognition, Prediction & Muscle Analytics
Faithfully styled according to the user's reference design (Coral, Cream, Dark Teal & Pastel Cards)
"""

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import time
from collections import deque

# --- Detailed Exercise & Muscle Target Configuration ---
EXERCISES = {
    "bicep_curl": {
        "name": "Bicep Curl",
        "category": "Upper Body Strength",
        "primary_muscle": "Biceps Brachii",
        "secondary_muscles": ["Brachialis", "Brachioradialis", "Front Deltoid"],
        "left_landmarks": [11, 13, 15],   # Left Shoulder, Elbow, Wrist
        "right_landmarks": [12, 14, 16],  # Right Shoulder, Elbow, Wrist
        "angle_threshold_down": 155,      # Extension starting phase
        "angle_threshold_up": 40,         # Full contraction peak
        "cal_per_rep": 0.32,
        "instructions": [
            "1. Stand upright with feet shoulder-width apart.",
            "2. Keep your upper arm stationary near your torso.",
            "3. Curl your wrist up towards your shoulder until fully contracted.",
            "4. Pause briefly, then slowly lower down to full arm extension."
        ],
        "form_tips": [
            "💡 Keep elbow fixed in place (avoid swinging forward)",
            "💡 Squeeze biceps at the top of the motion",
            "💡 Maintain full range of motion on every rep"
        ]
    },
    "squat": {
        "name": "Barbell / Bodyweight Squat",
        "category": "Lower Body & Core",
        "primary_muscle": "Quadriceps",
        "secondary_muscles": ["Gluteus Maximus", "Hamstrings", "Calves", "Core Stabilizers"],
        "left_landmarks": [23, 25, 27],   # Left Hip, Knee, Ankle
        "right_landmarks": [24, 26, 28],  # Right Hip, Knee, Ankle
        "angle_threshold_down": 90,       # Deep squat (thighs parallel)
        "angle_threshold_up": 160,        # Standing tall extension
        "cal_per_rep": 0.45,
        "instructions": [
            "1. Stand with feet slightly wider than shoulder-width.",
            "2. Keep chest upright, core engaged, and gaze straight.",
            "3. Push hips back and bend knees until thighs are parallel to the ground.",
            "4. Drive back up through your heels to return to standing position."
        ],
        "form_tips": [
            "💡 Ensure knees track in line with your toes",
            "💡 Reach at least 90° knee angle for full depth credit",
            "💡 Keep heel firmly planted on the ground"
        ]
    }
}

# --- Helper Biomechanical Functions ---
def calculate_angle(a, b, c):
    """Calculate joint angle in degrees between three 2D keypoints [x, y]."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle

def detect_best_side(landmarks, left_indices, right_indices):
    """Detect whether left or right side keypoints have higher visibility confidence."""
    try:
        left_vis = sum(landmarks[i].visibility for i in left_indices) / len(left_indices)
        right_vis = sum(landmarks[i].visibility for i in right_indices) / len(right_indices)
        
        if right_vis > left_vis + 0.05:
            return "right", right_indices, right_vis
        return "left", left_indices, left_vis
    except:
        return "left", left_indices, 1.0

def calculate_rom_percentage(exercise_key, angle):
    """Calculate 0% to 100% repetition progress based on joint angle."""
    if exercise_key not in EXERCISES:
        return 0
    cfg = EXERCISES[exercise_key]
    if exercise_key == "bicep_curl":
        down_ang = cfg["angle_threshold_down"]
        up_ang = cfg["angle_threshold_up"]
        if angle is None: return 0
        pct = (down_ang - angle) / (down_ang - up_ang) * 100.0
        return int(np.clip(pct, 0, 100))
    elif exercise_key == "squat":
        up_ang = cfg["angle_threshold_up"]
        down_ang = cfg["angle_threshold_down"]
        if angle is None: return 0
        pct = (up_ang - angle) / (up_ang - down_ang) * 100.0
        return int(np.clip(pct, 0, 100))
    return 0

def predict_exercise_from_buffer(history_buffer):
    """
    Real-Time AI Classifier: Evaluates temporal motion buffer over 30 frames 
    and predicts exercise action (bicep_curl, squat, or idle) + confidence score.
    """
    if len(history_buffer) < 10:
        return "idle", 50
        
    elbow_angles = [snap['elbow'] for snap in history_buffer if snap['elbow'] is not None]
    knee_angles = [snap['knee'] for snap in history_buffer if snap['knee'] is not None]
    hip_ys = [snap['hip_y'] for snap in history_buffer if snap['hip_y'] is not None]
    wrist_ys = [snap['wrist_y'] for snap in history_buffer if snap['wrist_y'] is not None]
    
    if not elbow_angles or not knee_angles:
        return "idle", 50
        
    elbow_range = max(elbow_angles) - min(elbow_angles)
    knee_range = max(knee_angles) - min(knee_angles)
    hip_move = max(hip_ys) - min(hip_ys) if hip_ys else 0
    wrist_move = max(wrist_ys) - min(wrist_ys) if wrist_ys else 0
    
    if knee_range > 30 and hip_move > 0.04:
        confidence = min(99, int(75 + (knee_range / 80.0) * 24))
        return "squat", confidence
    elif elbow_range > 40 and wrist_move > 0.04 and hip_move < 0.05:
        confidence = min(99, int(75 + (elbow_range / 110.0) * 24))
        return "bicep_curl", confidence
    else:
        return "idle", 85

# --- Main Application Class ---
class FitnessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🏋️ Personal AI Fitness Trainer")
        self.root.geometry("1120x860")
        self.root.configure(bg='#FAF7F0')  # Warm Cream backdrop matching reference
        
        # Color Palette from User Reference Design
        self.CORAL_PINK = "#FF5A79"
        self.CREAM_BG = "#FAF7F0"
        self.WARM_YELLOW = "#FFF5E0"
        self.DARK_TEAL = "#1C4E4B"
        self.MINT_GREEN = "#2EC4B6"
        self.GOLD_AMBER = "#FFB800"
        self.WHITE = "#FFFFFF"
        self.TEXT_DARK = "#2D3748"
        self.TEXT_MUTED = "#718096"
        
        # Application State
        self.is_running = False
        self.cap = None
        self.counter = 0
        self.calories_burned = 0.0
        self.stage = None
        self.mode_selection = "auto_detect"
        self.active_exercise = "bicep_curl"
        self.predicted_exercise = "idle"
        self.prediction_confidence = 0
        self.current_rom = 0
        self.form_status = "READY - START REP"
        self.pose_history = deque(maxlen=30)
        
        # MediaPipe pose setup
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Setup UI
        self.setup_ui()
        self.update_exercise_details()
        
    def setup_ui(self):
        # 1. Header Frame (Coral Pink Header matching Reference 1)
        header_frame = tk.Frame(self.root, bg=self.CORAL_PINK, pady=14, padx=24)
        header_frame.pack(fill='x', side='top')
        
        title_box = tk.Frame(header_frame, bg=self.CORAL_PINK)
        title_box.pack(side='left')
        
        greeting_label = tk.Label(title_box, text="Hi, User 👋", 
                                  font=('Segoe UI', 11, 'bold'), bg=self.CORAL_PINK, fg='#FFE5EC')
        greeting_label.pack(anchor='w')
        
        main_title = tk.Label(title_box, text="Personal AI Workout Trainer", 
                              font=('Segoe UI', 20, 'bold'), bg=self.CORAL_PINK, fg=self.WHITE)
        main_title.pack(anchor='w')
        
        badge_frame = tk.Frame(header_frame, bg=self.CORAL_PINK)
        badge_frame.pack(side='right')
        
        self.ai_mode_badge = tk.Label(badge_frame, text=" 🤖 Auto AI Mode ", 
                                      font=('Segoe UI', 10, 'bold'), bg='#1C1C1C', fg=self.WHITE, padx=14, pady=6)
        self.ai_mode_badge.pack(side='right')

        # 2. Main Content Container (Warm Cream Background)
        main_container = tk.Frame(self.root, bg=self.CREAM_BG)
        main_container.pack(fill='both', expand=True, padx=16, pady=14)
        
        # --- LEFT COLUMN: Video Stream Viewport ---
        left_column = tk.Frame(main_container, bg=self.WHITE, highlightbackground='#EFECE6', highlightthickness=1)
        left_column.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        viewport_header = tk.Frame(left_column, bg=self.WHITE, pady=10, padx=14)
        viewport_header.pack(fill='x')
        
        vp_title = tk.Label(viewport_header, text="📹 Live Camera Tracking", 
                            font=('Segoe UI', 12, 'bold'), bg=self.WHITE, fg=self.DARK_TEAL)
        vp_title.pack(side='left')
        
        self.side_badge = tk.Label(viewport_header, text="SIDE: AUTO DETECT", 
                                   font=('Segoe UI', 9, 'bold'), bg=self.WARM_YELLOW, fg=self.DARK_TEAL, padx=10, pady=3)
        self.side_badge.pack(side='right')
        
        # Camera Frame Viewport
        self.video_frame = tk.Frame(left_column, bg='#142826', width=640, height=480)
        self.video_frame.pack(fill='both', expand=True, padx=10, pady=5)
        self.video_frame.pack_propagate(False)
        
        self.video_label = tk.Label(self.video_frame, bg='#142826')
        self.video_label.pack(expand=True, fill='both')
        
        self.placeholder_text = tk.Label(self.video_frame, 
                                         text="▶ Click 'Start Webcam' below to test AI Exercise Prediction",
                                         font=('Segoe UI', 13, 'bold'), bg='#142826', fg='#A0AEC0')
        self.placeholder_text.place(relx=0.5, rely=0.5, anchor='center')
        
        # --- RIGHT COLUMN: Design Cards matching Reference Images ---
        right_column = tk.Frame(main_container, bg=self.CREAM_BG, width=390)
        right_column.pack(side='right', fill='both', padx=(5, 0))
        
        # CARD 1: Learning Progress / Muscle Target Card (Warm Yellow Background like Reference 1)
        muscle_card = tk.Frame(right_column, bg=self.WARM_YELLOW, bd=0, padx=16, pady=14)
        muscle_card.pack(fill='x', pady=(0, 12))
        
        card1_top = tk.Frame(muscle_card, bg=self.WARM_YELLOW)
        card1_top.pack(fill='x', pady=(0, 6))
        
        card1_title = tk.Label(card1_top, text="Exercise Progress", 
                               font=('Segoe UI', 13, 'bold'), bg=self.WARM_YELLOW, fg=self.TEXT_DARK)
        card1_title.pack(side='left')
        
        card1_badge = tk.Label(card1_top, text="Active", 
                               font=('Segoe UI', 9, 'bold'), bg=self.WHITE, fg=self.MINT_GREEN, padx=8, pady=2)
        card1_badge.pack(side='right')
        
        self.prediction_badge = tk.Label(muscle_card, text="AI PREDICTION: IDLE", 
                                         font=('Segoe UI', 10, 'bold'), bg=self.WHITE, fg=self.CORAL_PINK, 
                                         padx=10, pady=5)
        self.prediction_badge.pack(fill='x', pady=(4, 10))
        
        p_muscle_lbl = tk.Label(muscle_card, text="PRIMARY TARGET MUSCLE", 
                                font=('Segoe UI', 8, 'bold'), bg=self.WARM_YELLOW, fg=self.TEXT_MUTED)
        p_muscle_lbl.pack(anchor='w')
        
        self.primary_muscle_badge = tk.Label(muscle_card, text="Biceps Brachii", 
                                             font=('Segoe UI', 13, 'bold'), bg=self.CORAL_PINK, fg=self.WHITE, 
                                             padx=14, pady=6)
        self.primary_muscle_badge.pack(anchor='w', pady=(4, 10))
        
        s_muscle_lbl = tk.Label(muscle_card, text="SECONDARY MUSCLES", 
                                font=('Segoe UI', 8, 'bold'), bg=self.WARM_YELLOW, fg=self.TEXT_MUTED)
        s_muscle_lbl.pack(anchor='w')
        
        self.secondary_muscles_label = tk.Label(muscle_card, text="Brachialis • Forearms • Deltoid", 
                                                font=('Segoe UI', 9, 'bold'), bg=self.WHITE, fg=self.DARK_TEAL, 
                                                padx=10, pady=5, wraplength=330, justify='left')
        self.secondary_muscles_label.pack(anchor='w', pady=(4, 2))
        
        # CARD 2: Repetitions & ROM Card (Dark Teal Accent Card like Reference 1)
        metrics_card = tk.Frame(right_column, bg=self.DARK_TEAL, padx=16, pady=14)
        metrics_card.pack(fill='x', pady=(0, 12))
        
        metrics_title = tk.Label(metrics_card, text="Workout Metrics", 
                                 font=('Segoe UI', 12, 'bold'), bg=self.DARK_TEAL, fg=self.WHITE)
        metrics_title.pack(anchor='w', pady=(0, 8))
        
        metrics_grid = tk.Frame(metrics_card, bg=self.DARK_TEAL)
        metrics_grid.pack(fill='x')
        
        # Rep Counter Pill Box (Coral Pink Pill matching Ref #1 Leaderboard)
        rep_box = tk.Frame(metrics_grid, bg=self.CORAL_PINK, padx=12, pady=10)
        rep_box.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        
        tk.Label(rep_box, text="REPETITIONS", font=('Segoe UI', 8, 'bold'), bg=self.CORAL_PINK, fg='#FFE5EC').pack()
        self.rep_count_val = tk.Label(rep_box, text="0", font=('Segoe UI', 24, 'bold'), bg=self.CORAL_PINK, fg=self.WHITE)
        self.rep_count_val.pack()
        
        # Calories Box (Amber Pill matching Ref #1 Leaderboard)
        cal_box = tk.Frame(metrics_grid, bg=self.GOLD_AMBER, padx=12, pady=10)
        cal_box.grid(row=0, column=1, sticky='nsew', padx=(5, 0))
        
        tk.Label(cal_box, text="EST. CALORIES", font=('Segoe UI', 8, 'bold'), bg=self.GOLD_AMBER, fg='#5C4300').pack()
        self.cal_count_val = tk.Label(cal_box, text="0.0 kcal", font=('Segoe UI', 18, 'bold'), bg=self.GOLD_AMBER, fg=self.TEXT_DARK)
        self.cal_count_val.pack(pady=(4, 0))
        
        metrics_grid.columnconfigure(0, weight=1)
        metrics_grid.columnconfigure(1, weight=1)
        
        # ROM Progress Bar
        rom_frame = tk.Frame(metrics_card, bg=self.DARK_TEAL, pady=8)
        rom_frame.pack(fill='x')
        
        rom_lbl_row = tk.Frame(rom_frame, bg=self.DARK_TEAL)
        rom_lbl_row.pack(fill='x')
        
        self.angle_display_lbl = tk.Label(rom_lbl_row, text="Joint Angle: --°", font=('Segoe UI', 10, 'bold'), bg=self.DARK_TEAL, fg=self.WHITE)
        self.angle_display_lbl.pack(side='left')
        
        self.rom_pct_lbl = tk.Label(rom_lbl_row, text="ROM: 0%", font=('Segoe UI', 10, 'bold'), bg=self.DARK_TEAL, fg=self.MINT_GREEN)
        self.rom_pct_lbl.pack(side='right')
        
        style = ttk.Style()
        style.theme_use('default')
        style.configure("Fitness.Horizontal.TProgressbar", thickness=12, troughcolor='#123331', background=self.MINT_GREEN)
        self.rom_progress = ttk.Progressbar(rom_frame, style="Fitness.Horizontal.TProgressbar", 
                                            orient="horizontal", mode="determinate", length=320)
        self.rom_progress.pack(fill='x', pady=4)
        
        # Form Status Banner
        self.form_status_badge = tk.Label(metrics_card, text="STATUS: READY", 
                                          font=('Segoe UI', 10, 'bold'), bg=self.WHITE, fg=self.DARK_TEAL, pady=6)
        self.form_status_badge.pack(fill='x', pady=(6, 0))
        
        # CARD 3: Coaching Tips
        coaching_card = tk.Frame(right_column, bg=self.WHITE, padx=16, pady=12, highlightbackground='#EFECE6', highlightthickness=1)
        coaching_card.pack(fill='both', expand=True)
        
        tk.Label(coaching_card, text="💡 Coaching Tips", font=('Segoe UI', 11, 'bold'), bg=self.WHITE, fg=self.TEXT_DARK).pack(anchor='w')
        
        self.tips_label = tk.Label(coaching_card, text="", font=('Segoe UI', 9), 
                                   bg=self.WHITE, fg=self.TEXT_MUTED, justify='left', anchor='nw', wraplength=340)
        self.tips_label.pack(fill='both', expand=True, pady=(4, 0))

        # 3. Bottom Control Toolbar
        control_bar = tk.Frame(self.root, bg=self.WHITE, pady=12, padx=20, highlightbackground='#EFECE6', highlightthickness=1)
        control_bar.pack(fill='x', side='bottom')
        
        select_lbl = tk.Label(control_bar, text="SELECT MODE:", font=('Segoe UI', 10, 'bold'), bg=self.WHITE, fg=self.TEXT_DARK)
        select_lbl.pack(side='left', padx=(0, 8))
        
        self.mode_var = tk.StringVar(value="auto_detect")
        self.exercise_combo = ttk.Combobox(control_bar, textvariable=self.mode_var, 
                                           values=["auto_detect", "bicep_curl", "squat"], state="readonly", width=22, font=('Segoe UI', 10))
        self.exercise_combo.pack(side='left', padx=(0, 15))
        self.exercise_combo.bind('<<ComboboxSelected>>', self.on_mode_change)
        
        # Action Buttons matching Reference Design Pills
        self.start_btn = tk.Button(control_bar, text="▶ Start Webcam", command=self.start_webcam, 
                                   bg=self.DARK_TEAL, fg=self.WHITE, font=('Segoe UI', 10, 'bold'), 
                                   activebackground='#153D3B', activeforeground=self.WHITE, padx=18, pady=6, bd=0, cursor='hand2')
        self.start_btn.pack(side='left', padx=5)
        
        self.stop_btn = tk.Button(control_bar, text="⏹ Stop", command=self.stop_webcam, 
                                  bg=self.CORAL_PINK, fg=self.WHITE, font=('Segoe UI', 10, 'bold'), 
                                  activebackground='#E04362', activeforeground=self.WHITE, padx=18, pady=6, bd=0, state='disabled', cursor='hand2')
        self.stop_btn.pack(side='left', padx=5)
        
        self.reset_btn = tk.Button(control_bar, text="🔄 Reset", command=self.reset_counter, 
                                   bg=self.GOLD_AMBER, fg=self.TEXT_DARK, font=('Segoe UI', 10, 'bold'), 
                                   activebackground='#E5A600', activeforeground=self.TEXT_DARK, padx=18, pady=6, bd=0, cursor='hand2')
        self.reset_btn.pack(side='left', padx=5)

    def update_exercise_details(self):
        """Update UI fields when active exercise updates."""
        ex_key = self.active_exercise
        cfg = EXERCISES[ex_key]
        
        # Primary & Secondary muscles
        self.primary_muscle_badge.config(text=f"🎯 {cfg['primary_muscle']}")
        self.secondary_muscles_label.config(text=" • ".join(cfg['secondary_muscles']))
        
        # Coaching tips
        tips_text = "\n\n".join(cfg['form_tips'])
        self.tips_label.config(text=tips_text)

    def on_mode_change(self, event=None):
        self.mode_selection = self.mode_var.get()
        if self.mode_selection != "auto_detect":
            self.active_exercise = self.mode_selection
            self.ai_mode_badge.config(text=f" Manual: {EXERCISES[self.active_exercise]['name']} ", bg=self.DARK_TEAL)
        else:
            self.ai_mode_badge.config(text=" 🤖 Auto AI Mode ", bg='#1C1C1C')
        self.reset_counter()
        self.update_exercise_details()

    def reset_counter(self):
        self.counter = 0
        self.calories_burned = 0.0
        self.stage = None
        self.current_rom = 0
        self.pose_history.clear()
        self.rep_count_val.config(text="0")
        self.cal_count_val.config(text="0.0 kcal")
        self.rom_progress['value'] = 0
        self.rom_pct_lbl.config(text="ROM: 0%")
        self.form_status_badge.config(text="STATUS: READY", bg=self.WHITE, fg=self.DARK_TEAL)

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.form_status_badge.config(text="❌ WEBCAM ERROR", bg='#FFE5EC', fg=self.CORAL_PINK)
            return
            
        self.is_running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        if hasattr(self, 'placeholder_text') and self.placeholder_text:
            self.placeholder_text.destroy()
        self.form_status_badge.config(text="STATUS: TRACKING ACTIVE", bg='#E6FFFA', fg=self.DARK_TEAL)
        self.update_frame()

    def stop_webcam(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.form_status_badge.config(text="STATUS: STOPPED", bg='#EDF2F7', fg=self.TEXT_MUTED)

    def update_frame(self):
        if not self.is_running or self.cap is None:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.form_status_badge.config(text="❌ FRAME ERROR", bg='#FFE5EC', fg=self.CORAL_PINK)
            self.stop_webcam()
            return
            
        # Process Video Frame with MediaPipe & AI Classification
        processed_frame, rep_inc, angle, current_stage, rom_pct, form_msg, side_name, pred_name, conf = self.process_frame(frame)
        
        # Update AI Prediction Badge
        if pred_name == "idle":
            self.prediction_badge.config(text="AI PREDICTION: IDLE / STANDING", bg=self.WHITE, fg=self.CORAL_PINK)
        else:
            self.prediction_badge.config(text=f"AI PREDICTED: {EXERCISES[pred_name]['name'].upper()} ({conf}% CONF)", 
                                         bg=self.CORAL_PINK, fg=self.WHITE)
            if self.mode_selection == "auto_detect" and pred_name != self.active_exercise:
                self.active_exercise = pred_name
                self.update_exercise_details()
                self.stage = None

        # Update reps & calories
        if rep_inc == 1 and current_stage != self.stage:
            self.counter += 1
            self.calories_burned += EXERCISES[self.active_exercise]["cal_per_rep"]
            self.stage = current_stage
            self.rep_count_val.config(text=str(self.counter))
            self.cal_count_val.config(text=f"{self.calories_burned:.1f} kcal")
            self.form_status_badge.config(text="🟢 PERFECT FORM — REP COMPLETED!", bg='#E6FFFA', fg=self.DARK_TEAL)
        elif rep_inc == 0:
            self.stage = current_stage
            if form_msg:
                self.form_status_badge.config(text=form_msg, 
                                              bg=self.WARM_YELLOW if 'DEEPER' in form_msg or 'HIGHER' in form_msg else self.WHITE,
                                              fg=self.TEXT_DARK if 'DEEPER' in form_msg or 'HIGHER' in form_msg else self.DARK_TEAL)

        # ROM progress bar update
        if rom_pct is not None:
            self.rom_progress['value'] = rom_pct
            self.rom_pct_lbl.config(text=f"ROM: {rom_pct}%")
            
        # Angle display
        if angle is not None:
            self.angle_display_lbl.config(text=f"Joint Angle: {int(angle)}°")
        else:
            self.angle_display_lbl.config(text="Joint Angle: --°")
            
        # Side badge
        self.side_badge.config(text=f"DETECTED: {side_name.upper()}")

        # Convert frame for Tkinter Viewport
        processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(processed_rgb)
        img_tk = ImageTk.PhotoImage(image=img)
        
        self.video_label.config(image=img_tk)
        self.video_label.image = img_tk
        
        self.root.after(30, self.update_frame)

    def process_frame(self, frame):
        frame = cv2.resize(frame, (640, 480))
        h, w, _ = frame.shape
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            results = pose.process(image_rgb)
            
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        rep_inc = 0
        current_stage = self.stage
        angle = None
        rom_pct = 0
        form_msg = ""
        active_side_name = "LEFT"
        pred_exercise = "idle"
        conf = 50
        
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image_bgr,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
            
            landmarks = results.pose_landmarks.landmark
            
            l_shoulder, l_elbow, l_wrist = landmarks[11], landmarks[13], landmarks[15]
            l_hip, l_knee, l_ankle = landmarks[23], landmarks[25], landmarks[27]
            
            l_elbow_ang = calculate_angle([l_shoulder.x*w, l_shoulder.y*h], [l_elbow.x*w, l_elbow.y*h], [l_wrist.x*w, l_wrist.y*h])
            l_knee_ang = calculate_angle([l_hip.x*w, l_hip.y*h], [l_knee.x*w, l_knee.y*h], [l_ankle.x*w, l_ankle.y*h])
            
            self.pose_history.append({
                'elbow': l_elbow_ang,
                'knee': l_knee_ang,
                'hip_y': l_hip.y,
                'wrist_y': l_wrist.y
            })
            
            pred_exercise, conf = predict_exercise_from_buffer(self.pose_history)
            
            ex_key = self.active_exercise
            cfg = EXERCISES[ex_key]
            
            side, target_indices, vis_score = detect_best_side(landmarks, cfg["left_landmarks"], cfg["right_landmarks"])
            active_side_name = f"{side.upper()} {'ARM' if ex_key == 'bicep_curl' else 'LEG'}"
            
            try:
                pts = [[landmarks[i].x * w, landmarks[i].y * h] for i in target_indices]
                angle = calculate_angle(pts[0], pts[1], pts[2])
                rom_pct = calculate_rom_percentage(ex_key, angle)
                
                pt1, pt2, pt3 = (int(pts[0][0]), int(pts[0][1])), (int(pts[1][0]), int(pts[1][1])), (int(pts[2][0]), int(pts[2][1]))
                # Coral Pink & Mint highlights matching reference design
                cv2.line(image_bgr, pt1, pt2, (121, 90, 255), 4)
                cv2.line(image_bgr, pt2, pt3, (121, 90, 255), 4)
                cv2.circle(image_bgr, pt1, 8, (255, 255, 255), -1)
                cv2.circle(image_bgr, pt2, 8, (182, 196, 46), -1)  # Mint joint
                cv2.circle(image_bgr, pt3, 8, (255, 255, 255), -1)
                
                cv2.putText(image_bgr, f"{int(angle)} deg", (pt2[0] + 15, pt2[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if ex_key == "bicep_curl":
                    if angle > cfg["angle_threshold_down"]:
                        current_stage = "down"
                        form_msg = "READY — CURL UP"
                    elif angle < cfg["angle_threshold_up"] and self.stage == "down":
                        current_stage = "up"
                        rep_inc = 1
                        form_msg = "🟢 PERFECT CURL!"
                    elif self.stage == "down" and 40 <= angle <= 110:
                        form_msg = f"🟡 CURL HIGHER (ROM: {rom_pct}%)"
                        
                elif ex_key == "squat":
                    if angle > cfg["angle_threshold_up"]:
                        current_stage = "up"
                        form_msg = "READY — LOWER DOWN"
                    elif angle <= cfg["angle_threshold_down"] and self.stage == "up":
                        current_stage = "down"
                        rep_inc = 1
                        form_msg = "🟢 DEEP SQUAT ACHIEVED!"
                    elif self.stage == "up" and 90 < angle <= 140:
                        form_msg = f"🟡 SQUAT DEEPER (ROM: {rom_pct}%)"
                        
            except Exception as e:
                pass
        else:
            cfg = EXERCISES[self.active_exercise]

        # In-Video Visual Overlay Banners
        cv2.rectangle(image_bgr, (10, 10), (420, 80), (28, 78, 75), -1)
        pred_label_txt = f"AI PREDICTED: {pred_exercise.upper()} ({conf}%)" if pred_exercise != "idle" else "AI PREDICTED: IDLE"
        cv2.putText(image_bgr, pred_label_txt, (20, 34), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (182, 196, 46) if pred_exercise != "idle" else (0, 184, 255), 2)
        
        cv2.putText(image_bgr, f"TRACKING: {cfg['name']} ({cfg['primary_muscle']})", (20, 64), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Bottom Status Banner
        cv2.rectangle(image_bgr, (0, 440), (640, 480), (20, 40, 38), -1)
        status_txt = form_msg if form_msg else f"ACTIVE SIDE: {active_side_name}"
        cv2.putText(image_bgr, status_txt, (20, 468), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (182, 196, 46), 2)
        
        return image_bgr, rep_inc, angle, current_stage, rom_pct, form_msg, active_side_name, pred_exercise, conf

    def on_closing(self):
        self.stop_webcam()
        self.root.destroy()

# --- Application Entry Point ---
if __name__ == "__main__":
    root = tk.Tk()
    app = FitnessApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
