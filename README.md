# AI-Powered Fitness Trainer with Automatic Exercise Recognition and Repetition Counting

## Problem Statement

Engaging in regular physical exercise without certified professional supervision often leads to improper form, inaccurate tracking of workout metrics, and an increased likelihood of musculoskeletal injuries. Conventional fitness applications predominantly depend on manual user logging, a process that is both error-prone and disruptive during workout sessions. Consequently, there is a distinct need for an automated, accessible, and computer-vision-based fitness monitoring system capable of accurately analyzing human biomechanics in real time without necessitating wearable sensors or specialized hardware.

This project addresses these challenges by developing an AI-driven personal fitness evaluation system. Leveraging state-of-the-art computer vision and deep learning techniques, the application captures video from a standard webcam, extracts body landmarks, classifies exercise actions, and counts repetitions in real time.

---

## Objectives

1. **Automated Exercise Recognition**: Accurately classify physical exercises (including push-ups, squats, bicep curls, and shoulder presses) from temporal landmark sequences utilizing a Bidirectional Long Short-Term Memory (BiLSTM) network.
2. **Real-Time Repetition Counting**: Dynamically track repetitions through geometric joint-angle estimation derived from coordinate landmark positions, eliminating the need for manual logging.
3. **Low-Latency Feedback**: Process live video streams and display instantaneous visual feedback, metric updates, and posture status overlays on user interfaces.
4. **Offline Video Analysis**: Support recorded video uploads for retrospective performance assessment and repetition verification.
5. **Interactive Fitness Advisory**: Integrate a conversational assistant to answer exercise-related queries and provide contextual fitness guidance.

---

## Scope of Contribution and Planned Enhancements

While the baseline framework is grounded in research on real-time exercise classification (*arXiv:2411.11548*), this capstone initiative focuses on extending and standardizing the system through the following enhancements:

### 1. Exercise Analytics & Target Muscle Intelligence
- **Bicep Curl**: Tracks *Biceps Brachii* (Primary Target) alongside *Brachialis*, *Brachioradialis (Forearms)*, and *Anterior Deltoid*. Features real-time elbow joint angle tracking and range-of-motion (ROM) progress percentage.
- **Squat**: Tracks *Quadriceps* (Primary Target) along with *Gluteus Maximus*, *Hamstrings*, *Calves*, and *Core Stabilizers*. Monitors knee flexion angle and validates parallel squat depth (<= 90°).

### 2. Biomechanical Form & Multi-Side Auto Detection
- **Dynamic Side Tracking**: Automatically detects whether the left or right side of the body is more visible to the camera stream to maintain keypoint tracking accuracy.
- **Real-Time Form Feedback Banner**: Live warnings and cues overlaid directly on the UI (e.g. `🟢 PERFECT FORM — REP COMPLETED!`, `🟡 SQUAT DEEPER`, `🟡 CURL HIGHER`).
- **Calorie Estimation**: Calculates session-level energy expenditure based on per-exercise repetition MET constants.

### 3. UI Dashboard
- Inspired by modern fitness mobile & web applications, featuring muscle targeting pill badges, live ROM progress meters, repetition stat counters, and coaching instruction cards.

---

## Technical Stack

| Component | Framework / Library | Role |
|---|---|---|
| Pose Estimation | MediaPipe Pose | Real-time 33-point skeletal landmark extraction |
| Temporal Classification | TensorFlow / Keras | Bidirectional LSTM neural network architecture |
| Application Interface | Tkinter | Sleek UI for webcam feed, real-time counters, muscle cards, and analytics |
| Computer Vision | OpenCV | Frame manipulation, angle visualization, and video streaming |
| Conversational Assistant | OpenAI API / LangChain | Fitness knowledge querying and user interaction |
| Core Language | Python 3.9+ | Backend pipeline and algorithmic execution |

---

## Getting Started & Running the Demo

The interactive prototype application is located in the [`demo/`](demo/) directory ([`fitness_demo_idle.py`](demo/fitness_demo_idle.py)). It provides a real-time desktop interface for webcam-based pose landmark tracking, joint-angle calculation, and automated repetition counting (supporting bicep curls and squats).

### Prerequisites

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

*(Note: `tkinter` is included by default with standard Python distributions on Windows, macOS, and Linux).*

### Execution Commands

You can run the application either from inside the demo directory or directly from the project root:

**Option 1: Run from the demo directory**
```bash
# Navigate to the demo folder (e.g., cd path/to/demo)
cd demo

# Launch the application
python3 fitness_demo_idle.py
```

**Option 2: Run directly from the project root**
```bash
python3 demo/fitness_demo_idle.py
```

---

## Research Foundation & Literature Survey

The architecture and methodology of this project are informed by peer-reviewed literature and key research papers curated in the [`RESEARCH/`](RESEARCH/) directory:

| Paper / Document | Focus Area & Relevance to Project | File Path |
|---|---|---|
| **BlazePose: On-device Real-time Body Pose Tracking** | Details the single-person 33-keypoint 3D landmark topology and lightweight inference pipeline powering the MediaPipe Pose backend. | [`BlazePose On-device Real-time Body Pose tracking.pdf`](RESEARCH/BlazePose%20On-device%20Real-time%20Body%20Pose%20tracking.pdf) |
| **Real-Time Fitness Exercise Classification and Counting from Video Frames** | Core baseline paper outlining sequential landmark extraction, BiLSTM neural network classification, and geometric state-machine repetition counting. | [`Real-Time Fitness Exercise Classification and.pdf`](RESEARCH/Real-Time%20Fitness%20Exercise%20Classification%20and.pdf) |
| **Muscle Vision: Real Time Keypoint Based Pose Classification of Physical Exercises** | Investigates skeletal keypoint geometry, kinematic angle calculation, and pose classification across standard calisthenics exercises. | [`Muscle Vision Real Time Keypoint Based Pose Classification of Physical Exercises.pdf`](RESEARCH/Muscle%20Vision%20Real%20Time%20Keypoint%20Based%20Pose%20Classification%20of%20Physical%20Exercises.pdf) |
| **Recognizing Exercises and Counting Repetitions in Real Time** | Analyzes continuous repetition segmentation, peak-valley cycle detection, and repetition validation algorithms under varying user camera perspectives. | [`Recognizing Exercises and Counting Repetitions in real time.pdf`](RESEARCH/Recognizing%20Exercises%20and%20Counting%20Repetitions%20in%20real%20time.pdf) |
| **Workout Classification Using a Convolutional Neural Network in Ensemble Learning** | Explores spatio-temporal feature extraction, CNN models, and ensemble methods for workout motion classification, informing model benchmark comparisons. | [`Workout Classification Using a Convolutional Neural Network in Ensemble Learning.pdf`](RESEARCH/Workout%20Classification%20Using%20a%20Convolutional%20Neural%20Network%20in%20Ensemble%20Learning.pdf) |

---

## Repository Structure

```text
AI-Fitness_trainer-Capstone-Project/
├── LICENSE
├── README.md
├── requirements.txt
├── demo/
│   └── fitness_demo_idle.py
└── RESEARCH/
    ├── BlazePose On-device Real-time Body Pose tracking.pdf
    ├── Muscle Vision Real Time Keypoint Based Pose Classification of Physical Exercises.pdf
    ├── Real-Time Fitness Exercise Classification and.pdf
    ├── Recognizing Exercises and Counting Repetitions in real time.pdf
    └── Workout Classification Using a Convolutional Neural Network in Ensemble Learning.pdf
```

---

## References

1. Bazrev, V., et al. (2020). *BlazePose: On-device Real-time Body Pose tracking.* arXiv:2006.10204.
2. Riccio, R. (2024). *Real-Time Fitness Exercise Classification and Counting from Video Frames.* arXiv:2411.11548.
3. Lugaresi, C., et al. (2019). *MediaPipe: A Framework for Building Perception Pipelines.* arXiv:1906.08172.
4. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735-1780.
5. Schuster, M., & Paliwal, K. K. (1997). *Bidirectional Recurrent Neural Networks.* IEEE Transactions on Signal Processing, 45(11), 2673-2681.

