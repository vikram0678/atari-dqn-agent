# METHODOLOGY — Atari DQN Agent
 
## 1. Overview
 
This project implements **Deep Q-Network (DQN)** to train an AI agent to play Atari Pong from raw pixel inputs. The agent learns by playing thousands of games and improving its strategy through trial and error — a process called Reinforcement Learning.
 
**Final Result:** Average reward of **17.07** over 100 episodes on ALE/Pong-v5 (target was 10.0).
 
---
 
## 2. How Reinforcement Learning Works
 
```
Agent sees game screen
        ↓
Agent picks an action (move paddle up/down/stay)
        ↓
Game returns reward (+1 win point, -1 lose point)
        ↓
Agent learns from this experience
        ↓
Repeat 1,600,000+ times → Agent becomes expert
```
 
---
 
## 3. Environment Preprocessing
 
Raw Atari frames are 210×160 RGB images. We preprocess them to make training faster and more stable.
 
### Steps Applied
 
| Step | Input | Output | Reason |
|------|-------|--------|--------|
| Grayscale | (210,160,3) RGB | (210,160) gray | Remove color redundancy |
| Resize | (210,160) | (84,84) | Reduce computation |
| Normalize | uint8 [0,255] | float32 [0,1] | Stable gradients |
| Frame Stack | (84,84) | (4,84,84) | Capture motion |
 
### Frame Skipping
- Agent repeats each action for 4 frames
- Max pool over last 2 frames removes flickering
- Reduces computation by 4x
 
### Reward Clipping
- All rewards clipped to `[-1, +1]`
- Prevents large gradient updates
- Makes learning stable across different games
 
---
 
## 4. Neural Network Architecture (CNN)
 
```
Input: (batch, 4, 84, 84)  ← 4 stacked grayscale frames
 
Conv2d(4 → 32,  kernel=8, stride=4)  → (32, 20, 20)  + ReLU
Conv2d(32 → 64, kernel=4, stride=2)  → (64,  9,  9)  + ReLU
Conv2d(64 → 64, kernel=3, stride=1)  → (64,  7,  7)  + ReLU
 
Flatten → 3136 neurons
 
Linear(3136 → 512)  + ReLU
Linear(512  → 6)    ← Q-value for each action
 
Output: (batch, 6)  ← Q-value per action
```
 
**Why This Architecture:**
- Same as original DeepMind DQN paper
- 3 conv layers extract spatial features from frames
- Dense layers map features to action values
- 6 outputs = 6 possible Pong actions
 
---
 
## 5. DQN Key Components
 
### 5.1 Experience Replay Buffer
 
Instead of learning from each frame immediately, we store experiences and sample randomly:
 
```
Buffer stores: (state, action, reward, next_state, done)
Capacity: 50,000 to 100,000 transitions
Batch size: 32 random transitions per update
```
 
**Why it works:**
- Breaks correlation between consecutive frames
- Same experience can be learned from multiple times
- More stable and efficient training
 
### 5.2 Target Network
 
Two identical networks:
- **Main Network** — updated every step (learns)
- **Target Network** — updated every 1,000 steps (stable reference)
 
**Why it works:**
- Without target network: agent chases a moving target
- With target network: stable Q-value targets for 1,000 steps
- Prevents oscillation and divergence
 
### 5.3 Bellman Equation (TD Target)
 
```
Q_target = reward + γ × max(Q_target_network(next_state))
 
where γ = 0.99 (discount factor)
```
 
The agent learns to predict this target accurately.
 
### 5.4 Epsilon-Greedy Exploration
 
```
Start: ε = 1.0  → 100% random actions (explore)
End:   ε = 0.05 → 5% random, 95% learned policy (exploit)
Decay: ε × 0.995 per episode
```
 
This ensures the agent explores early and exploits later.
 
---
 
## 6. Loss Function
 
**Huber Loss (Smooth L1):**
```
L(δ) = 0.5 × δ²     if |δ| ≤ 1
       |δ| - 0.5     otherwise
```
 
**Why Huber over MSE:**
- MSE amplifies large errors → unstable training
- Huber is quadratic for small errors (precise)
- Huber is linear for large errors (robust)
- Prevents exploding gradients
 
---
 
## 7. Hyperparameters
 
| Parameter | Value | Reason |
|-----------|-------|--------|
| Learning rate | 0.0001 | Small = stable RL training |
| Discount γ | 0.99 | Values rewards ~100 steps ahead |
| Replay buffer | 50,000-100,000 | Diverse experience samples |
| Batch size | 32 | Standard DQN batch |
| Target update | every 1,000 steps | Balance stability vs freshness |
| ε start | 1.0 | Full random exploration |
| ε end | 0.05 | 5% exploration maintained |
| ε decay | 0.995/episode | ~300 episodes to reach minimum |
| Min replay size | 5,000-10,000 | Fill buffer before training |
| Gradient clip | 10.0 | Prevent exploding gradients |
| Frame skip | 4 | Standard Atari setting |
 
---
 
## 8. Training Process
 
### Training Timeline
 
| Phase | Episodes | Avg Reward | Description |
|-------|----------|------------|-------------|
| Random | 1-300 | -21 to -18 | Agent explores randomly |
| Early Learning | 300-700 | -18 to -10 | Agent starts learning patterns |
| Rapid Improvement | 700-1200 | -10 to +5 | Big jumps in performance |
| Mastery | 1200-1600 | +5 to +17 | Agent consistently wins |
 
### Training Stability Measures
1. **Reward clipping** → consistent gradients
2. **Target network** → stable TD targets
3. **Experience replay** → breaks temporal correlation
4. **Huber loss** → robust to large errors
5. **Gradient clipping** → prevents exploding gradients
6. **Warm-up phase** → buffer fills before training starts
 
---
 
## 9. Training Results
 
### Key Milestones
 
```
Episode  300 → Avg: -18.0  (random policy)
Episode  700 → Avg: -12.0  (learning paddle control)
Episode 1000 → Avg:  -5.0  (winning some points)
Episode 1200 → Avg:  +5.0  (winning most points)
Episode 1400 → Avg: +10.0  (target achieved!)
Episode 1600 → Avg: +17.0  (near-perfect play)
 
Final evaluation (100 episodes):
  Average Reward : 17.07
  Std Dev        : 2.90
  Min Reward     : 5.0
  Max Reward     : 21.0 (perfect game!)
```
 
### What The Numbers Mean
```
Pong score range: -21 (lose all) to +21 (win all)
Random agent:     -21 (loses everything)
Our agent:        +17 (wins most points) ✅
```
 
---
 
## 10. MLOps Pipeline
 
### Experiment Tracking (TensorBoard)
Logs per episode:
- Total reward
- Average Q-value
- Huber loss
- Epsilon value
- Episode duration
 
### Model Checkpointing
- Saved every 100 episodes
- Best model saved when avg reward improves
- Latest model always saved for resume
 
### Resume Training
- Automatic checkpoint detection
- Loads weights, epsilon, step count
- Continues from last saved episode
 
### Early Stopping
- Triggers when avg reward ≥ 18.0
- Prevents unnecessary computation
 
---
 
## 11. Inference API
 
Trained model served via FastAPI:
 
```
POST /predict
Input:  { "state": (4, 84, 84) float array }
Output: { "action": int, "q_values": [float] }
 
Latency: < 5ms on CPU
```
 
---
 
## 12. Containerization
 
### Dockerfile.train
- Full training environment
- PyTorch with CUDA support
- All training dependencies
- ~14 GB image (includes torch)
 
### Dockerfile.inference
- Lightweight serving environment
- PyTorch CPU only (~2.4 GB)
- Only inference dependencies
- Faster startup, smaller footprint
 
---
 
## 13. Key Design Decisions
 
| Decision | Choice | Alternative | Reason |
|----------|--------|-------------|--------|
| Framework | PyTorch | TensorFlow | More flexible for RL |
| Loss | Huber | MSE | Robust to outliers |
| Optimizer | Adam | SGD | Adaptive learning rate |
| Buffer | deque | numpy array | Auto-removes old data |
| API | FastAPI | Flask | Faster, async support |
 
---
 
