🎮 Atari DQN Agent
A Deep Q-Network (DQN) agent trained to play Atari Pong from raw pixels using PyTorch. Achieves average reward of 16.81 on Pong-v5 over 100 episodes. Includes Docker containerization, FastAPI inference endpoint, and TensorBoard experiment tracking.

📊 Results





























MetricValueGameALE/Pong-v5Average Reward (100 eps)16.81 ✅Training Episodes1600Best Reward21.0Target Requirement10.0

📁 Project Structure
atari-dqn-agent/
├── src/
│   ├── agent/           ← DQN agent, Q-network, exploration
│   ├── environment/     ← Atari wrapper, preprocessing
│   ├── replay_buffer/   ← Experience replay buffer
│   ├── api/             ← FastAPI inference server
│   ├── config/          ← Hyperparameters YAML
│   └── utils/           ← Logger, checkpointing, video
├── scripts/
│   ├── train.py         ← Training loop
│   ├── evaluate.py      ← Evaluation script
│   └── play.py          ← Record gameplay video
├── notebooks/
│   └── analysis.ipynb   ← Kaggle GPU training notebook
├── models/              ← Saved model checkpoints (.pth)
├── gameplay_videos/     ← Recorded MP4 demos
├── tests/               ← Unit tests (31 tests)
├── Dockerfile.train     ← Training container
├── Dockerfile.inference ← Inference API container
├── docker-compose.yml   ← Multi-service orchestration
├── submission.yml       ← Automated commands
├── .env.example         ← Environment variables example
├── README.md
└── METHODOLOGY.md


⚙️ Setup
Requirements

Docker Desktop (required)
Git
Python 3.8+ (optional, for local testing)

Step 1 — Clone Repo
git clone https://github.com/vikram0678/atari-dqn-agent.gitcd atari-dqn-agent
Step 2 — Copy Environment File
cp .env.example .env
Step 3 — Build Docker Images
docker build -t rl-agent-train -f Dockerfile.train .docker build -t rl-agent-inference -f Dockerfile.inference .

🏋️ Training
Option 1 — Docker (Local CPU)
Good for testing and short runs:
docker run --rm \  -v $(pwd)/models:/app/models \  -v $(pwd)/logs:/app/logs \  rl-agent-train \  python scripts/train.py --game ALE/Pong-v5 --episodes 1000
Resume from checkpoint:
docker run --rm \  -v $(pwd)/models:/app/models \  rl-agent-train \  python scripts/train.py \  --resume /app/models/latest_model.pth \  --episodes 1000
Model is saved to models/latest_model.pth automatically.

Option 2 — Kaggle GPU (Recommended for Full Training)
Full training (1600 episodes) takes:

Kaggle GPU T4 → ~8 hours ✅
Local CPU → ~5-7 days ❌

Steps:

Go to kaggle.com → New Notebook
Upload notebooks/analysis.ipynb
Settings → Accelerator → GPU T4 x2 ✅
Settings → Internet → ON ✅
Run all cells in order (Cell 1 → Cell 10)
After training completes → Output panel → Download:

best_model.pth
latest_model.pth


Place downloaded files in models/ folder:

cp ~/Downloads/best_model.pth   models/best_model.pthcp ~/Downloads/latest_model.pth models/latest_model.pth

Note: Pre-trained model already included in this repo.
Average reward achieved: 16.81 over 100 evaluation episodes.


📊 Evaluation
Run 100 episodes and get average reward:
docker run --rm \  -v $(pwd)/models:/app/models \  rl-agent-train \  python scripts/evaluate.py \  --game ALE/Pong-v5 \  --model /app/models/best_model.pth \  --episodes 100
Expected output:
EVALUATION RESULTS (100 episodes)
Average Reward : 16.81
AVERAGE_REWARD=16.81


🎬 Record Gameplay Video
docker run --rm \  -v $(pwd)/models:/app/models \  -v $(pwd)/gameplay_videos:/app/gameplay_videos \  rl-agent-train \  python scripts/play.py \  --game ALE/Pong-v5 \  --model /app/models/best_model.pth \  --output /app/gameplay_videos/demo.mp4
Video saved to gameplay_videos/demo.mp4

🌐 Inference API
Start Server
docker run -d \  -p 8000:8000 \  -v $(pwd)/models:/app/models \  -e MODEL_PATH=/app/models/best_model.pth \  -e ATARI_GAME_ID=ALE/Pong-v5 \  -e N_ACTIONS=6 \  rl-agent-inference
Health Check
curl http://localhost:8000/health
{"status": "ok", "model_loaded": true, "game": "ALE/Pong-v5"}
Predict Action
python -c "import requests, numpy as npstate = np.zeros((4, 84, 84), dtype=np.float32).tolist()r = requests.post('http://localhost:8000/predict', json={'state': state})print(r.json())"
{"action": 3, "q_values": [1.64, 1.61, 1.65, 1.67, 1.60, 1.59]}
Swagger UI
Open browser → http://localhost:8000/docs

API Endpoints






























MethodEndpointDescriptionGET/API infoGET/healthHealth checkPOST/predictGet action from stateGET/docsSwagger UI

🔧 Environment Variables



































VariableDefaultDescriptionATARI_GAME_IDALE/Pong-v5Atari game IDMODEL_PATHmodels/latest_model.pthModel checkpoint pathN_ACTIONS6Number of discrete actionsEPISODES1000Training episodes overrideLR0.0001Learning rate override

🐳 Submission Commands
# BUILD both imagesdocker build -t rl-agent-train -f Dockerfile.train . && \docker build -t rl-agent-inference -f Dockerfile.inference .
# TRAIN agentdocker run --rm \  -v $(pwd)/models:/app/models \  rl-agent-train \  python scripts/train.py --game ALE/Pong-v5 --episodes 1000
# EVALUATE agent (100 episodes)docker run --rm \  -v $(pwd)/models:/app/models \  rl-agent-train \  python scripts/evaluate.py \  --game ALE/Pong-v5 \  --model /app/models/best_model.pth \  --episodes 100
# PLAY — record gameplay videodocker run --rm \  -v $(pwd)/models:/app/models \  -v $(pwd)/gameplay_videos:/app/gameplay_videos \  rl-agent-train \  python scripts/play.py \  --game ALE/Pong-v5 \  --model /app/models/best_model.pth \  --output /app/gameplay_videos/demo.mp4

🧪 Run Tests
docker run --rm rl-agent-train pytest tests/ -v
Expected: 31 passed ✅

📈 TensorBoard
docker run --rm -p 6006:6006 \  -v $(pwd)/logs:/app/logs \  rl-agent-train \  tensorboard \  --logdir /app/logs/tensorboard \  --host 0.0.0.0 \  --port 6006
Open: http://localhost:6006

🤖 How The Agent Works
1. Agent sees raw Pong screen (210x160 RGB pixels)
2. Converts to grayscale → resizes to 84x84
3. Stacks 4 consecutive frames to capture motion
4. CNN neural network outputs Q-value for each action
5. Agent picks action with highest Q-value
6. Gets reward: +1 (score point), -1 (lose point)
7. Learns from 1600 games using experience replay
8. Final avg reward: 16.81 / 21 maximum possible


🔧 Troubleshooting
Docker Build Takes Too Long
Problem: requirements downloading for hours
Fix: Make sure you have good internet
     Or use --no-cache flag:
     docker build --no-cache -t rl-agent-train -f Dockerfile.train .

Port Already In Use
Problem: port 8000 already allocated
Fix: Stop all containers first:
     docker stop $(docker ps -q)
     Then run API again

Model Not Found Error
Problem: Model file not in models/ folder
Fix: Check models folder:
     ls -lh models/
     Make sure best_model.pth exists
     If missing, download from Kaggle or retrain

ALE Namespace Not Found (Kaggle)
Problem: NamespaceNotFound: ALE
Fix: Run this before training:
     import ale_py, gymnasium as gym
     gym.register_envs(ale_py)

GPU Not Available
Problem: Training is slow
Fix: Use Kaggle GPU notebook
     Settings → Accelerator → GPU T4 x2

API model_loaded = false
Problem: Model path wrong
Fix: Use MSYS_NO_PATHCONV=1 on Windows Git Bash:
     MSYS_NO_PATHCONV=1 docker run -d -p 8000:8000 \
     -v $(pwd)/models:/app/models \
     -e MODEL_PATH=/app/models/best_model.pth \
     rl-agent-inference

Out Of Memory During Training
Problem: RAM crash on Kaggle
Fix: Reduce replay buffer size in CONFIG:
     "replay_buffer_size": 50000
     "min_replay_size": 5000

Resume Training After Crash
The training auto-resumes from latest checkpoint.
Just re-run Cell 1 → Cell 8 in Kaggle notebook.
It automatically finds the highest episode checkpoint.

