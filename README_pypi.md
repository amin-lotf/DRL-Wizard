# DRL Wizard

**DRL Wizard** is a clean, modular Deep Reinforcement Learning toolkit for training, comparing, and understanding modern RL algorithms. It supports multiple environments, multiple algorithms, real-time monitoring, and a unified workflow through a FastAPI backend and Streamlit UI.

---

## 🚀 Features

- **Algorithms:** PPO, TRPO, DQN, Double DQN, Dueling DQN, SAC (more coming)
- **Environments:** Gymnasium, Atari (ALE), image-based & multi-discrete action spaces
- **Architecture:** FastAPI backend, Streamlit UI, SQLAlchemy storage, Pydantic configs
- **Experiment Tools:** concurrent jobs, graceful stop, NDJSON logs, TensorBoard, job archives
- **Extensible:** easy to add algorithms, envs, or visualization components

---

---

## 📦 Installation
Basic
```bash
pip install drl-wizard
```

UI + Dev tools:
```bash
pip install drl-wizard[ui,dev]
```

## 🖥️ Running
 - **Running the UI & Backend:**
```bash
drl-wizard-run
```

 - **Running Backend:**
```bash
drl-wizard-api
```

 - **Running UI:**
```bash
drl-wizard-ui
```

## 📝 Notes
- drl-wizard-run launches the full platform.

- Extras ([ui], [dev]) include Streamlit, and development dependencies.