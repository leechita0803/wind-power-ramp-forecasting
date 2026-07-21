# Contributing to Wind Power Ramp Forecasting

Thank you for your interest in contributing to this project.

This repository presents an end-to-end framework for **8-hour wind power forecasting** and **wind ramp detection** using **Variational Mode Decomposition (VMD)**, **XGBoost**, and an interactive **Streamlit dashboard**.

We welcome contributions that improve the project, fix bugs, enhance documentation, or introduce new forecasting techniques.

---

## How to Contribute

### 1. Fork the Repository

Create your own copy of the repository by clicking the **Fork** button on GitHub.

---

### 2. Clone the Repository

```bash
git clone https://github.com/<your-github-username>/wind-power-ramp-forecasting.git
cd wind-power-ramp-forecasting
```

---

### 3. Create a Virtual Environment

Windows

```bash
python -m venv .venv
```

Activate

```bash
.venv\Scripts\activate
```

macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 5. Create a New Branch

```bash
git checkout -b feature/your-feature-name
```

Example

```bash
git checkout -b feature/improve-ramp-detection
```

---

### 6. Make Your Changes

Please ensure that your contribution:

- follows the existing project structure
- is well documented
- includes comments where appropriate
- does not break the existing pipeline

---

### 7. Test the Project

Run the forecasting pipeline

```bash
python main.py
```

Run the Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

Verify that:

- Forecast metrics are generated
- Ramp detection executes successfully
- Dashboard loads without errors
- Output files are created correctly

---

### 8. Commit Your Changes

```bash
git add .
git commit -m "Add meaningful commit message"
```

Example

```bash
git commit -m "Improve ramp detection visualisation"
```

---

### 9. Push Your Branch

```bash
git push origin feature/your-feature-name
```

---

### 10. Open a Pull Request

Create a Pull Request describing:

- What was changed
- Why the change was made
- Any relevant screenshots (if applicable)
- Testing performed

---

# Contribution Guidelines

Please:

- Keep code modular.
- Follow PEP 8 coding conventions.
- Add meaningful comments where necessary.
- Update documentation if functionality changes.
- Maintain compatibility with the existing project structure.

---

# Reporting Issues

If you discover a bug, please create a GitHub Issue and include:

- Operating system
- Python version
- Error message
- Steps to reproduce
- Expected behaviour
- Screenshots (if applicable)

---

# Feature Requests

Suggestions for future improvements are welcome, including:

- Improved forecasting models
- Alternative decomposition methods
- Enhanced dashboard visualisations
- Performance optimisation
- Additional evaluation metrics
- New ramp detection techniques

---

# Code of Conduct

Please be respectful and constructive when contributing.

All contributors are expected to maintain a welcoming and collaborative environment.

---

# Citation

If you use this project in academic research, please cite the associated publication listed in the repository's `CITATION.cff` file.

---

Thank you for contributing to the Wind Power Ramp Forecasting project.