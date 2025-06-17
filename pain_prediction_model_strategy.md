**Pain Intensity Prediction Model: Feature Engineering and Generalization Strategy**

---

### 🔢 Signal Types and Features

#### 1. **Electromyogram (EMG)**

**Muscles Used:**

- **Corrugator Supercilii**: Activated during frowning; highly correlated with pain and negative affect.
- **Zygomaticus Major**: Associated with smiling; useful for distinguishing non-pain-related expressions.
- **Trapezius**: Reflects postural tension and stress-related muscle activation; relevant for chronic or muscular pain.

**Best Reliability:** Corrugator EMG is most reliable for **emotional and facial pain response**, while Trapezius EMG is informative for **somatic tension**. Zygomaticus helps to detect false positives.

**Key Features:**

- **Root Mean Square (RMS)**: \(\text{RMS} = \sqrt{\frac{1}{N} \sum_{i=1}^N x_i^2}\)
- **Log RMS**: \(\log(1 + \text{RMS})\)
- **Standard Deviation**: \(\sqrt{\frac{1}{N} \sum (x_i - \bar{x})^2}\)
- **Waveform Length**: Sum of absolute differences in signal amplitude.
- **Zero-Crossing Rate (ZCR)**: Number of times the signal crosses zero.

#### 2. **Skin Conductance Level (SCL)**

**Importance:** Highly sensitive to sympathetic arousal, making it useful for detecting **phasic responses to acute pain**.

**Key Features:**

- **Mean SCL**
- **Number of Skin Conductance Responses (SCRs)**
- **SCR Amplitude**
- **Slope**: \(\frac{SCL_{end} - SCL_{start}}{\text{duration}}\)
- **Area Under Curve (AUC)**
- **Log-transformed values** for improved model performance

#### 3. **Electrocardiogram (ECG)**

**Importance:** Reflects autonomic nervous system activity. Useful for **tonic or sustained pain** states.

**Key Features:**

- **Heart Rate (HR)**
- **R-R Interval**
- **HRV Time-domain**: RMSSD, SDNN
- **HRV Frequency-domain**: LF, HF, LF/HF Ratio
- **Poincaré Plot Features**: SD1, SD2
- **Log LF/HF or RMSSD** for normalization

#### 4. **Time-Frequency Features**

**Approach:** Apply **Discrete Wavelet Transform (DWT)** (e.g., Daubechies db4 or db6)

**Key Features:**

- **Wavelet Energy**: Power in signal bands
- **Entropy**: Signal complexity
- **Higher-order statistics**: Skewness, kurtosis

---

### 🧠 Feature Preprocessing & Normalization

- **Z-score normalization per 10s segment**: \(z = \frac{x - \mu}{\sigma}\)
- **Log transformation** for skewed or heavy-tailed distributions
- **Per-segment normalization** supports generalization to new subjects

---

### 🚫 Handling Noisy Signals

**Noise Challenges:** Simulated pain may introduce artifacts due to facial acting or inconsistent expressions, particularly in EMG signals.

**Noise Reduction Strategies:**

- Apply **bandpass filters** tailored to each signal (e.g., EMG: 20–450 Hz, ECG: 0.5–50 Hz, SCL: 0.05–5 Hz)
- Use **notch filters** at 50/60 Hz to remove powerline noise
- **Downsample** to 250 Hz for efficiency after filtering
- Apply **Independent Component Analysis (ICA)** for artifact rejection (especially in EMG)
- Prefer **robust features** like RMS, wavelet energy over raw amplitude
- Use **wavelet-based denoising** to isolate meaningful signal components
- Include **slope, ZCR, and waveform length** to mitigate spurious bursts

---

### 🤖 Modeling Strategy

#### Train-Test Setup

- Use **Leave-One-Subject-Out (LOSO) Cross-Validation** for generalization
- Segment the dataset into **10-second non-overlapping windows**

#### Machine Learning Models

- **XGBoost**: Excellent baseline for structured data
- **SVM (RBF Kernel)**: Effective on small noisy datasets
- **KNN/Decision Tree**: Interpretable and lightweight
- **SVR (Support Vector Regression)**: For estimating continuous pain scores

#### Ensemble Strategy

- Combine base models using **soft or adaptive voting**
- Assign weights based on validation F1-score or accuracy

---

### 📊 Generalization & Domain Adaptation

#### Handling Subject Variability

- Train using **subject-independent protocols**
- Normalize each window independently to reduce bias
- Focus on **relative features**: \(\Delta\text{RMS}, \text{ratios}, \text{slope}, \text{change-from-baseline}\)

#### Domain-Invariant Learning

- Incorporate **CORAL (CORrelation ALignment)** loss to align features across subjects
- Use **MMD (Maximum Mean Discrepancy)** to minimize distribution shift
- Apply **adversarial training** if using deep models (e.g., Domain-Adversarial Neural Networks)

#### Test-Time Adaptation (Unlabeled Data)

- Apply **adaptive batch normalization**
- Use **entropy minimization** to sharpen predictions without ground truth

---

### 💡 Summary Workflow

```text
[Preprocessing] → Bandpass Filter, Segment, Normalize
[Feature Extraction] → Time, Frequency, Relative Metrics
[Noise Reduction] → Wavelet Denoising, ICA, Slope/ZCR Filters
[Modeling] → LOSO CV, XGBoost/SVM, Voting Ensemble
[Generalization] → Domain losses, No label adaptation, Robust features
```

This complete strategy ensures robust generalization to **new, unseen, unlabeled subjects**, making it ideal for practical applications in clinical pain monitoring or wearable pain detection devices.
