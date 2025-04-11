# TAbKAN

# TabKAN: Advancing Tabular Data Analysis using Kolmograv-Arnold Network

Welcome to the repository for **"TabKAN: Advancing Tabular Data Analysis using Kolmograv-Arnold Network"**.  
It provides multiple KAN-based architectures like ChebyshevKAN, FourierKAN, PadeRKAN, and FastKAN — including Mixer-style variants.

## Requirements

### Prerequisites
To run this code, ensure that you have:
- Python 3.9.21 installed on your system.

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/aseslamian/TAbKAN
   cd TAbKAN
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Examples

You can find example datasets and usage scenarios in the **`Example`** folder. These examples provide step-by-step instructions on how to use the code effectively.

---
## Usage
If you intend to use TabMixer block as seperate module:

```bash
pip install tabkan
```

```python
from tabkan import ChebyshevKANMixer, FourierKAN, ChebyshevKAN, JacobiKAN, FractionalKAN, PadeKAN, FastKAN, ChebyshevKANMixer, FourierKANMixer

model = ChebyshevKANMixer(
    num_features=30, 
    num_classes=2,
    num_layers=4,
    token_dim=64,
    channel_dim=128,
    token_order=3,
    channel_order=3
)
output = model(X)  # X: (batch_size, num_features)

```

---
