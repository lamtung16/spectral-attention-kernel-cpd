# Kernel Change-Point Detection with Spectral Attention
## Algorithm
### Inputs
- sequence `Y` (`n`time steps, `d` features)
- penalty value `pen` (penalize the changepoint presence)
- model `srbf` (Spectral Radial Basis Function)
- chunk size `T`

### Output
- change points

## Example:
Example of an expected split point of `y` is 5 (`python` index is 4):
```python
import numpy as np
from kernelcpd import pelt

y = np.array([
    [1.5,  3.6],
    [1.4,  3.5],
    [1.4,  3.5],
    [1.4,  3.5],
    [1.5,  3.6],
    [3.4,  1.5],
    [3.5,  1.4],
    [3.4,  1.6],
    [3.6,  1.5],
    [3.6,  1.4],
])

print(pelt(sequence = y, pen = 0.2, model = 'srbf', T = 2))
```
```bash
[4]
```

## Developer
[spectral rbf kernel](https://github.com/lamtung16/spectral-attention-kernel-cpd/blob/main/kernelcpd.py#L11)
