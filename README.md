# Kernel Change-Point Detection with Spectral Attention
## Algorithm
### Inputs
- observation `y` $ \in R^{n \times d}$  (`n`time steps, `d` features)
- chunk size `T` $ \in N^+$

### Output
- split point `s_star`
### Pseudocode
```
// initialization
kernel k
maxMMD = 0
s_star = 1

// evaluate all split point candidates s
for s = T to n-T+1:
    MMD(s) is a real value depending on (y, T, k, s)
    if MMD(s) > maxMMD:
        maxMMD = MMD(s)
        s_star = s
```

## Example:
Example of an expected split point of `y` is 5 (`python` index is 4):
```python
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

T = 2

print(spectral_cpd(y, T))
```
```bash
4
```

## Developer
[kernel](https://github.com/lamtung16/spectral-attention-kernel-cpd/blob/main/source.py#L4)