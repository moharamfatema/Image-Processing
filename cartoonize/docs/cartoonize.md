# Cartoonizing images

## Imports

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt
```

## Constants

These are used as parameters for the filters. Tweaking them would give different results.

```python
MEDIAN_K = 5
LAPLACE_K = 5
THRESHOLD = 180

DIAMETER = 9
SIGMA_COLOR = 9
SIGMA_SPACE = 7

REPS = 10
```

## Steps of the algorithm

### Reading the image

```python
img = cv2.imread('img/a.png')
ax = plt.imshow(img[:,:,::-1])
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.show()
```

![png](output_6_0.png)

### 1.1.1 Noise Reduction Using Median Filter

```python
edg_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edg_img = cv2.medianBlur(edg_img, MEDIAN_K)
plt.imshow(edg_img, cmap='gray')
plt.show()
```

![png](output_8_0.png)

### 1.1.2 Edge Detection Using Laplacian Filter

```python
edg_img = cv2.Laplacian(edg_img, cv2.CV_8U, ksize=LAPLACE_K)
plt.imshow(edg_img, cmap='gray')
plt.show()
```

![png](output_10_0.png)

### Threshold

```python
thr_img = cv2.threshold(edg_img, THRESHOLD, 255, cv2.THRESH_BINARY)[1]
thr_img = 255 * (thr_img == 0).astype(np.uint8)
plt.imshow(thr_img, cmap='gray')
plt.show()
```

![png](output_12_0.png)

## 1.2 Generating a color painting and a cartoon

```python
bi_img = cv2.bilateralFilter(img, DIAMETER, SIGMA_COLOR, SIGMA_SPACE)
for i in range(REPS):
    bi_img = cv2.bilateralFilter(bi_img, DIAMETER, SIGMA_COLOR, SIGMA_SPACE)
plt.imshow(bi_img[:,:,::-1])
plt.show()
```

![png](output_14_0.png)

## Final Result

```python
res = bi_img * (np.repeat(thr_img,3, axis=-1).reshape(bi_img.shape) // 255)
plt.imshow(res[:,:,::-1])
plt.show()
```

![png](output_16_0.png)

## Defining the steps in a function

```python
def cartoonize(path: str):
    # Read image
    img = cv2.imread(path)

    # Convert to grayscale
    edg_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edg_img = cv2.medianBlur(edg_img, MEDIAN_K)

    # Detect edges and threshold
    edg_img = cv2.Laplacian(edg_img, cv2.CV_8U, ksize=LAPLACE_K)
    edg_img = cv2.threshold(edg_img, THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    edg_img = 255 * (edg_img == 0).astype(np.uint8)

    # Apply bilateral filter
    bi_img = cv2.bilateralFilter(img, DIAMETER, SIGMA_COLOR, SIGMA_SPACE)
    for i in range(REPS - 1):
        bi_img = cv2.bilateralFilter(bi_img, DIAMETER, SIGMA_COLOR, SIGMA_SPACE)

    # sketch over painting
    return bi_img * (np.repeat(edg_img,3, axis=-1).reshape(bi_img.shape) // 255)

```

```python
def view(path: str):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(cv2.imread(path)[:,:,::-1])
    axs[0].axes.get_xaxis().set_visible(False)
    axs[0].axes.get_yaxis().set_visible(False)
    axs[1].imshow(cartoonize(path)[:,:,::-1])
    axs[1].axes.get_xaxis().set_visible(False)
    axs[1].axes.get_yaxis().set_visible(False)
    plt.show()
```

## Testing on different pictures of different objects and noise intensity

```python
view('img/a.png')
```

![png](output_21_0.png)

```python
view('img/b.jpg')
```

![png](output_22_0.png)

```python
view('img/c.jpg')
```

![png](output_23_0.png)
