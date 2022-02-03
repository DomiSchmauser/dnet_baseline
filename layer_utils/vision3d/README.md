# Vision3d Layer

## Installation
```
python setup.py install
```

## Usage
```python
from vision3d.layers import nms3d, nms
```

## Example
```python
from vision3d.layers import nms3d
bboxes = torch.Tensor([[10,20,30,50,60,70],[11,20,30,50,60,70],[0,1,2,10,11,12]]).cuda()
scores =  torch.Tensor([0.2,1,0.6]).cuda()
nms_threshold = 0.5
nms3d(a,s,nms_threshold)
```
```python
>>> tensor([1, 2], device='cuda:0')
```