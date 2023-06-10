# Aliproduct-BLIP-cvpr2023
## This is the solution for Aliproduct Largs-Scale Competition on CVPR2023 workshop.

We finish the [AliProduct competition](https://tianchi.aliyun.com/competition/entrance/532077/introduction?spm=a2c22.12281957.0.0.605a3b74xC42iA), upload our solution on GitHub and the report will come soon.

## Result
Our solution achieves an average recall of **0.76** on the val dataset **without pre-trained models and doesn't require additional dirty data pre-processing and multi-stage training** and can achieve a speed of 0.16s per image per gpu.
We train the model on 8*A100 40G with the aliproduct dataset, which include 4 million image-context pairs.

## How to use?

### 1.install the environment
```bash
conda create -n air python=3.9
conda activate air
pip install -r requirements.txt
```

### 2.train the model
You can change some hyperparameters in **train_retrieval.py** before run.
```bash
bash run.sh
```

### 3.val and predict
After finish the train steps, you can use the **itm_predict.py** or **itc_predict.py** to predict the result. If you want to test the preformance, do this:
```bash
bash test.sh
```
The **test.sh** will compute the itm_socre or itc_score for top_k image-context pairs. The start and end is for image index to accelerate by multi-gpus. Each 10 image-context pairs need 3.6s on an A100 80G gpu.

## Visualization
1. **context**: "M & D Simple Modern Light Luxury Comfort Good Quality Living Room with a Double Motor Lounge Chair Sofa TE04"
<table>
    <tr>
        <td ><center><img src=imgs/cap1_1.jpg width=200 height=200/></center></td>
        <td ><center><img src=imgs/cap1_2.jpg width=200 height=200/></center></td>
        <td ><center><img src=imgs/cap1_3.jpg width=200 height=200/></center></td>
    </tr>
</table>


1. **context**: "er tong hua xing che fang ce fan niu niu che 1-3 sui bao bao wan ju che yin le ke zuo ke qi si lun lium che"
<table>
    <tr>
        <td ><center><img src=imgs/cap2_1.jpg width=200 height=200/></center></td>
        <td ><center><img src=imgs/cap2_2.jpg width=200 height=200/></center></td>
        <td ><center><img src=imgs/cap2_3.jpg width=200 height=200/></center></td>
    </tr>
</table>

1. **context**: "feiyangg/LP Paragraph Style Electric Guitar Tiger Veneer Factory Direct Color Can Be Customized"
<table>
    <tr>
        <td ><center><img src=imgs/cap3_1.jpg width=200 height=200/></center></td>
        <td ><center><img src=imgs/cap3_2.jpg width=200 height=200/></center></td>
        <td ><center><img src=imgs/cap3_3.jpg width=200 height=200/></center></td>
    </tr>
</table>