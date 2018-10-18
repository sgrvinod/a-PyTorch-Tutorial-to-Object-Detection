This is a **[PyTorch](https://pytorch.org) Tutorial to Object Detection**.

This is the fourth in [a series of tutorials](https://github.com/sgrvinod/Deep-Tutorials-for-PyTorch) I plan to write about _implementing_ cool models on your own with the amazing PyTorch library.

Basic knowledge of PyTorch, convolutional neural networks is assumed.

If you're new to PyTorch, first read [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) and [Learning PyTorch with Examples](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html).

Questions, suggestions, or corrections can be posted as issues.

I'm using `PyTorch 0.4` in `Python 3.6`.

# Contents

[***Objective***](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#objective)

[***Concepts***](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#tutorial-in-progress)

[***Overview***](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#tutorial-in-progress)

[***Implementation***](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#tutorial-in-progress)

[***Training***](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#tutorial-in-progress)

[***Inference***](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#tutorial-in-progress)

[***Frequently Asked Questions***](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#tutorial-in-progress)

# Objective

**To build a model that can detect and localize specific objects in images.**

<p align="center">
<img src="./img/baseball.gif">
</p>

 We will be implementing the [Single Shot Multibox Detector (SSD)](https://arxiv.org/abs/1512.02325), a popular, powerful, and especially nimble network for this task. The authors' original implementation can be found [here](https://github.com/weiliu89/caffe/tree/ssd).

Here are a few examples of images not seen during training:

---

<p align="center">
<img src="./img/000001.jpg">
</p>

---

<p align="center">
<img src="./img/000022.jpg">
</p>

---

<p align="center">
<img src="./img/000069.jpg">
</p>

---

<p align="center">
<img src="./img/000082.jpg">
</p>

---

<p align="center">
<img src="./img/000144.jpg">
</p>

---

<p align="center">
<img src="./img/000139.jpg">
</p>

---

<p align="center">
<img src="./img/000116.jpg">
</p>

---

<p align="center">
<img src="./img/000098.jpg">
</p>

---

<p align="center">
<img src="./img/airport.gif">
</p>

There will be more examples at the [end of the tutorial](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#tutorial-in-progress).

# Tutorial in Progress

I am still writing this tutorial.

<p align="center">
<img src="./img/incomplete.jpg">
</p>

In the meantime, you could take a look at the code – it works!

We achieve an mAP of **77.1** (against **77.2** in the paper) on VOC2007-test.
