This is a **[PyTorch](https://pytorch.org) Tutorial to Text Classification**.

This is the third in [a series of tutorials](https://github.com/sgrvinod/Deep-Tutorials-for-PyTorch) I plan to write about _implementing_ cool models on your own with the amazing PyTorch library.

Basic knowledge of PyTorch, recurrent neural networks is assumed.

If you're new to PyTorch, first read [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) and [Learning PyTorch with Examples](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html).

Questions, suggestions, or corrections can be posted as issues.

I'm using `PyTorch 0.4` in `Python 3.6`.

# Contents

[***Objective***](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Text-Classification#objective)

[***Concepts***](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Text-Classification#tutorial-in-progress)

[***Overview***](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Text-Classification#tutorial-in-progress)

[***Implementation***](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Text-Classification#tutorial-in-progress)

[***Training***](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Text-Classification#tutorial-in-progress)

[***Inference***](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Text-Classification#tutorial-in-progress)

[***Frequently Asked Questions***](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Text-Classification#tutorial-in-progress)

# Objective

**To build a model that can label a text document as one of several categories.**

 We will be implementing the [Hierarchial Attention Network (HAN)](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf), one of the more interesting and interpretable text classification models.

This model not only classifies a document, but also _chooses_ specific parts of the text –  sentences and individual words – that it thinks are most important.

---

> "I think I'm falling sick. There was some indigestion at first. But now a fever is beginning to take hold."

<p align="center">
<img src="./img/baseball.gif">
</p>
---

> "But think about it! It's so cool. Physics is really all about math. What Feynman said, hehe."

![](./img/science.png)

---

> "I want to tell you something important. Get into the stock market and investment funds. Make some money so you can buy yourself some yogurt."

![](./img/finance.png)

---

> "How do computers work? I have a CPU I want to use. But my keyboard and motherboard do not help."
>
> "You can just google how computers work. Honestly, its easy."

![](./img/computers.png)

---

> "You know what's wrong with this country? Republicans and democrats. Always at each other's throats"
>
> "There's no respect, no bipartisanship."

![](./img/politics.png)

---

# Tutorial in Progress

I am still writing this tutorial.

<p align="center">
<img src="./img/incomplete.jpg">
</p>

In the meantime, you could take a look at the code – it works!
