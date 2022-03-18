<h4>File Description</h4>
<ul>
  <li>The code in file densenet.py is the training code of DenseNet model</li>
</ul>

<h4>The architecture of DenseNet model</h4>
<p>Our “DenseNet” model consist of a convolutional block and four Dense-Blocks, and there is a transition layer between each two Dense-Blocks. The convolutional block (ConvBlock1) contains two convolution layers with the same number of convolutional kernels (72 filters, kernel width 3, ReLU, L2 regularization) and an average pooling layer (stride 2). In each Dense-Block, there are Li convolutional blocks (ConvBlock2, which is similar to ConvBlock1 but does not contain a pooling layer; Li is 6, 12, 24 and 16 in turn), and all of these blocks connect (with matching feature map sizes)  directly with each other. Each block obtains additional inputs from all preceding layers and passes on its own features to all subsequent blocks. Between each two Dense-Blocks, there is a transition layer that contains a compressed convolutional layer and an average pooling layer (stride 2). The output of the last Dense-Block is fed to an average pooling layer (stride 2). The output of the last Dense-Block is fed to an average pooling layer and connect it to a fully connected layer to generate an output value. The fully connected layer and all convolution layers are initialized with Xavier normal initialization.</p>
