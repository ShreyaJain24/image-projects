Agenda - style transfer architecture, uisng pre trained features, pytorch model, visualinf pytorch models

1. Use pre-trained VGG (after Transformer Net)
2. Inputs ->  content image(original), style image, input image(initialized random or content) to be the final output image
-> Instead of doing per-pixel (low level features) loss after transformer Net(downsampling-upsampling) model, we use these two losses: from pre trained VGG network, at various layers
3. Content loss = Euclidean distance between content and input image, layer wise
4. Style loss = mean square loss of feature maps (Gram matrix), layer wise
5. SGD, Adam optimizer


- Side note - For super-resolution with an upsampling factor of f, we use several residual blocks followed by log2 f convolutional layers
with stride 1/2. This is different from  bicubic interpolation to upsample the low-resolution input before passing it to the network.

- Residual Connections - residual connections make it easy for the network to learn the identify function; this is an appealing property
for image transformation networks, since in most cases the output image should share structure with the input image

- Downsampling and Upsampling 1) Wider perceptive field when downsampled
2) Low computation cost



- add dynamic loading pytorch