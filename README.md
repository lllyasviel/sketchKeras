# sketchKeras
An u-net with some algorithm to take sketch from paints.

# requirement
* Keras
* Opencv
* tensorflow/theano
* numpy

# download mod
see release

# performance
Currently there are many edge-detecting algrithoms or nerual networks. But few of them has good performance on paintings, espatially those from comic or animate. Most of these existing methods just detect the edge and then add lines to the edge. However, we need a method to convert the painting to a sketch which looks like a painter drawed the outline of picture. It is important when we want to train a nerual networks to colorlize pictures.(Paper is on the way)

Here is a example of artificial sketch for reference.

![goal](https://raw.githubusercontent.com/lllyasviel/sketchKeras/master/github/example.png)

Here a conclusion of existing methods to handle the problem.
* use opencv and implements a high-pass effect to get edge
* train a nerual network (HED Edge Detect)(PaintsChainer's lnet)
* use this sketchKeras which combined algorithm and nerual networks

