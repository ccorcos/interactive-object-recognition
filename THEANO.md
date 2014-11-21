# Debugging Notes

[This is a good resource](http://deeplearning.net/software/theano/tutorial/debug_faq.html).

Print out `.type` of a theano variable to get information about the tensor type.

Also, try running your program with

    THEANO_FLAGS="optimizer=None" python program.py

This will give you line numbers and more information.
