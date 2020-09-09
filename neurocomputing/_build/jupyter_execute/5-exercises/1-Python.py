# Introduction To Python

## Introduction

Python is a powerful, flexible programming language you can use for scientific computing, in web/Internet development, to write desktop graphical user interfaces (GUIs), create games, and much more.

Python is an high-level, interpreted, object-oriented language written in C, which means it is compiled on-the-fly, at run-time execution.

Its syntax is close to C, but without prototyping (whether a variable is an integer or a string will be automatically determined by the context).

It can be executed either directly in an interpreter (à la Matlab), in a script or in a notebook (as here).

The documentation on Python can be found at [http://docs.python.org](http://docs.python.org).


Many resources to learn Python exist on the Web:

-  Free book [Dive into Python](http://www.diveintopython.net/).
-  [Learn Python the hard way](http://learnpythonthehardway.org).
-  Learn Python on [Code academy](http://www.codecademy.com/tracks/python).
-  Scipy lectures note [http://www.scipy-lectures.org](http://www.scipy-lectures.org/)
-  An Introduction to Interactive Programming in Python on [Coursera](https://www.coursera.org/course/interactivepython).

## Working With Python

### Python Interpreter

To start the Python interpreter, simply type its name in a terminal under Linux:

```bash
user@machine ~ $ python
```

```
Python 3.7.4 (default, Jul 16 2019, 07:12:58) 
[GCC 9.1.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> 

```

To exit Python call the `exit()` function (or `Ctrl+d`):

```python
>>> exit()
```


### Scripting

Instead of using the interpreter, you can run scripts which can be executed sequentially. Simply edit a text file called `MyScript.py` containing for example:

```python
# MyScript.py
# Implements the Hello World example.

text = 'Hello World!' # define a string variable

print(text)
```

The `#` character is used for comments. Execute this script by typing in a Terminal:

```bash
python MyScript.py
```

As it is a scripted language, each instruction in the script is executed from the beginning to the end, except for the declared functions or classes which can be used later.

**Note:** In the B202/B201, it is advised to use `kate` or `kwrite` to edit Python files. The default editor bluefish is for html files...

### Jupyter Notebooks

A third recent (but very useful) option is to use Jupyter notebooks (formerly IPython notebooks). 

Jupyter notebooks allow you to edit Python code in your browser (but also Julia, R, Scala...) and run it locally. 

To launch a Jupyter notebook, type in a terminal:

```bash
jupyter notebook
```

and create a new notebook (Python 3)

When a Jupyter notebook already exists (here `Python.ipynb`), you can also start it directly:

```bash
jupyter notebook Python.ipynb
```

The main particularity is that code is not executed sequentially from the beginning to the end, but only when a **cell** is explicitly run with **Ctrl + Enter** (the active cell stays the same) or **Shift + Enter** (the next cell becomes active).

To edit a cell, select it and press **Enter** (or double-click).

In the next cell, run the Hello World! example:

```python
text = 'Hello World!'
print(text)
```

print("Hello World!")

There are three types of cells:

* Python cells allow to execute Python code (the default)
* Markdown cells which allow to document the code nicely (code, equations), like the current one.
* Raw cell are passed to nbconvert directly, it allows you to generate html or pdf versions of your notebook (not used here).

**Beware that the order of execution of the cells matters!**

In the next three cells, put the following commands:

1. `text = "Text A"`
2. `text = "Text B"`
3. `print(text)`

and run them in different orders (e.g. 1, 2, 3, 1, 3)







Executing a cell can therefore influence cells before and after it. If you want to run the notebook sequentially, select **Kernel/Restart & Run all**.

Take a moment to explore the options in the menu (Insert cells, Run cells, Download as Python, etc).

## Basics In Python

### Print Statement And Print Function

In python 3, the `print()` function is a regular function:

```python
print(value1, value2, ...)
```

You can give it as many arguments as you want (of whatever type), they will be printed one after another separated by spaces.

Try to print "Hello World!" using two different strings "Hello" and "world!":



### Data Types

As Python is an interpreted language, variables can be assigned without specifying their type: it will be inferred at execution time.

The only thing that counts is how you initialize them and which operations you perform on them.

```python
a = 42 # Integer
b = 3.14159 # Double precision float
c = 'My string' # String
d = False # Boolean
e = a > b # Boolean
```

Print these variables as well as their type:

```python
print(type(a))
```



### Assignment Statement And Operators

#### Assignment Statement

The assignment can be done for a single variable, or for a tuple of variables separated by commas:


```python
m = 5 + 7

x, y = 10, 20

a, b, c, d = 5, 'Text', None, x==y
```

Try these assignments and print the values of the variables.



#### Operators

Most usual operators are available:

```python
+ , - , * , ** , / , // , %
== , != , > , >= , < , <=
and , or , not
```

Try them and comment on their behaviour. Observe in particular what happens when you add an integer and a float.



Notice how integer division is handled by python 3:

```python
print(5/2)
print(5/2.)
```



### Strings

A string in Python can be surrounded by either single or double quotes. Use the function `print()` to see the results of the following statements:

```python
string1 = 'abc'

string2 = "def"

string3 = """aaa
bbb
ccc"""

string4 = "xxx'yyy"

string5 = 'mmm"nnn'

string6 = "aaa\nbbb\nccc"
```



### Lists

Python knows a number of compound data types, used to group together other values. The most versatile is the list, which can be written as a list of comma-separated values (items) between square brackets `[]`. List items need not all to have the same type. 

```python
A = ['spam', 'eggs', 100, 1234]
```

Define this list and print it:



The length of a list is available through the `len()` function:

```python
len(A)
```



To access the elements of the list, indexing and slicing can be used. As in C, indices start at 0:

```python
A[0] # First element

A[3] # Fourth element

A[-2] # Negative indices starts from the last element

A[1:-1] # Second until last element

A[:2] + ['bacon', 2*2] # Lists can be concatenated
```



Copying lists can cause some problems: 

```python
A = [1,2,3] # Initial list

B = A # "Copy" the list by reference 

A[0] = 9 # Change one item of the initial list
```

Now print A and B. What happens?



The solution is to use the `copy` method of lists:

```python
A = [1, 2, 3]
B = A.copy()
A[0] = 9
```



Lists are objects, with a lot of different methods (type `help(list)`):

-   `list.append(x)`: Add an item to the end of the list.
-   `list.extend(L)`: Extend the list by appending all the items in the given list.
-   `list.insert(i, x)`: Insert an item at a given position.
-   `list.remove(x)`: Remove the first item from the list whose value is x.
-   `list.pop(i)`: Remove the item at the given position in the list, and return it.
-   `list.index(x)`: Return the index in the list of the first item whose value is x.
-   `list.count(x)`: Return the number of times x appears in the list.
-   `list.sort()`: Sort the items of the list, in place.
-   `list.reverse()`: Reverse the elements of the list, in place.

Try out quickly these methods.



### Dictionaries

Another useful data type built into Python is the dictionary. Unlike lists, which are indexed by a range of numbers, dictionaries are indexed by keys, which can be any immutable type; strings and numbers can always be keys.

Dictionaries can be defined by curly braces `{}` instead of square brackets. The content is defined by `key:value` pairs:

```python
tel = {'jack': 4098, 'sape': 4139}
tel['guido'] = 4127
print(tel)
print(tel['jack'])
```

Warning: assigning a value to a new key creates the entry!



The `keys()` method of a dictionary object returns a **list** of all the keys used in the dictionary, in arbitrary order (if you want it sorted, just apply the `sorted()` function on it). 

```python
K = tel.keys()
L = sorted(K)
```

To check whether a single key is in the dictionary, use the `in` keyword:

```python
'guido' in tel
```



### Flow Control

#### If Statement 

Perhaps the most well-known conditional statement type is the `if` statement. For example:

```python
if x < 0 :
    print('x =', x, 'is negative')
elif x == 0:
    print('x =', x, 'is zero')
else:
    print('x =', x, 'is positive')
```

Give a value to the variable `x` and see what this statement does:



**Important!** The main particularity of the Python syntax is that the scope of the different structures (functions, for, if, while, etc...) is defined by the indentation.

A reasonable choice is to use four spaces for the indentation instead of tabs (configure your text editor if you are not using Jupyter).

When the scope is terminated, you need to come back at **exactly** the same level of indentation. Try this misaligned structure and observe what happens:

```python
if x < 0 :
    print('x =', x, 'is negative')
 elif x == 0:
    print('x =', x, 'is zero')
 else:
    print('x =', x, 'is positive')
```

Jupyter is nice enough to highlight it for you, but not all text editors do that...



In a if statement, here can be zero or more elif parts, and the else part is optional. What to do when the condition is true should be indented. 

The keyword `"elif"` is a shortened form of `"else if"`, and is useful to avoid excessive indentation.

An `if ... elif ... elif ...` sequence is a substitute for the switch or case statements found in other languages.

#### For Loop

The for statement in Python differs a bit from what you may be used to in C or Pascal.

Rather than always iterating over an arithmetic progression of numbers (like in Pascal), or giving the user the ability to define both the iteration step and halting condition (as C), Python's for statement iterates over the items of any sequence (a list or a string), in the order they appear in the sequence.

```python
list_words = ['cat', 'window', 'defenestrate']

for word in list_words:
    print(word, len(word))
```



If you do need to iterate over a sequence of numbers, the built-in function `range()` comes in handy. It generates lists containing arithmetic progressions:

```python
for i in range(5):
    print(i)
```



`range(N)` generates a list of N number starting from 0 until N-1.

It is possible to specify a start value (0 by default), an end value (excluded) and even a step:

```python
range(5, 10)
range(5, 10, 2)
```



To iterate over the indices of a sequence, you can combine range() and len() as follows:

```python
list_words = ['Mary', 'had', 'a', 'little', 'lamb']

for idx in range(len(list_words)):
    print(idx, list_words[idx])
```



The `enumerate()` function allows to get at the same time the index and the content:

```python
for idx, word in enumerate(list_words):
    print(idx, word)
```



### Functions

As in most procedural languages, you can define functions. Functions are defined by the keyword `def`. Only the parameters of the function are specified (without type), not the return values.

The content of the function has to be incremented as with for loops.

Return values can be specified with the `return` keywork. It is possible to return several values at the same time, separated by commas.

```python
def say_hello_to(first, second):
    question = 'Hello, I am '+ first + '!'
    answer = 'Hello '+ first + '! I am ' + second + '!'
    return question, answer

question, answer = say_hello_to('Jack', 'Gill')
```



Functions can have several parameters (with default values or not). The name of the parameter can be specified during the call, so their order won't matter.



Try to call the `cos_and_sin()` function in different ways:

```python
# import the math package
from math import *

def cos_and_sin(value, freq, phase=0):
    """
    Returns the cosine and sine functions
    of a value, given a frequency and a phase.
    """
    angle = 2*pi * freq * value + phase
    return cos(angle), sin(angle)

v = 1.7
f = 4
p = pi/2

c, s = cos_and_sin(v, f)
c, s = cos_and_sin(freq=f, value=v)
c, s = cos_and_sin(value=v, phase=p, freq=f)
```



## Exercise

In cryptography, a Caesar cipher is a very simple encryption technique in which each letter in the plain text is replaced by a letter some fixed number of positions down the alphabet. For example, with a shift of 3, A would be replaced by D, B would become E, and so on. The method is named after Julius Caesar, who used it to communicate with his generals. ROT-13 ("rotate by 13 places") is a widely used example of a Caesar cipher where the shift is 13. In Python, the key for ROT-13 may be represented by means of the following dictionary:

code = {'a':'n', 'b':'o', 'c':'p', 'd':'q', 'e':'r', 'f':'s',
        'g':'t', 'h':'u', 'i':'v', 'j':'w', 'k':'x', 'l':'y',
        'm':'z', 'n':'a', 'o':'b', 'p':'c', 'q':'d', 'r':'e',
        's':'f', 't':'g', 'u':'h', 'v':'i', 'w':'j', 'x':'k',
        'y':'l', 'z':'m', 'A':'N', 'B':'O', 'C':'P', 'D':'Q',
        'E':'R', 'F':'S', 'G':'T', 'H':'U', 'I':'V', 'J':'W',
        'K':'X', 'L':'Y', 'M':'Z', 'N':'A', 'O':'B', 'P':'C',
        'Q':'D', 'R':'E', 'S':'F', 'T':'G', 'U':'H', 'V':'I', 
        'W':'J', 'X':'K', 'Y':'L', 'Z':'M'}

Your task in this exercise is to implement an encoder/decoder of ROT-13. Once you're done, you will be able to read the following secret message:

```
BZT! guvf vf fb obevat.
```

The idea is to write a `decode()` function taking the message and the code dictionary as inputs, and returning the decoded message. It should iterate over all letters of the message and replace them with the decoded letter. If the letter is not in the dictionary, keep it as it is. 



