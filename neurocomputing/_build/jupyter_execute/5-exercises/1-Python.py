# Introduction To Python

Python is a powerful, flexible programming language widely used for scientific computing, in web/Internet development, to write desktop graphical user interfaces (GUIs), create games, and much more. It became the *de facto* standard for machine learning, with a huge variety of specialized libraries such as:

* `scikit-learn` <https://scikit-learn.org/>, a toolbox with a multitude of ML algorithms already implemented.
* `tensorflow` <https://tensorflow.org/>, an automatic differentiation library by Google for deep learning.
* `pytorch` <https://pytorch.org/>, another popular automatic differentiation library by Facebook.

Python is an high-level, interpreted, object-oriented language written in C, which means it is compiled on-the-fly, at run-time execution. Its syntax is close to C, but without prototyping (whether a variable is an integer or a string will be automatically determined by the context). It can be executed either directly in an interpreter (à la Matlab), in a script or in a notebook (as here).

The documentation on Python can be found at [http://docs.python.org](http://docs.python.org).

Many resources to learn Python exist on the Web:

-  Free book [Dive into Python](http://www.diveintopython.net/).
-  [Learn Python](https://www.learnpython.org/).
-  [Learn Python the hard way](http://learnpythonthehardway.org).
-  Learn Python on [Code academy](http://www.codecademy.com/tracks/python).
-  Scipy lectures note [http://www.scipy-lectures.org](http://www.scipy-lectures.org/)
-  An Introduction to Interactive Programming in Python on [Coursera](https://www.coursera.org/course/interactivepython).

This notebook only introduces you to the basics and skips functionalities such as classes, as we will not need them in the exercises, so feel free to study additional resources if you want to master Python programming.

## Installation

Python should be already installed if you use Linux, a very old version if you use MacOS, and probably nothing under Windows. Moreover, Python 2.7 became obsolete in December 2019 but is still the default on some distributions. 

For these reasons, we strongly recommend installing Python 3 using the Anaconda distribution:

<https://www.anaconda.com/products/individual>

Anaconda offers all the major Python packages in one place, with a focus on data science and machine learning. To install it, simply download the installer / script for your OS and follow the instructions. Beware, the installation takes quite a lot of space on the disk (around 1 GB), so choose the installation path wisely.

To install packages (for example `tensorflow`), you just have to type in a terminal:

```bash
conda install tensorflow
```

Refer to the docs (<https://docs.anaconda.com/anaconda/>) to know more. If you prefer your local Python installation, the `pip` utility allows to also install virtually any Python package:

```bash
pip install tensorflow
```

Another option is to run the notebooks in the cloud, for example on Google Colab:

<https://colab.research.google.com/>

Colab has all major ML packages already installed, so you do not have to care about anything. Under conditions, you can also use a GPU for free (but for maximally 24 hours in a row).

## Working With Python

There are basically three ways to program in Python: the interpreter for small commands, scripts for longer programs and notebooks (as here) for interactive programming.

### Python Interpreter

To start the Python interpreter, simply type its name in a terminal under Linux:

```bash
user@machine ~ $ python
```

```python
Python 3.7.4 (default, Jul 16 2019, 07:12:58) 
[GCC 9.1.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> 
```

You can then type anything at the prompt, for example a print statement:

```python
>>> print("Hello World!")
Hello World!
```

To exit Python call the `exit()` function (or `Ctrl+d`):

```python
>>> exit()
```


## Working With Python

There are basically three ways to program in Python: the interpreter for small commands, scripts for longer programs and notebooks (as here) for interactive programming.

### Python Interpreter

To start the Python interpreter, simply type its name in a terminal under Linux:

```bash
user@machine ~ $ python
```

```python
Python 3.7.4 (default, Jul 16 2019, 07:12:58) 
[GCC 9.1.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> 
```

You can then type anything at the prompt, for example a print statement:

```python
>>> print("Hello World!")
Hello World!
```

To exit Python call the `exit()` function (or `Ctrl+d`):

```python
>>> exit()
```

### Jupyter Notebooks

A third recent (but very useful) option is to use Jupyter notebooks (formerly IPython notebooks). 

Jupyter notebooks allow you to edit Python code in your browser (but also Julia, R, Scala...) and run it locally. 

To launch a Jupyter notebook, type in a terminal:

```bash
jupyter notebook
```

and create a new notebook (Python 3)

When a Jupyter notebook already exists (here `1-Python.ipynb`), you can also start it directly:

```bash
jupyter notebook 1-Python.ipynb
```

Alternatively, Jupyter lab has a more modern UI, but is still in beta.

The main particularity of notebooks is that code is not executed sequentially from the beginning to the end, but only when a **cell** is explicitly run with **Ctrl + Enter** (the active cell stays the same) or **Shift + Enter** (the next cell becomes active).

To edit a cell, select it and press **Enter** (or double-click).

**Q:** In the next cell, run the Hello World! example:



There are three types of cells:

* Python cells allow to execute Python code (the default)
* Markdown cells which allow to document the code nicely (code, equations), like the current one.
* Raw cell are passed to nbconvert directly, it allows you to generate html or pdf versions of your notebook (not used here).

**Beware that the order of execution of the cells matters!**

**Q:** In the next three cells, put the following commands:

1. `text = "Text A"`
2. `text = "Text B"`
3. `print(text)`

and run them in different orders (e.g. 1, 2, 3, 1, 3)







Executing a cell can therefore influence cells before and after it. If you want to run the notebook sequentially, select **Kernel/Restart & Run all**.

Take a moment to explore the options in the menu (Insert cells, Run cells, Download as Python, etc).

## Basics In Python

### Print Statement

In Python 3, the `print()` function is a regular function:

```python
print(value1, value2, ...)
```

You can give it as many arguments as you want (of whatever type), they will be printed one after another separated by spaces.

**Q:** Try to print "Hello World!" using two different strings "Hello" and "World!":



### Data Types

As Python is an interpreted language, variables can be assigned without specifying their type: it will be inferred at execution time.

The only thing that counts is how you initialize them and which operations you perform on them.

```python
a = 42          # Integer
b = 3.14159     # Double precision float
c = 'My string' # String
d = False       # Boolean
e = a > b       # Boolean
```

**Q:** Print these variables as well as their type:

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

**Q:** Try these assignments and print the values of the variables.



#### Operators

Most usual operators are available:

```python
+ , - , * , ** , / , // , %
== , != , <> , > , >= , < , <=
and , or , not
```

**Q:** Try them and comment on their behaviour. Observe in particular what happens when you add an integer and a float.



**Q:** Notice how integer division is handled by python 3 by dividing an integer by either an integer or a float:



### Strings

A string in Python can be surrounded by either single or double quotes (no difference as long as they match). Three double quotes allow to print new lines directly (equivalent of `\n` in C).

**Q:** Use the function `print()` to see the results of the following statements:

```python
a = 'abc'

b = "def"

c = """aaa
bbb
ccc"""

d = "xxx'yyy"

e = 'mmm"nnn'

f = "aaa\nbbb\nccc"
```



### Lists

Python knows a number of compound data types, used to group together other values. The most versatile is the list, which can be written as a list of comma-separated values (items) between square brackets `[]`. List items need not all to have the same type. 

```python
a = ['spam', 'eggs', 100, 1234]
```

**Q:** Define a list of various variables and print it:



The number of items in a list is available through the `len()` function applied to the list:

```python
len(a)
```

**Q:** Apply `len()` on the list, as well as on a string:



To access the elements of the list, indexing and slicing can be used. 

* As in C, indices start at 0, so `a[0]` is the first element of the list, `a[3]` is its fourth element. 

* Negative indices start from the end of the list, so `a[-1]` is the last element, `a[-2]` the last but one, etc.

* Slices return a list containing a subset of elements, with the form `a[start:stop]`, `stop` being excluded. `a[1:3]` returns the second and third elements. WHen omitted, `start` is 0 (`a[:2]` returns the two first elements) and `stop` is the length of the list (`a[1:]` has all elements of `a` except the first one).  

**Q:** Experiment with indexing and slicing on your list.



Copying lists can cause some problems: 

```python
a = [1,2,3] # Initial list

b = a # "Copy" the list by reference 

a[0] = 9 # Change one item of the initial list
```

**Q:** Now print `a` and `b`. What happens?



The solution is to use the built-in `copy()` method of lists:

```python
b = a.copy()
```

**Q:** Try it and observe the difference.



Lists are objects, with a lot of different built-in methods (type `help(list)` in the interpreter or in a cell):

-   `a.append(x)`: Add an item to the end of the list.
-   `a.extend(L)`: Extend the list by appending all the items in the given list.
-   `a.insert(i, x)`: Insert an item at a given position.
-   `a.remove(x)`: Remove the first item from the list whose value is x.
-   `a.pop(i)`: Remove the item at the given position in the list, and return it.
-   `a.index(x)`: Return the index in the list of the first item whose value is x.
-   `a.count(x)`: Return the number of times x appears in the list.
-   `a.sort()`: Sort the items of the list, in place.
-   `a.reverse()`: Reverse the elements of the list, in place.

**Q:** Try out quickly these methods, in particular `append()` which we will use quite often.



### Dictionaries

Another useful data type built into Python is the dictionary. Unlike lists, which are indexed by a range of numbers from 0 to length -1, dictionaries are indexed by keys, which can be any *immutable* type; strings and numbers can always be keys.

Dictionaries can be defined by curly braces `{}` instead of square brackets. The content is defined by `key:item` pairs, the item can be of any type:

```python
tel = {
    'jack': 4098, 
    'sape': 4139
}
```

To retrieve an item, simply use the key:

```python
tel_jack = tel['jack']
```

To add an entry to the dictionary, just use the key and assign a value to the item. It automatically extends the dictionary (warning, it can be dangerous!).

```python
tel['guido'] = 4127
```

**Q:** Create a dictionary and elements to it.



The `keys()` method of a dictionary object returns a **list** of all the keys used in the dictionary, in the order in which you added the keys (if you want it sorted, just apply the `sorted()` function on it). 

```python
a = tel.keys()
b = sorted(tel.keys())
```

`values()` does the same for the value of the items:

```python
c = tel.values()
```

**Q:** Do it on your dictionary.



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

**Q:** Give a value to the variable `x` and see what this statement does.



**Important!** The main particularity of the Python syntax is that the scope of the different structures (functions, for, if, while, etc...) is defined by the indentation, not by curly braces `{}`. As long as the code stays at the same level, it is in the same scope:

```python
if x < 0 :
    print('x =', x, 'is negative')
    x = -x
    print('x =', x, 'is now positive')
elif x == 0:
    print('x =', x, 'is zero')
else:
    print('x =', x, 'is positive')
```

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



In a if statement, there can be zero or more elif parts. What to do when the condition is true should be indented. The keyword `"elif"` is a shortened form of `"else if"`, and is useful to avoid excessive indentation. An `if ... elif ... elif ...` sequence is a substitute for the switch or case statements found in other languages.

The `elif` and `else` statements are optional. You can also only use the if statement alone:

```python
a = [1, 2, 0]
has_zero = False
if 0 in a:
    has_zero = True
```

Note the use of the `in` keyword to know if an element exists in a list.

#### For Loop

The for statement in Python differs a bit from what you may be used to in C, Java or Pascal.

Rather than always iterating over an arithmetic progression of numbers (like in Pascal), or giving the user the ability to define both the iteration step and halting condition (as C), Python's for statement iterates over the items of any sequence (a list or a string), in the order they appear in the sequence.

```python
list_words = ['cat', 'window', 'defenestrate']

for word in list_words:
    print(word, len(word))
```

**Q:** Iterate over the list you created previously and print each element.



If you do need to iterate over a sequence of numbers, the built-in function `range()` comes in handy. It generates lists containing arithmetic progressions:

```python
for i in range(5):
    print(i)
```

**Q:** Try it.



`range(N)` generates a list of N number starting from 0 until N-1.

It is possible to specify a start value (0 by default), an end value (excluded) and even a step:

```python
range(5, 10)
range(5, 10, 2)
```

**Q:** Print the second and fourth elements of your list (`['spam', 'eggs', 100, 1234]`) using `range()`.



To iterate over all the indices of a list (0, 1, 2, etc), you can combine range() and len() as follows:

```python
for idx in range(len(a)):
```

The `enumerate()` function allows to get at the same time the index and the content:

```python
for i, val in enumerate(a):
    print(i, val)
```



To get iteratively the keys and items of a dictionary, use the `items()` method of dictionary:

```python
for key, val in tel.items():
```

**Q:** Print one by one all keys and values of your dictionary.



### Functions

As in most procedural languages, you can define functions. Functions are defined by the keyword `def`. Only the parameters of the function are specified (without type), not the return values.

The content of the function has to be incremented as with for loops.

Return values can be specified with the `return` keywork. It is possible to return several values at the same time, separated by commas.

```python
def say_hello_to(first, second):
    question = 'Hello, I am '+ first + '!'
    answer = 'Hello '+ first + '! I am ' + second + '!'
    return question, answer
```

To call that function, pass the arguments that you need and retrieve the retruned values separated by commas.

```python
question, answer = say_hello_to('Jack', 'Gill')
```

**Q:** Test it with different names as arguments.



**Q:** Redefine the `tel` dictionary `{'jack': 4098, 'sape': 4139, 'guido': 4127}` if needed, and create a function that returns a list of names whose number is higher than 4100.



Functions can take several arguments (with default values or not). The name of the argument can be specified during the call, so their order won't matter.

**Q:** Try these different calls to the `say_hello_to()` function:

```python
question, answer = say_hello_to('Jack', 'Gill')
question, answer = say_hello_to(first='Jack', second='Gill')
question, answer = say_hello_to(second='Jack', first='Gill')
```



Default values can be specified for the last arguments, for example:

```python
def add (a, b=1):
    return a + b

x = add(2, 3) # returns 5
y = add(2) # returns 3
z = add(a=4) # returns 5
```

**Q:** Modify `say_hello_to()` so that the second argument is your own name by default.



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

**Q:** Your task in this final exercise is to implement an encoder/decoder of ROT-13. Once you're done, you will be able to read the following secret message:

```
Jnvg, jung qbrf vg unir gb qb jvgu qrrc yrneavat??
```

The idea is to write a `decode()` function taking the message and the code dictionary as inputs, and returning the decoded message. It should iterate over all letters of the message and replace them with the decoded letter. If the letter is not in the dictionary (e.g. punctuation), keep it as it is. 

