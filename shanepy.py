# Run this to update:
# py update-shanepy

# sudo python2 setup.py build -b /tmp/shanepy install --record /tmp/files.txt
# sudo python3 setup.py build -b /tmp/shanepy install --record /tmp/files.txt

from __future__ import print_function

import os
import sys
import subprocess
import pprint
import json
import json as jn
import re

import pydoc
from pydoc import locate
# locate("spacy_pytorch_transformers.language.PyTT_Language")

# Use try because I might be importing shanepy into an environment which doesn't
# have these packages
try:
    import pandas as pd
except:
    True

try:
    import numpy
    import numpy as np
except:
    True

import pickle

try:
    import scipy
except:
    True

import xml
import xml.etree.ElementTree as ET
from xml.dom import minidom

try:
    import sqlparse
except:
    True

try:
    from StringIO import StringIO
    from StringIO import StringIO as sio
except ImportError:
    from io import StringIO
    from io import StringIO as sio


from importlib import import_module
sys.path.append(os.path.expanduser("~/.ptpython/"))
ptconfig = import_module("config")
from ptpython.repl import embed

def myembed(globals, locals):
    """This embeds ptpython and honors the ptpython config"""
    os.environ["EDITOR"] = "sp"
    embed(globals, locals, configure=ptconfig.configure)

# def b(command, inputstring="", timeout=0):
#     """Runs a shell command"""
#     #print(command, file=sys.stderr)
#     p = subprocess.Popen(command, shell=True, executable="/bin/sh", stdin=subprocess.PIPE,
#                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
#     #p.stdin.write(bytearray(inputstring, 'utf-8'))
#     p.stdin.write(str(inputstring).encode('utf-8'))
#     p.stdin.close()
#     output = p.stdout.read().decode("utf-8")
#     # if (sys.version_info < (3, 0)) and isinstance(o, unicode):
#     #     output = p.stdout.read().decode("utf-8")
#     # else:
#     #     output = p.stdout.read()
#     p.wait()
#     # print(output)
#     #return [output.rstrip(), p.returncode]
#     # I don't want rstrip because my output might have trailing spaces, not just
#     # newlines
#     return [str(output), p.returncode]

import importlib

def xv(s):
    return os.path.expandvars(s)

def b(c, inputstring="", timeout=0):
    """Runs a shell c"""

    c = xv(c)

    p = subprocess.Popen(c, shell=True, executable="/bin/sh", stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
    p.stdin.write(str(inputstring).encode('utf-8'))
    p.stdin.close()
    output = p.stdout.read().decode("utf-8")
    p.wait()
    return [str(output), p.returncode]

def bsh(c, inputstring="", timeout=0):
    """Runs a shell c"""
    return b(c, inputstring, timeout)

def bash(c, inputstring="", timeout=0):
    """Runs a shell c"""
    return b(c, inputstring, timeout)

def q(inputstring=""):
    return b("q", inputstring)[0]

def ns(inputstring=""):
    return b("ns", inputstring)[0]

def umn(inputstring=""):
    return b("umn", inputstring)[0]

def mnm(inputstring=""):
    return b("mnm", inputstring)[0]

def cat(path):
    # return b("cat " + q(umn(path)))[0]
    # This is more reliable as it handles file descriptors
    # cat can't handle file descriptors because /proc/self/fd/12 has self in it
    return open(umn(path), 'r').read()


def splitlines(s):
    return s.split("\n")


def tabulate_string(s, delim="\t"):
    return b("tabulate " + q(delim), s)[0]



# joinargs(["hi", "hi yo", "hi\""])                                                                                                                                        │
def joinargs(args):
    """Like the cmd script"""
    commandstring = '';

    MyOut = subprocess.Popen(["cmd"] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
    stdout,stderr = MyOut.communicate()
    # print(stdout)
    # print(stderr)

    return stdout.decode("utf-8")


def list_imports_here():
    return splitlines(b("find . -name '*.py' -exec grep -HnP '^(class|def) ' {} \\;")[0])

def list_imports_here_pp():
    # return print(b("find . -name '*.py' -exec grep -HnP '^(class|def) ' {} \\; | tabulate :")[0])
    return print(tabulate_string(b("find . -name '*.py' -exec grep -HnP '^(class|def) ' {} \\;")[0], ":"))

def sayhi():
    print("hi")


# reload_shanepy(); from shanepy import *
def reload_shanepy():
    print(b("cr $MYGIT/mullikine/shanepy/shanepy.py")[0])

    import shanepy
    importlib.reload(shanepy)

    print("You must run this manually: \"from shanepy import *\"")
    ns("You must run this manually: \"from shanepy import *\"")


def ipy():
    """Splits the screen, and starts an ipython."""

    bash("tm -d sph -c ~ -n ipython ipython &")[0]


pp = pprint.PrettyPrinter(indent=2, width=1000, depth=20)


def ppr(o):
    if isinstance(o, xml.etree.ElementTree.ElementTree):
        print(pprs(o))
        return None

    if isinstance(o, xml.etree.ElementTree.Element):
        print(pprs(o))
        return None

    pp.pprint(o)


def pprs(o):
    if isinstance(o, xml.etree.ElementTree.ElementTree):
        root = o.getroot()
        rough_string = ET.tostring(root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="\t")

    if isinstance(o, xml.etree.ElementTree.Element):
        rough_string = ET.tostring(o, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="\t")

def pretty_sql(sql):
    #  import bpython; bpython.embed(locals_=dict(globals(), **locals()))
    return sqlparse.format(sql, reindent=True, keyword_case='upper')

def k(o):
    """
    Pickle an object then put it into a vim.

    :param o: The object to be opened in tmux
    """

    if o is None:
        return None

    try:
        import tempfile
        pickle_file = tempfile.NamedTemporaryFile('wb', suffix='.pickle', delete=False)

        #return bash("tm -d nw 'vim -'", str(pickle.dumps(o)))
        pickle.dump(o,pickle_file)
        #return bash("tmux-temp-vim.sh -e pickle", str())

        return bash("tm -d nw \"vim \\\"" + pickle_file.name + "\\\"\"")
    except:
        pass

if (sys.version_info > (3, 0)):
    # Even with higher versions, past might not be available
    try:
        from past.builtins import execfile
    except:
        True

def source_file(fp):
    """Directly includes python source code."""

    sys.stdout = open(os.devnull, 'w')
    execfile(fp)
    sys.stdout = sys.__stdout__


#  if (sys.version_info > (3, 0)):
    #  import shanepy3
    #  from shanepy3 import *
    #  source_file("shanepy3.py")

def read_csv_smart(*args, **kwargs):
    try:
        # How do I make this code run on python2?
        # Default values must go before args and kwargs

        #  return pd.read_csv(*args, **kwargs, dtype={'SysDefaultVal': object})
        # It's not possible to ignore this python3 syntax when running under
        # python2 unless I conditionally include the module.
        # https://stackoverflow.com/questions/32482502/how-can-i-ignore-python-3-syntax-when-running-script-under-python-2
        return pd.read_csv(dtype={'SysDefaultVal': np.float32}, *args, **kwargs)
    except:
        pass

    try:
        #return pd.read_csv(*args, **kwargs, encoding='latin1', dtype={'SysDefaultVal': object})
        return pd.read_csv(dtype={'SysDefaultVal': np.float32}, encoding='latin1', *args, **kwargs)
    except:
        pass


def my_to_csv(*args, **kwargs):
    return pd.DataFrame.to_csv(na_rep="", *args, **kwargs)


def catin():
    """Get stdin as a string"""
    import sys
    return sys.stdin.read()

def T(list_of_lists):
    """Transposes a list of lists"""

    import six
    return list(map(list, six.moves.zip_longest(*list_of_lists, fillvalue=' ')))

def list_of_lists_to_text(arg):
    """Saves a list of lists to a text file"""

    return "\n".join([''.join(l) for l in arg])

def s(o, fp):
    """Save object repr to path"""

    with open(fp, 'w') as f:
        f.write(o)

def p(o):
    """Save object repr to stdout"""

    # sys.stdout.write(o)

    print(o,end="")

def xc(o):
    b("xc -i", p(o))

def ftf(s):
    """Creates a temporary file from a string and returns the path"""

    import tempfile
    f = tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False)
    f.seek(0)  # start from the beginning of the file
    f.write(s)
    return f.name

#  if hasattr(type(o), 'readlines') and callable(getattr(type(o), 'readlines')):
#  def strip_list(l):
#  return l[:-1] if l[-1] == '\n' else l
#  return [strip_list(list(line)) for line in o.readlines()]

def l(fp):
    return [line.rstrip('\n') for line in open(fp)]

def o(fp):
    """
    Opens the file given by the path into a python object.

    :param str fp: The path of the file
    """

    fp = xv(fp)

    #  , dtype={'ID': object}

    import re

    if re.match(r'.*\.npy', fp) is not None:
        import numpy as np
        try:
            ret = np.load(fp)
            return ret
        except:
            pass

        try:
            ret = np.load(fp, encoding="latin1")
            return ret
        except:
            pass

        try:
            ret = np.load(fp, encoding="bytes")
            return ret
        except:
            pass

    if re.match(r'.*\.xml', fp) is not None:
        import pandas as pd
        ret = read_csv_smart(fp)
        #  sys.stdout.write(str(type(ret)))
        return ret

    if re.match(r'.*\.csv', fp) is not None:
        import pandas as pd
        ret = read_csv_smart(fp)
        #  sys.stdout.write(str(type(ret)))
        return ret

    if re.match(r'.*\.xls', fp) is not None:
        import pandas as pd
        ret = pd.read_excel(fp)
        #  sys.stdout.write(str(type(ret)))
        return ret

    if (re.match(r'.*\.pickle$', fp) is not None or re.match(r'.*\.p$', fp) is not None):
        import pickle

        with open(fp, 'rUb') as f:
            data = f.read()

        ret = pickle.loads(data)
        return ret

    #  import pandas as pd
    with open(fp, 'rU') as f:
        def strip_list(l):
            return l[:-1] if l[-1] == '\n' else l

        return [strip_list(list(line)) for line in f.readlines()]

    #  ret = pd.DataFrame(matrix)

def r(o):
    """
    Splits the terminal, and runs a program on the string representation of the object, depending on its type, such as visidata for a DataFrame.

    :param o: The object to be opened in tmux
    """

    sys.stdout.write(str(type(o)))

    # python 2 only
    if (sys.version_info < (3, 0)) and isinstance(o, unicode):
        return bash("tnw 'fpvd'", o.decode("utf-8"))

    #elif isinstance(o, pd.core.series.Series):
    #    return bash("tnw 'fpvd'", pd.Series(o).to_csv(index=False))

    if isinstance(o, pd.core.frame.DataFrame):
        return bash("tnw fpvd", o.to_csv(index=False))
    elif isinstance(o, set):
        # For a set of tuples
        # Say, created like this:
        # diff = set(zip(df1.CDID, df1.ElementName)) - set(zip(df2.CDID, df2.ElementName))
        return bash("tnw fpvd", pd.DataFrame(dict(o)).to_csv(index=False))
    elif isinstance(o, tuple):
        return bash("tnw fpvd", pd.DataFrame(list(o)).to_csv(index=False))
    elif isinstance(o, numpy.ndarray):
        return bash("tnw fpvd", pd.DataFrame(o).to_csv(index=False))
    elif isinstance(o, set):
        return bash("tnw fpvd", pd.DataFrame(o).to_csv(index=False))
    elif isinstance(o, list):
        return bash("tnw fpvd", pd.DataFrame(o).to_csv(index=False))
    elif hasattr(type(o), 'to_csv') and callable(getattr(type(o), 'to_csv')):
        return bash("tnw fpvd", o.to_csv(index=False))
    elif isinstance(o, dict):
        bash("dict")
        return bash("tnw v", ppr(o))
    elif isinstance(o, str):
        return bash("tnw v", o)
    elif isinstance(o, xml.etree.ElementTree.ElementTree) or isinstance(o, xml.etree.ElementTree.Element):
        return bash("tnw v", pprs(o))
    elif o is None:
        pass
    else:
        try:
            return bash("tm -d -tout nw 'v'", pickle.dumps(o))
        except:
            pass

def v(o):
    """
    Splits the terminal, and runs a program on the string representation of the object, depending on its type, such as visidata for a DataFrame.

    :param o: The object to be opened in tmux
    """

    sys.stdout.write(str(type(o)))

    # python 2 only
    if (sys.version_info < (3, 0)) and isinstance(o, unicode):
        return bash("tm -d nw 'vlf'", o.decode("utf-8"))

    #elif isinstance(o, pd.core.series.Series):
    #    return bash("tm -d nw 'vim -'", pd.Series(o).to_csv(index=False))

    if isinstance(o, pd.core.frame.DataFrame):
        return bash("tm -d nw 'vlf'", o.to_csv(index=False))
    elif isinstance(o, set):
        # For a set of tuples
        # Say, created like this:
        # diff = set(zip(df1.CDID, df1.ElementName)) - set(zip(df2.CDID, df2.ElementName))
        return bash("tm -d nw 'vlf'", pd.DataFrame(dict(o)).to_csv(index=False))
    elif isinstance(o, tuple):
        return bash("tm -d nw 'vlf'", pd.DataFrame(list(o)).to_csv(index=False))
    elif isinstance(o, numpy.ndarray):
        return bash("tm -d nw 'vlf'", pd.DataFrame(o).to_csv(index=False))
    elif isinstance(o, set):
        return bash("tm -d nw 'vlf'", pd.DataFrame(o).to_csv(index=False))
    elif isinstance(o, list):
        return bash("tm -d nw 'vlf'", pd.DataFrame(o).to_csv(index=False))
    elif hasattr(type(o), 'to_csv') and callable(getattr(type(o), 'to_csv')):
        return bash("tm -d nw 'vlf'", o.to_csv(index=False))
    elif isinstance(o, dict):
        bash("dict")
        return bash("tm -d nw 'vlf'", ppr(o))
    elif isinstance(o, str):
        return bash("tm -d nw 'vlf'", o)
    elif isinstance(o, xml.etree.ElementTree.ElementTree) or isinstance(o, xml.etree.ElementTree.Element):
        return bash("tm -d nw 'vlf'", pprs(o))
    elif o is None:
        pass
    else:
        try:
            return bash("tm -d nw 'vlf'", pickle.dumps(o))
        except:
            pass

# This is actually a kinda stupid command -- unless I use it frequently.
def c(command):
    """Run a bash command and open the result in fpvd."""

    lf=pd.DataFrame(bash(command)[0].splitlines())
    r(lf)

def bpy():
    """Splits the screen, and starts a bpython."""

    bash("tm -d sph -c ~ -n bpython bpython &")[0].decode("utf-8")

def t(o):
    """Alias for type()"""

    return type(o)

def env(s):
    """Alias for os.environ()"""

    return os.environ[s]

# Fully qualified type name
def full_type_name(o):
  # o.__module__ + "." + o.__class__.__qualname__ is an example in
  # this context of H.L. Mencken's "neat, plausible, and wrong."
  # Python makes no guarantees as to whether the __module__ special
  # attribute is defined, so we take a more circumspect approach.
  # Alas, the module name is explicitly excluded from __qualname__
  # in Python 3.

  module = o.__class__.__module__
  if module is None or module == str.__class__.__module__:
    return o.__class__.__name__  # Avoid reporting __builtin__
  else:
    return module + '.' + o.__class__.__name__

def ts(o):
    """type() string name"""

    # return type(o).__name__
    return full_type_name(o)

def pdl(s):
    """get type from string"""

    return pydoc.locate(s)


# Get the path of the thing -- could be a method
# lm is a bad name:
# - list-modules?
# - list-methods of a module?
# - locate-method?
# - locate-module?
# def lm(o):
#     """Get the path of the thing"""
#     return inspect.getsourcefile(o)

def pwd():
    """Just runs bash pwd"""

    #  return bash("pwd")[0].rstrip()
    return bash("pwd")[0].rstrip('\n') # Only strip newlines


def ls():
    """Just runs bash ls"""

    return bash("ls")[0].rstrip('\n').splitlines()


def lsautofiles(dir=None):
    """Just runs bash lsautofiles"""

    if dir:
        return bash("lsautofiles \"" + dir + "\"")[0].splitlines()
    else:
        return bash("lsautofiles")[0].splitlines()


def find():
    """Just runs bash find"""

    return bash("f find-all-no-git")[0].splitlines()


def dirinfo():
    """Just runs bash dirinfo"""

    return bash("u dirinfo")[0]


def tv(input=""):
    """Puts the output into vim"""

    return bash("tv",input)[0]

def tsp():
    """Just runs bash tmux split pane"""

    return bash("tm -te -d sph")[0]

def cipe(cmd="", ins=""):
    """vipe but any command"""
    if not cmd:
        cmd = "vipe"

    return bash("tm spv " + q(cmd), ins)[0]

# smart
def spv(cmd="", ins="", has_output=False):
    """Just runs bash tmux split pane"""

    return bash("tm -S -tout spv " + q(cmd), ins)[0]

# smart
def sph(cmd="", ins="", has_output=False):
    """Just runs bash tmux split pane"""

    return bash("tm -S -tout sph " + q(cmd), ins)[0]

def spvi(cmd="", ins=""):
    """spv. takes stdin. has tty out"""

    return bash("tm -S -tout spv " + q(cmd), ins)[0]

def sphi(cmd="", ins=""):
    """sph. takes stdin. has tty out"""

    return bash("tm -S -tout sph " + q(cmd), ins)[0]

def spvio(cmd="", ins=""):
    """spv. takes stdin. returns stdout"""

    return bash("tm spv " + q(cmd) + " | cat", ins)[0]

def sphio(cmd="", ins=""):
    """sph. takes stdin. returns stdout"""

    return bash("tm sph " + q(cmd) + " | cat", ins)[0]

def vipe(ins=""):
    """vipe"""

    # return bash("tm vipe", ins)[0]
    return spvio("vipe", ins)

def spvd(cmd=""):
    """Just runs bash tmux split pane"""

    bash("tm -te -d spv " + q(cmd))[0]
    return None

def sphd(cmd=""):
    """Just runs bash tmux split pane"""

    bash("tm -te -d sph " + q(cmd))[0]
    return None

def fish():
    """Just runs fish in a tmux split pane"""

    return bash("tm -te -d sph fish")[0]


def zsh():
    """Just runs zsh in a tmux split pane"""

    return bash("tm -te -d sph")[0]


def sh():
    """Just runs bash in a tmux split pane"""

    return bash("tm -te -d sph bash")[0]


def tcp():
    """Just runs bash tmux capture"""

    return bash("tm -te -d capture -clean -editor ec")[0]


def sl():
    """Opens locals() in pvd"""

    r(locals())


def rl():
    """Opens locals() in pvd"""

    sl()


def map_funcs(obj, func_list):
    return [func(obj) for func in func_list]


def get_stats_dataframe_for_series(o):
    df=pd.DataFrame(o)

    fl=[pd.Series.std, pd.Series.median, pd.Series.mean, pd.Series.max, pd.Series.min]

    for c in df.columns.tolist():
        dti=list(zip(list(map(lambda f: f.__name__, fl)), map_funcs(c, fl)))
        dfi=pd.DataFrame(dti)

    return dfi

# I want this function to return a table of statistics where the column names
# are things like std, mean, etc. and the row indices/keys are the column names
# of the original dataframe together with the dataframe's name etc. "df1.CDID",
# df2.CDID
#def get_stats_dataframe_for_dataframe(df):
#    for c in df.columns.tolist():
#        dti=list(zip(list(map(lambda f: f.__name__, fl)), map_funcs(c, fl)))
#        dfi=pd.DataFrame(dti)
#
#
#    return dfi


def show_stats(o):
    #if isinstance(o, pd.core.frame.DataFrame):
    #    return bash("tm -d nw 'fpvd'", o.to_csv(index=False))
    #else:

    try:
        df=pd.DataFrame(o)
    except ValueError:
        print("Can't be converted to DataFrame")
        return None

    fl=[pd.Series.std, pd.Series.median, pd.Series.mean, pd.Series.max, pd.Series.min]

    for c in df.columns.tolist():
        try:
            dti=list(zip(list(map(lambda f: f.__name__, fl)), map_funcs(df[c], fl)))
            dfi=pd.DataFrame(dti)
            print(c)
            # This isn't very nicely formatted
            #  print(dfi.to_string(index=False, header=False))
            ppr(dfi)
            #  print(dfi.to_csv(index=False, header=False))
            print("\n")
        except:
            pass

    #return bash("tm -d nw 'fpvd'", dfi.to_csv(header=False))
    #return bash("tm -d nw 'fpvd'", dfi.to_csv(index=False))

#  pp.pprint({})

#  print json.dumps(dict, sort_keys=True, indent=4)

#  result=bash("get-rtm-list.sh")
#  rtmlist = result[0].split('\n')



import time
import threading
from functools import wraps

def rate_limited(max_per_second, mode='wait', delay_first_call=False):
    """
    Decorator that make functions not be called faster than

    set mode to 'kill' to just ignore requests that are faster than the
    rate.

    set delay_first_call to True to delay the first call as well
    """
    lock = threading.Lock()
    min_interval = 1.0 / float(max_per_second)
    def decorate(func):
        last_time_called = [0.0]
        @wraps(func)
        def rate_limited_function(*args, **kwargs):
            def run_func():
                lock.release()
                ret = func(*args, **kwargs)
                last_time_called[0] = time.perf_counter()
                return ret
            lock.acquire()
            elapsed = time.perf_counter() - last_time_called[0]
            left_to_wait = min_interval - elapsed
            if delay_first_call:
                if left_to_wait > 0:
                    if mode == 'wait':
                        time.sleep(left_to_wait)
                        return run_func()
                    elif mode == 'kill':
                        lock.release()
                        return
                else:
                    return run_func()
            else:
                # Allows the first call to not have to wait
                if not last_time_called[0] or elapsed > min_interval:
                    return run_func()
                elif left_to_wait > 0:
                    if mode == 'wait':
                        time.sleep(left_to_wait)
                        return run_func()
                    elif mode == 'kill':
                        lock.release()
                        return
        return rate_limited_function
    return decorate

def make_unicode(input):
    if type(input) != unicode:
        input =  input.decode('utf-8')
        return input
    else:
        return input


def exhaust_properties(o):
    """enumerate properties (for finding methods)"""

    methods = [method_name for method_name in dir(o)
                  if callable(getattr(o, method_name))]

    return "\n".join(methods)
    # for name in methods:
    #     print(name)


def list_methods(o):
    return exhaust_properties(o)
def list_methods_and_classes(o):
    return exhaust_properties(o)
def list_children(o):
    return exhaust_properties(o)
def ep(o):
    return exhaust_properties(o)
def m(o):
    """Members"""
    return exhaust_properties(o)


import inspect
from inspect import ismodule

# enumerate submodules
def em(o):
    """enumerate properties (for finding submodules)"""

    modules = [module_name for module_name in dir(o)
                  if ismodule(getattr(o, module_name))]

    return "\n".join(modules)

import os, sys

GLOBAL_INDENT=0

def wi(*args):
   """ Function to print lines indented according to level """

   if GLOBAL_INDENT: print(' '*GLOBAL_INDENT),
   for arg in args: print(arg),
   print()

def global_indent():
   """ Increase indentation """

   global GLOBAL_INDENT
   GLOBAL_INDENT += 4

def global_dedent():
   """ Decrease indentation """

   global GLOBAL_INDENT
   GLOBAL_INDENT -= 4

def describe_builtin(obj):
   """ Describe a builtin function """

   wi('+Built-in Function: %s' % obj.__name__)
   # Built-in functions cannot be inspected by
   # inspect.getargspec. We have to try and parse
   # the __doc__ attribute of the function.
   docstr = obj.__doc__
   args = ''

   if docstr:
      items = docstr.split('\n')
      if items:
         func_descr = items[0]
         s = func_descr.replace(obj.__name__,'')
         idx1 = s.find('(')
         idx2 = s.find(')',idx1)
         if idx1 != -1 and idx2 != -1 and (idx2>idx1+1):
            args = s[idx1+1:idx2]
            wi('\t-Method Arguments:', args)

   if args=='':
      wi('\t-Method Arguments: None')

   print

def describe_func(obj, method=False):
   """ Describe the function object passed as argument.
   If this is a method object, the second argument will
   be passed as True """

   if method:
      wi('+Method: %s' % obj.__name__)
   else:
      wi('+Function: %s' % obj.__name__)

   try:
       arginfo = inspect.getargspec(obj)
   except TypeError:
      print
      return

   args = arginfo[0]
   argsvar = arginfo[1]

   if args:
       if args[0] == 'self':
           wi('\t%s is an instance method' % obj.__name__)
           args.pop(0)

       wi('\t-Method Arguments:', args)

       if arginfo[3]:
           dl = len(arginfo[3])
           al = len(args)
           defargs = args[al-dl:al]
           wi('\t--Default arguments:',zip(defargs, arginfo[3]))

   if arginfo[1]:
       wi('\t-Positional Args Param: %s' % arginfo[1])
   if arginfo[2]:
       wi('\t-Keyword Args Param: %s' % arginfo[2])

   print

def describe_klass(obj):
   """ Describe the class object passed as argument,
   including its methods """

   wi('+Class: %s' % obj.__name__)

   global_indent()

   count = 0

   for name in obj.__dict__:
       item = getattr(obj, name)
       if inspect.ismethod(item):
           count += 1;describe_func(item, True)

   if count==0:
      wi('(No members)')

   global_dedent()
   print


def describe_module(module):
   """ Describe the module object passed as argument
   including its classes and functions """

   wi('[Module: %s]\n' % module.__name__)

   global_indent()

   count = 0

   for name in dir(module):
       obj = getattr(module, name)
       if inspect.isclass(obj):
          count += 1; describe_klass(obj)
       elif (inspect.ismethod(obj) or inspect.isfunction(obj)):
          count +=1 ; describe_func(obj)
       elif inspect.isbuiltin(obj):
          count += 1; describe_builtin(obj)

   if count==0:
      wi('(No members)')

   global_dedent()

#def describe_object(o):
#    """
#    Describe an object
#    """
#
#    if isinstance(o, numpy.ndarray):
#        return scipy.stats.describe(o)
#
#    ppr(o)

import scipy

def describe_ndarray(a):
    try:
        return scipy.stats.describe
    except:
        pass
        return None

def d(obj):
    """
    Describe an object
    """

    print(type(obj))

    switchDict = {
        "module": describe_module,
        "type": describe_klass,
        "function": describe_func,
        "builtin_function_or_method": describe_builtin,
        "numpy.ndarray": describe_ndarray
    }

    try:
        switchDict[type(obj).__name__](obj)
    except KeyError:
        ppr(obj)


def mygetsourcefile(thing):
    path = ""

    try:
        path = inspect.getsourcefile(thing)
    except:
        pass

    if not path:
        print("source code not available")
        return None
    else:
        return path


def pathof(thing):
    """
    Describe a thing
    """

    ppr(thing)

    # print(type(thing).__name__)

    switchDict = {
        "module": lambda x: x.__file__,
        "type": lambda x: mygetsourcefile(x),
        "function": lambda x: mygetsourcefile(x),
        "method": lambda x: mygetsourcefile(x),
        "builtin_function_or_method": None
    }

    try:
        return switchDict[type(thing).__name__](thing)
    except:
        # It might be an object
        return mygetsourcefile(type(thing))


# Get the path of the type of the thing
def lt(th):
    """ Get the path of the type of the thing"""
    return pathof(th)

# Get the path of the type of the thing
def po(th):
    """ Get the path of the type of the thing"""
    return pathof(th)

# alias
def pathoftypeof(obj):
    return pathof(obj)

def version():
    print(sys.version_info)
    return sys.version_info

import shlex
def py_q(s):
    return shlex.quote(s)

import django

def getenv(varname):
    """
    gets an environment variable
    """
    return os.environ.get(varname)

# getenv("HOME")



def sy__get_dep_path(span1, span2):
    import spacy
    assert span1.sent == span2.sent, "sent1: {}, span1: {}, sent2: {}, span2: {}".format(span1.sent, span1, span2.sent, span2)

    up = []
    down = []

    head = span1[0]
    while head.dep_ != 'ROOT':
        up.append(head)
        head = head.head
    up.append(head)

    head = span2[0]
    while head.dep_ != 'ROOT':
        down.append(head)
        head = head.head
    down.append(head)
    down.reverse()

    for n1, t1 in enumerate(up):
        for n2, t2 in enumerate(down):
            if t1 == t2:
                return ["{}::{}".format(u.dep_, 'up') for u in up[1:n1]] + ["{}::{}".format(d.dep_, 'down') for d in down[n2:]]


def i(e):
    "Exhaust an enumberable"
    return [i for i in e]


from tabulate import tabulate

def sleep(s):
    time.sleep(s)