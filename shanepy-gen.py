from hy.core.language import count, name
from __future__ import print_function


class Py2HyReturnException(Exception):

    def __init__(self, retvalue):
        self.retvalue = retvalue
        return None


import os
import sys
import subprocess
import pprint
import json
import json as jn
import pydoc
from pydoc import locate
try:
    import pandas as pd
    _hy_anon_var_1 = None
except Py2HyReturnException as e:
    raise e
    _hy_anon_var_1 = None
except:
    _hy_anon_var_1 = True
try:
    import numpy
    import numpy as np
    _hy_anon_var_2 = None
except Py2HyReturnException as e:
    raise e
    _hy_anon_var_2 = None
except:
    _hy_anon_var_2 = True
import pickle
try:
    import scipy
    _hy_anon_var_3 = None
except Py2HyReturnException as e:
    raise e
    _hy_anon_var_3 = None
except:
    _hy_anon_var_3 = True
import xml
import xml.etree.ElementTree as ET
from xml.dom import minidom
try:
    import sqlparse
    _hy_anon_var_4 = None
except Py2HyReturnException as e:
    raise e
    _hy_anon_var_4 = None
except:
    _hy_anon_var_4 = True
try:
    from StringIO import StringIO
    from StringIO import StringIO as sio
    _hy_anon_var_5 = None
except Py2HyReturnException as e:
    raise e
    _hy_anon_var_5 = None
except ImportError:
    from io import StringIO
    from io import StringIO as sio
    _hy_anon_var_5 = None
import importlib


def xv(s):
    return os.path.expandvars(s)


def b(c, inputstring='', timeout=0):
    """Runs a shell c"""
    c = xv(c)
    p = subprocess.Popen(c, shell=True, executable='/bin/sh', stdin=
        subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        close_fds=True)
    p.stdin.write(str(inputstring).encode('utf-8'))
    p.stdin.close()
    output = p.stdout.read().decode('utf-8')
    p.wait()
    return [str(output), p.returncode]


def bsh(c, inputstring='', timeout=0):
    """Runs a shell c"""
    return b(c, inputstring, timeout)


def bash(c, inputstring='', timeout=0):
    """Runs a shell c"""
    return b(c, inputstring, timeout)


def q(inputstring=''):
    return b('q', inputstring)[0]


def ns(inputstring=''):
    return b('ns', inputstring)[0]


def umn(inputstring=''):
    return b('umn', inputstring)[0]


def mnm(inputstring=''):
    return b('mnm', inputstring)[0]


def cat(path):
    return b('cat ' + q(umn(path)))[0]


def reload_shanepy():
    print(b('cr $MYGIT/mullikine/shanepy/shanepy.py')[0])
    import shanepy
    importlib.reload(shanepy)
    print('You must run this manually: "from shanepy import *"')
    return ns('You must run this manually: "from shanepy import *"')


def ipy():
    """Splits the screen, and starts an ipython."""
    return bash('tm -d sph -c ~ -n ipython ipython &')[0]


pp = pprint.PrettyPrinter(indent=2, width=1000, depth=20)


def ppr(o):
    try:
        if isinstance(o, xml.etree.ElementTree.ElementTree):
            print(pprs(o))
            raise Py2HyReturnException(None)
            _hy_anon_var_6 = None
        else:
            _hy_anon_var_6 = None
        if isinstance(o, xml.etree.ElementTree.Element):
            print(pprs(o))
            raise Py2HyReturnException(None)
            _hy_anon_var_7 = None
        else:
            _hy_anon_var_7 = None
        _hy_anon_var_8 = pp.pprint(o)
    except Py2HyReturnException as e:
        _hy_anon_var_8 = e.retvalue
    return _hy_anon_var_8


def pprs(o):
    try:
        if isinstance(o, xml.etree.ElementTree.ElementTree):
            root = o.getroot()
            rough_string = ET.tostring(root, 'utf-8')
            reparsed = minidom.parseString(rough_string)
            raise Py2HyReturnException(reparsed.toprettyxml(indent='\t'))
            _hy_anon_var_9 = None
        else:
            _hy_anon_var_9 = None
        if isinstance(o, xml.etree.ElementTree.Element):
            rough_string = ET.tostring(o, 'utf-8')
            reparsed = minidom.parseString(rough_string)
            raise Py2HyReturnException(reparsed.toprettyxml(indent='\t'))
            _hy_anon_var_10 = None
        else:
            _hy_anon_var_10 = None
        _hy_anon_var_11 = _hy_anon_var_10
    except Py2HyReturnException as e:
        _hy_anon_var_11 = e.retvalue
    return _hy_anon_var_11


def pretty_sql(sql):
    return sqlparse.format(sql, reindent=True, keyword_case='upper')


def k(o):
    """
    Pickle an object then put it into a vim.

    :param o: The object to be opened in tmux
    """
    try:
        if o is None:
            raise Py2HyReturnException(None)
            _hy_anon_var_12 = None
        else:
            _hy_anon_var_12 = None
        try:
            import tempfile
            pickle_file = tempfile.NamedTemporaryFile('wb', suffix=
                '.pickle', delete=False)
            pickle.dump(o, pickle_file)
            raise Py2HyReturnException(bash('tm -d nw "vim \\"' +
                pickle_file.name + '\\""'))
            _hy_anon_var_13 = None
        except Py2HyReturnException as e:
            raise e
            _hy_anon_var_13 = None
        except:
            _hy_anon_var_13 = None
        _hy_anon_var_14 = _hy_anon_var_13
    except Py2HyReturnException as e:
        _hy_anon_var_14 = e.retvalue
    return _hy_anon_var_14


if sys.version_info > (3, 0):
    try:
        from past.builtins import execfile
        _hy_anon_var_15 = None
    except Py2HyReturnException as e:
        raise e
        _hy_anon_var_15 = None
    except:
        _hy_anon_var_15 = True
    _hy_anon_var_16 = _hy_anon_var_15
else:
    _hy_anon_var_16 = None


def source_file(fp):
    """Directly includes python source code."""
    sys.stdout = open(os.devnull, 'w')
    execfile(fp)
    sys.stdout = sys.__stdout__


def read_csv_smart(*args, **kwargs):
    try:
        try:
            raise Py2HyReturnException(pd.read_csv(unpack_iterable(args),
                unpack_mapping(kwargs)))
            _hy_anon_var_17 = None
        except Py2HyReturnException as e:
            raise e
            _hy_anon_var_17 = None
        except:
            _hy_anon_var_17 = None
        try:
            raise Py2HyReturnException(pd.read_csv(unpack_iterable(args),
                unpack_mapping(kwargs)))
            _hy_anon_var_18 = None
        except Py2HyReturnException as e:
            raise e
            _hy_anon_var_18 = None
        except:
            _hy_anon_var_18 = None
        _hy_anon_var_19 = _hy_anon_var_18
    except Py2HyReturnException as e:
        _hy_anon_var_19 = e.retvalue
    return _hy_anon_var_19


def my_to_csv(*args, **kwargs):
    return pd.DataFrame.to_csv(unpack_iterable(args), unpack_mapping(kwargs))


def i():
    """Get stdin as a string"""
    import sys
    return sys.stdin.read()


def T(list_of_lists):
    """Transposes a list of lists"""
    import six
    return list(map(list, six.moves.zip_longest(unpack_iterable(
        list_of_lists), fillvalue=' ')))


def list_of_lists_to_text(arg):
    """Saves a list of lists to a text file"""
    return '\n'.join([''.join(l) for l in arg])


def s(o, fp):
    """Save object repr to path"""
    with open(fp, 'w') as f:
        _hy_anon_var_20 = f.write(o)
    return _hy_anon_var_20


def p(o):
    """Save object repr to stdout"""
    return print(o, end='')


def xc(o):
    return b('xc -i', p(o))


def ftf(s):
    """Creates a temporary file from a string and returns the path"""
    import tempfile
    f = tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False)
    f.seek(0)
    f.write(s)
    return f.name


def l(fp):
    return [line.rstrip('\n') for line in open(fp)]


def o(fp):
    """
    Opens the file given by the path into a python object.

    :param str fp: The path of the file
    """
    try:
        fp = xv(fp)
        import re
        if re.match('.*\\.npy', fp) is not None:
            import numpy as np
            try:
                ret = np.load(fp)
                raise Py2HyReturnException(ret)
                _hy_anon_var_21 = None
            except Py2HyReturnException as e:
                raise e
                _hy_anon_var_21 = None
            except:
                _hy_anon_var_21 = None
            try:
                ret = np.load(fp, encoding='latin1')
                raise Py2HyReturnException(ret)
                _hy_anon_var_22 = None
            except Py2HyReturnException as e:
                raise e
                _hy_anon_var_22 = None
            except:
                _hy_anon_var_22 = None
            try:
                ret = np.load(fp, encoding='bytes')
                raise Py2HyReturnException(ret)
                _hy_anon_var_23 = None
            except Py2HyReturnException as e:
                raise e
                _hy_anon_var_23 = None
            except:
                _hy_anon_var_23 = None
            _hy_anon_var_24 = _hy_anon_var_23
        else:
            _hy_anon_var_24 = None
        if re.match('.*\\.xml', fp) is not None:
            import pandas as pd
            ret = read_csv_smart(fp)
            raise Py2HyReturnException(ret)
            _hy_anon_var_25 = None
        else:
            _hy_anon_var_25 = None
        if re.match('.*\\.csv', fp) is not None:
            import pandas as pd
            ret = read_csv_smart(fp)
            raise Py2HyReturnException(ret)
            _hy_anon_var_26 = None
        else:
            _hy_anon_var_26 = None
        if re.match('.*\\.xls', fp) is not None:
            import pandas as pd
            ret = pd.read_excel(fp)
            raise Py2HyReturnException(ret)
            _hy_anon_var_27 = None
        else:
            _hy_anon_var_27 = None
        if re.match('.*\\.pickle$', fp) is not None or re.match('.*\\.p$', fp
            ) is not None:
            import pickle
            with open(fp, 'rUb') as f:
                data = f.read()
                _hy_anon_var_28 = None
            ret = pickle.loads(data)
            raise Py2HyReturnException(ret)
            _hy_anon_var_29 = None
        else:
            _hy_anon_var_29 = None
        with open(fp, 'rU') as f:

            def strip_list(l):
                return l[slice(None, -1, None)] if l[-1] == '\n' else l
            raise Py2HyReturnException([strip_list(list(line)) for line in
                f.readlines()])
            _hy_anon_var_30 = None
        _hy_anon_var_31 = _hy_anon_var_30
    except Py2HyReturnException as e:
        _hy_anon_var_31 = e.retvalue
    return _hy_anon_var_31


def r(o):
    """
    Splits the terminal, and runs a program on the string representation of the object, depending on its type, such as visidata for a DataFrame.

    :param o: The object to be opened in tmux
    """
    try:
        sys.stdout.write(str(type(o)))
        if sys.version_info < (3, 0) and isinstance(o, unicode):
            raise Py2HyReturnException(bash("tnw 'fpvd'", o.decode('utf-8')))
            _hy_anon_var_32 = None
        else:
            _hy_anon_var_32 = None
        if isinstance(o, pd.core.frame.DataFrame):
            raise Py2HyReturnException(bash('tnw fpvd', o.to_csv(index=False)))
            _hy_anon_var_44 = None
        else:
            if isinstance(o, set):
                raise Py2HyReturnException(bash('tnw fpvd', pd.DataFrame(
                    dict(o)).to_csv(index=False)))
                _hy_anon_var_43 = None
            else:
                if isinstance(o, tuple):
                    raise Py2HyReturnException(bash('tnw fpvd', pd.
                        DataFrame(list(o)).to_csv(index=False)))
                    _hy_anon_var_42 = None
                else:
                    if isinstance(o, numpy.ndarray):
                        raise Py2HyReturnException(bash('tnw fpvd', pd.
                            DataFrame(o).to_csv(index=False)))
                        _hy_anon_var_41 = None
                    else:
                        if isinstance(o, set):
                            raise Py2HyReturnException(bash('tnw fpvd', pd.
                                DataFrame(o).to_csv(index=False)))
                            _hy_anon_var_40 = None
                        else:
                            if isinstance(o, list):
                                raise Py2HyReturnException(bash('tnw fpvd',
                                    pd.DataFrame(o).to_csv(index=False)))
                                _hy_anon_var_39 = None
                            else:
                                if hasattr(type(o), 'to_csv') and callable(
                                    getattr(type(o), 'to_csv')):
                                    raise Py2HyReturnException(bash(
                                        'tnw fpvd', o.to_csv(index=False)))
                                    _hy_anon_var_38 = None
                                else:
                                    if isinstance(o, dict):
                                        bash('dict')
                                        raise Py2HyReturnException(bash('tnw v',
                                            ppr(o)))
                                        _hy_anon_var_37 = None
                                    else:
                                        if isinstance(o, str):
                                            raise Py2HyReturnException(bash('tnw v', o)
                                                )
                                            _hy_anon_var_36 = None
                                        else:
                                            if isinstance(o, xml.etree.ElementTree.
                                                ElementTree) or isinstance(o, xml.
                                                etree.ElementTree.Element):
                                                raise Py2HyReturnException(bash('tnw v',
                                                    pprs(o)))
                                                _hy_anon_var_35 = None
                                            else:
                                                if o is None:
                                                    _hy_anon_var_34 = None
                                                else:
                                                    try:
                                                        raise Py2HyReturnException(bash(
                                                            "tm -d -tout nw 'v'", pickle.dumps(o)))
                                                        _hy_anon_var_33 = None
                                                    except Py2HyReturnException as e:
                                                        raise e
                                                        _hy_anon_var_33 = None
                                                    except:
                                                        _hy_anon_var_33 = None
                                                    _hy_anon_var_34 = (_hy_anon_var_33)
                                                _hy_anon_var_35 = (_hy_anon_var_34)
                                            _hy_anon_var_36 = _hy_anon_var_35
                                        _hy_anon_var_37 = _hy_anon_var_36
                                    _hy_anon_var_38 = _hy_anon_var_37
                                _hy_anon_var_39 = _hy_anon_var_38
                            _hy_anon_var_40 = _hy_anon_var_39
                        _hy_anon_var_41 = _hy_anon_var_40
                    _hy_anon_var_42 = _hy_anon_var_41
                _hy_anon_var_43 = _hy_anon_var_42
            _hy_anon_var_44 = _hy_anon_var_43
        _hy_anon_var_45 = _hy_anon_var_44
    except Py2HyReturnException as e:
        _hy_anon_var_45 = e.retvalue
    return _hy_anon_var_45


def v(o):
    """
    Splits the terminal, and runs a program on the string representation of the object, depending on its type, such as visidata for a DataFrame.

    :param o: The object to be opened in tmux
    """
    try:
        sys.stdout.write(str(type(o)))
        if sys.version_info < (3, 0) and isinstance(o, unicode):
            raise Py2HyReturnException(bash("tm -d nw 'vlf'", o.decode(
                'utf-8')))
            _hy_anon_var_46 = None
        else:
            _hy_anon_var_46 = None
        if isinstance(o, pd.core.frame.DataFrame):
            raise Py2HyReturnException(bash("tm -d nw 'vlf'", o.to_csv(
                index=False)))
            _hy_anon_var_58 = None
        else:
            if isinstance(o, set):
                raise Py2HyReturnException(bash("tm -d nw 'vlf'", pd.
                    DataFrame(dict(o)).to_csv(index=False)))
                _hy_anon_var_57 = None
            else:
                if isinstance(o, tuple):
                    raise Py2HyReturnException(bash("tm -d nw 'vlf'", pd.
                        DataFrame(list(o)).to_csv(index=False)))
                    _hy_anon_var_56 = None
                else:
                    if isinstance(o, numpy.ndarray):
                        raise Py2HyReturnException(bash("tm -d nw 'vlf'",
                            pd.DataFrame(o).to_csv(index=False)))
                        _hy_anon_var_55 = None
                    else:
                        if isinstance(o, set):
                            raise Py2HyReturnException(bash(
                                "tm -d nw 'vlf'", pd.DataFrame(o).to_csv(
                                index=False)))
                            _hy_anon_var_54 = None
                        else:
                            if isinstance(o, list):
                                raise Py2HyReturnException(bash(
                                    "tm -d nw 'vlf'", pd.DataFrame(o).
                                    to_csv(index=False)))
                                _hy_anon_var_53 = None
                            else:
                                if hasattr(type(o), 'to_csv') and callable(
                                    getattr(type(o), 'to_csv')):
                                    raise Py2HyReturnException(bash(
                                        "tm -d nw 'vlf'", o.to_csv(index=
                                        False)))
                                    _hy_anon_var_52 = None
                                else:
                                    if isinstance(o, dict):
                                        bash('dict')
                                        raise Py2HyReturnException(bash(
                                            "tm -d nw 'vlf'", ppr(o)))
                                        _hy_anon_var_51 = None
                                    else:
                                        if isinstance(o, str):
                                            raise Py2HyReturnException(bash(
                                                "tm -d nw 'vlf'", o))
                                            _hy_anon_var_50 = None
                                        else:
                                            if isinstance(o, xml.etree.ElementTree.
                                                ElementTree) or isinstance(o, xml.
                                                etree.ElementTree.Element):
                                                raise Py2HyReturnException(bash(
                                                    "tm -d nw 'vlf'", pprs(o)))
                                                _hy_anon_var_49 = None
                                            else:
                                                if o is None:
                                                    _hy_anon_var_48 = None
                                                else:
                                                    try:
                                                        raise Py2HyReturnException(bash(
                                                            "tm -d nw 'vlf'", pickle.dumps(o)))
                                                        _hy_anon_var_47 = None
                                                    except Py2HyReturnException as e:
                                                        raise e
                                                        _hy_anon_var_47 = None
                                                    except:
                                                        _hy_anon_var_47 = None
                                                    _hy_anon_var_48 = (_hy_anon_var_47)
                                                _hy_anon_var_49 = (_hy_anon_var_48)
                                            _hy_anon_var_50 = _hy_anon_var_49
                                        _hy_anon_var_51 = _hy_anon_var_50
                                    _hy_anon_var_52 = _hy_anon_var_51
                                _hy_anon_var_53 = _hy_anon_var_52
                            _hy_anon_var_54 = _hy_anon_var_53
                        _hy_anon_var_55 = _hy_anon_var_54
                    _hy_anon_var_56 = _hy_anon_var_55
                _hy_anon_var_57 = _hy_anon_var_56
            _hy_anon_var_58 = _hy_anon_var_57
        _hy_anon_var_59 = _hy_anon_var_58
    except Py2HyReturnException as e:
        _hy_anon_var_59 = e.retvalue
    return _hy_anon_var_59


def c(command):
    """Run a bash command and open the result in fpvd."""
    lf = pd.DataFrame(bash(command)[0].splitlines())
    return r(lf)


def bpy():
    """Splits the screen, and starts a bpython."""
    return bash('tm -d sph -c ~ -n bpython bpython &')[0].decode('utf-8')


def t(o):
    """Alias for type()"""
    return type(o)


def env(s):
    """Alias for os.environ()"""
    return os.environ[s]


def full_type_name(o):
    try:
        module = o.__class__.__module__
        if module is None or module == str.__class__.__module__:
            raise Py2HyReturnException(o.__class__.__name__)
            _hy_anon_var_60 = None
        else:
            raise Py2HyReturnException(module + '.' + o.__class__.__name__)
            _hy_anon_var_60 = None
        _hy_anon_var_61 = _hy_anon_var_60
    except Py2HyReturnException as e:
        _hy_anon_var_61 = e.retvalue
    return _hy_anon_var_61


def ts(o):
    """type() string name"""
    return full_type_name(o)


def pdl(s):
    """get type from string"""
    return pydoc.locate(s)


def pwd():
    """Just runs bash pwd"""
    return bash('pwd')[0].rstrip('\n')


def ls():
    """Just runs bash ls"""
    return bash('ls')[0].rstrip('\n').splitlines()


def lsautofiles(dir=None):
    """Just runs bash lsautofiles"""
    try:
        if dir:
            raise Py2HyReturnException(bash('lsautofiles "' + dir + '"')[0]
                .splitlines())
            _hy_anon_var_62 = None
        else:
            raise Py2HyReturnException(bash('lsautofiles')[0].splitlines())
            _hy_anon_var_62 = None
        _hy_anon_var_63 = _hy_anon_var_62
    except Py2HyReturnException as e:
        _hy_anon_var_63 = e.retvalue
    return _hy_anon_var_63


def find():
    """Just runs bash find"""
    return bash('f find-all-no-git')[0].splitlines()


def dirinfo():
    """Just runs bash dirinfo"""
    return bash('u dirinfo')[0]


def tv(input=''):
    """Puts the output into vim"""
    return bash('tv', input)[0]


def tsp():
    """Just runs bash tmux split pane"""
    return bash('tm -te -d sph')[0]


def cipe(cmd='', ins=''):
    """vipe but any command"""
    if not cmd:
        cmd = 'vipe'
        _hy_anon_var_64 = None
    else:
        _hy_anon_var_64 = None
    return bash('tm spv ' + q(cmd), ins)[0]


def spv(cmd='', ins='', has_output=False):
    """Just runs bash tmux split pane"""
    return bash('tm -S -tout spv ' + q(cmd), ins)[0]


def sph(cmd='', ins='', has_output=False):
    """Just runs bash tmux split pane"""
    return bash('tm -S -tout sph ' + q(cmd), ins)[0]


def spvi(cmd='', ins=''):
    """spv. takes stdin. has tty out"""
    return bash('tm -S -tout spv ' + q(cmd), ins)[0]


def sphi(cmd='', ins=''):
    """sph. takes stdin. has tty out"""
    return bash('tm -S -tout sph ' + q(cmd), ins)[0]


def spvio(cmd='', ins=''):
    """spv. takes stdin. returns stdout"""
    return bash('tm spv ' + q(cmd) + ' | cat', ins)[0]


def sphio(cmd='', ins=''):
    """sph. takes stdin. returns stdout"""
    return bash('tm sph ' + q(cmd) + ' | cat', ins)[0]


def vipe(ins=''):
    """vipe"""
    return spvio('vipe', ins)


def spvd(cmd=''):
    """Just runs bash tmux split pane"""
    bash('tm -te -d spv ' + q(cmd))[0]
    return None


def sphd(cmd=''):
    """Just runs bash tmux split pane"""
    bash('tm -te -d sph ' + q(cmd))[0]
    return None


def fish():
    """Just runs fish in a tmux split pane"""
    return bash('tm -te -d sph fish')[0]


def zsh():
    """Just runs zsh in a tmux split pane"""
    return bash('tm -te -d sph')[0]


def sh():
    """Just runs bash in a tmux split pane"""
    return bash('tm -te -d sph bash')[0]


def tcp():
    """Just runs bash tmux capture"""
    return bash('tm -te -d capture -clean -editor ec')[0]


def sl():
    """Opens locals() in pvd"""
    return r(locals())


def rl():
    """Opens locals() in pvd"""
    return sl()


def map_funcs(obj, func_list):
    return [func(obj) for func in func_list]


def get_stats_dataframe_for_series(o):
    df = pd.DataFrame(o)
    fl = [pd.Series.std, pd.Series.median, pd.Series.mean, pd.Series.max,
        pd.Series.min]
    for c in df.columns.tolist():
        dti = list(zip(list(map(lambda f: f.__name__, fl)), map_funcs(c, fl)))
        dfi = pd.DataFrame(dti)
    return dfi


def show_stats(o):
    try:
        try:
            df = pd.DataFrame(o)
            _hy_anon_var_65 = None
        except Py2HyReturnException as e:
            raise e
            _hy_anon_var_65 = None
        except ValueError:
            print("Can't be converted to DataFrame")
            raise Py2HyReturnException(None)
            _hy_anon_var_65 = None
        fl = [pd.Series.std, pd.Series.median, pd.Series.mean, pd.Series.
            max, pd.Series.min]
        for c in df.columns.tolist():
            try:
                dti = list(zip(list(map(lambda f: f.__name__, fl)),
                    map_funcs(df[c], fl)))
                dfi = pd.DataFrame(dti)
                print(c)
                ppr(dfi)
                _hy_anon_var_66 = print('\n')
            except Py2HyReturnException as e:
                raise e
                _hy_anon_var_66 = None
            except:
                _hy_anon_var_66 = None
        _hy_anon_var_67 = None
    except Py2HyReturnException as e:
        _hy_anon_var_67 = e.retvalue
    return _hy_anon_var_67


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
    try:
        lock = threading.Lock()
        min_interval = 1.0 / float(max_per_second)

        def decorate(func):
            try:
                last_time_called = [0.0]

                @wraps(func)
                def rate_limited_function(*args, **kwargs):
                    try:

                        def run_func():
                            lock.release()
                            ret = func(unpack_iterable(args),
                                unpack_mapping(kwargs))
                            last_time_called[0] = time.perf_counter()
                            return ret
                        lock.acquire()
                        elapsed = time.perf_counter() - last_time_called[0]
                        left_to_wait = min_interval - elapsed
                        if delay_first_call:
                            if left_to_wait > 0:
                                if mode == 'wait':
                                    time.sleep(left_to_wait)
                                    raise Py2HyReturnException(run_func())
                                    _hy_anon_var_69 = None
                                else:
                                    if mode == 'kill':
                                        lock.release()
                                        raise Py2HyReturnException(None)
                                        _hy_anon_var_68 = None
                                    else:
                                        _hy_anon_var_68 = None
                                    _hy_anon_var_69 = _hy_anon_var_68
                                _hy_anon_var_70 = _hy_anon_var_69
                            else:
                                raise Py2HyReturnException(run_func())
                                _hy_anon_var_70 = None
                            _hy_anon_var_75 = _hy_anon_var_70
                        else:
                            if not last_time_called[0
                                ] or elapsed > min_interval:
                                raise Py2HyReturnException(run_func())
                                _hy_anon_var_74 = None
                            else:
                                if left_to_wait > 0:
                                    if mode == 'wait':
                                        time.sleep(left_to_wait)
                                        raise Py2HyReturnException(run_func())
                                        _hy_anon_var_72 = None
                                    else:
                                        if mode == 'kill':
                                            lock.release()
                                            raise Py2HyReturnException(None)
                                            _hy_anon_var_71 = None
                                        else:
                                            _hy_anon_var_71 = None
                                        _hy_anon_var_72 = _hy_anon_var_71
                                    _hy_anon_var_73 = _hy_anon_var_72
                                else:
                                    _hy_anon_var_73 = None
                                _hy_anon_var_74 = _hy_anon_var_73
                            _hy_anon_var_75 = _hy_anon_var_74
                        _hy_anon_var_76 = _hy_anon_var_75
                    except Py2HyReturnException as e:
                        _hy_anon_var_76 = e.retvalue
                    return _hy_anon_var_76
                raise Py2HyReturnException(rate_limited_function)
                _hy_anon_var_77 = None
            except Py2HyReturnException as e:
                _hy_anon_var_77 = e.retvalue
            return _hy_anon_var_77
        raise Py2HyReturnException(decorate)
        _hy_anon_var_78 = None
    except Py2HyReturnException as e:
        _hy_anon_var_78 = e.retvalue
    return _hy_anon_var_78


def make_unicode(input):
    try:
        if type(input) != unicode:
            input = input.decode('utf-8')
            raise Py2HyReturnException(input)
            _hy_anon_var_79 = None
        else:
            raise Py2HyReturnException(input)
            _hy_anon_var_79 = None
        _hy_anon_var_80 = _hy_anon_var_79
    except Py2HyReturnException as e:
        _hy_anon_var_80 = e.retvalue
    return _hy_anon_var_80


def exhaust_properties(o):
    """enumerate properties (for finding methods)"""
    methods = [method_name for method_name in dir(o) if callable(getattr(o,
        method_name))]
    return '\n'.join(methods)


def list_methods(o):
    return exhaust_properties(o)


def list_methods_and_classes(o):
    return exhaust_properties(o)


def list_children(o):
    return exhaust_properties(o)


def ep(o):
    return exhaust_properties(o)


import inspect
from inspect import ismodule


def em(o):
    """enumerate properties (for finding submodules)"""
    modules = [module_name for module_name in dir(o) if ismodule(getattr(o,
        module_name))]
    return '\n'.join(modules)


import os
import sys
GLOBAL_INDENT = 0


def wi(*args):
    """ Function to print lines indented according to level """
    (print(' ' * GLOBAL_INDENT),) if GLOBAL_INDENT else None
    for arg in args:
        print(arg),
    return print()


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
    docstr = obj.__doc__
    args = ''
    if docstr:
        items = docstr.split('\n')
        if items:
            func_descr = items[0]
            s = func_descr.replace(obj.__name__, '')
            idx1 = s.find('(')
            idx2 = s.find(')', idx1)
            if idx1 != -1 and idx2 != -1 and idx2 > idx1 + 1:
                args = s[slice(idx1 + 1, idx2, None)]
                _hy_anon_var_81 = wi('\t-Method Arguments:', args)
            else:
                _hy_anon_var_81 = None
            _hy_anon_var_82 = _hy_anon_var_81
        else:
            _hy_anon_var_82 = None
        _hy_anon_var_83 = _hy_anon_var_82
    else:
        _hy_anon_var_83 = None
    wi('\t-Method Arguments: None') if args == '' else None
    return print


def describe_func(obj, method=False):
    """ Describe the function object passed as argument.
   If this is a method object, the second argument will
   be passed as True """
    try:
        wi('+Method: %s' % obj.__name__) if method else wi('+Function: %s' %
            obj.__name__)
        try:
            arginfo = inspect.getargspec(obj)
            _hy_anon_var_84 = None
        except Py2HyReturnException as e:
            raise e
            _hy_anon_var_84 = None
        except TypeError:
            print
            raise Py2HyReturnException(None)
            _hy_anon_var_84 = None
        args = arginfo[0]
        argsvar = arginfo[1]
        if args:
            if args[0] == 'self':
                wi('\t%s is an instance method' % obj.__name__)
                _hy_anon_var_85 = args.pop(0)
            else:
                _hy_anon_var_85 = None
            wi('\t-Method Arguments:', args)
            if arginfo[3]:
                dl = len(arginfo[3])
                al = len(args)
                defargs = args[slice(al - dl, al, None)]
                _hy_anon_var_86 = wi('\t--Default arguments:', zip(defargs,
                    arginfo[3]))
            else:
                _hy_anon_var_86 = None
            _hy_anon_var_87 = _hy_anon_var_86
        else:
            _hy_anon_var_87 = None
        wi('\t-Positional Args Param: %s' % arginfo[1]) if arginfo[1] else None
        wi('\t-Keyword Args Param: %s' % arginfo[2]) if arginfo[2] else None
        _hy_anon_var_88 = print
    except Py2HyReturnException as e:
        _hy_anon_var_88 = e.retvalue
    return _hy_anon_var_88


def describe_klass(obj):
    """ Describe the class object passed as argument,
   including its methods """
    wi('+Class: %s' % obj.__name__)
    global_indent()
    count = 0
    for name in obj.__dict__:
        item = getattr(obj, name)
        if inspect.ismethod(item):
            count += 1
            _hy_anon_var_89 = describe_func(item, True)
        else:
            _hy_anon_var_89 = None
    wi('(No members)') if count == 0 else None
    global_dedent()
    return print


def describe_module(module):
    """ Describe the module object passed as argument
   including its classes and functions """
    wi('[Module: %s]\n' % module.__name__)
    global_indent()
    count = 0
    for name in dir(module):
        obj = getattr(module, name)
        if inspect.isclass(obj):
            count += 1
            _hy_anon_var_92 = describe_klass(obj)
        else:
            if inspect.ismethod(obj) or inspect.isfunction(obj):
                count += 1
                _hy_anon_var_91 = describe_func(obj)
            else:
                if inspect.isbuiltin(obj):
                    count += 1
                    _hy_anon_var_90 = describe_builtin(obj)
                else:
                    _hy_anon_var_90 = None
                _hy_anon_var_91 = _hy_anon_var_90
            _hy_anon_var_92 = _hy_anon_var_91
    wi('(No members)') if count == 0 else None
    return global_dedent()


import scipy


def describe_ndarray(a):
    try:
        try:
            raise Py2HyReturnException(scipy.stats.describe)
            _hy_anon_var_93 = None
        except Py2HyReturnException as e:
            raise e
            _hy_anon_var_93 = None
        except:
            raise Py2HyReturnException(None)
            _hy_anon_var_93 = None
        _hy_anon_var_94 = _hy_anon_var_93
    except Py2HyReturnException as e:
        _hy_anon_var_94 = e.retvalue
    return _hy_anon_var_94


def d(obj):
    """
    Describe an object
    """
    try:
        print(type(obj))
        switchDict = {'module': describe_module, 'type': describe_klass,
            'function': describe_func, 'builtin_function_or_method':
            describe_builtin, 'numpy.ndarray': describe_ndarray}
        try:
            _hy_anon_var_95 = switchDict[type(obj).__name__](obj)
        except Py2HyReturnException as e:
            raise e
            _hy_anon_var_95 = None
        except KeyError:
            _hy_anon_var_95 = ppr(obj)
        _hy_anon_var_96 = _hy_anon_var_95
    except Py2HyReturnException as e:
        _hy_anon_var_96 = e.retvalue
    return _hy_anon_var_96


def mygetsourcefile(thing):
    try:
        path = ''
        try:
            path = inspect.getsourcefile(thing)
            _hy_anon_var_97 = None
        except Py2HyReturnException as e:
            raise e
            _hy_anon_var_97 = None
        except:
            _hy_anon_var_97 = None
        if not path:
            print('source code not available')
            raise Py2HyReturnException(None)
            _hy_anon_var_98 = None
        else:
            raise Py2HyReturnException(path)
            _hy_anon_var_98 = None
        _hy_anon_var_99 = _hy_anon_var_98
    except Py2HyReturnException as e:
        _hy_anon_var_99 = e.retvalue
    return _hy_anon_var_99


def pathof(thing):
    """
    Describe a thing
    """
    try:
        ppr(thing)
        switchDict = {'module': lambda x: x.__file__, 'type': lambda x:
            mygetsourcefile(x), 'function': lambda x: mygetsourcefile(x),
            'method': lambda x: mygetsourcefile(x),
            'builtin_function_or_method': None}
        try:
            raise Py2HyReturnException(switchDict[type(thing).__name__](thing))
            _hy_anon_var_100 = None
        except Py2HyReturnException as e:
            raise e
            _hy_anon_var_100 = None
        except:
            raise Py2HyReturnException(mygetsourcefile(type(thing)))
            _hy_anon_var_100 = None
        _hy_anon_var_101 = _hy_anon_var_100
    except Py2HyReturnException as e:
        _hy_anon_var_101 = e.retvalue
    return _hy_anon_var_101


def lt(th):
    """ Get the path of the type of the thing"""
    return pathof(th)


def po(th):
    """ Get the path of the type of the thing"""
    return pathof(th)


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


from tabulate import tabulate

