(import [__future__ [print_function]])
(defclass Py2HyReturnException [Exception]
  (defn __init__ [self retvalue]
    (setv self.retvalue retvalue)))
(import [os])
(import [sys])
(import [subprocess])
(import [pprint])
(import [json])
(import [json :as jn])
(import [pydoc])
(import [pydoc [locate]])
(try
  (import [pandas :as pd])
  (except [e Py2HyReturnException]
    (raise e))
  (except []
    True))
(try
  (do
    (import [numpy])
    (import [numpy :as np]))
  (except [e Py2HyReturnException]
    (raise e))
  (except []
    True))
(import [pickle])
(try
  (import [scipy])
  (except [e Py2HyReturnException]
    (raise e))
  (except []
    True))
(import [xml])
(import [xml.etree.ElementTree :as ET])
(import [xml.dom [minidom]])
(try
  (import [sqlparse])
  (except [e Py2HyReturnException]
    (raise e))
  (except []
    True))
(try
  (do
    (import [StringIO [StringIO]])
    (import [StringIO [StringIO :as sio]]))
  (except [e Py2HyReturnException]
    (raise e))
  (except [ImportError]
    (import [io [StringIO]])
    (import [io [StringIO :as sio]])))
(import [importlib])
(defn xv [s]
  (os.path.expandvars s))
(defn b [c &optional [inputstring ""] [timeout 0]]
  "Runs a shell c"
  (setv c (xv c))
  (setv p (subprocess.Popen c :shell True :executable "/bin/sh" :stdin subprocess.PIPE :stdout subprocess.PIPE :stderr subprocess.STDOUT :close_fds True))
  (p.stdin.write ((. (str inputstring) encode) "utf-8"))
  (p.stdin.close)
  (setv output ((. (p.stdout.read) decode) "utf-8"))
  (p.wait)
  [(str output) p.returncode])
(defn bsh [c &optional [inputstring ""] [timeout 0]]
  "Runs a shell c"
  (b c inputstring timeout))
(defn bash [c &optional [inputstring ""] [timeout 0]]
  "Runs a shell c"
  (b c inputstring timeout))
(defn q [&optional [inputstring ""]]
  (get (b "q" inputstring) 0))
(defn ns [&optional [inputstring ""]]
  (get (b "ns" inputstring) 0))
(defn umn [&optional [inputstring ""]]
  (get (b "umn" inputstring) 0))
(defn mnm [&optional [inputstring ""]]
  (get (b "mnm" inputstring) 0))
(defn cat [path]
  (get (b (+ "cat " (q (umn path)))) 0))
(defn reload_shanepy []
  (print (get (b "cr $MYGIT/mullikine/shanepy/shanepy.py") 0))
  (import [shanepy])
  (importlib.reload shanepy)
  (print "You must run this manually: \"from shanepy import *\"")
  (ns "You must run this manually: \"from shanepy import *\""))
(defn ipy []
  "Splits the screen, and starts an ipython."
  (get (bash "tm -d sph -c ~ -n ipython ipython &") 0))
(setv pp (pprint.PrettyPrinter :indent 2 :width 1000 :depth 20))
(defn ppr [o]
  (try
    (do
      (when (isinstance o xml.etree.ElementTree.ElementTree)
        (print (pprs o))
        (raise (Py2HyReturnException None)))
      (when (isinstance o xml.etree.ElementTree.Element)
        (print (pprs o))
        (raise (Py2HyReturnException None)))
      (pp.pprint o))
    (except [e Py2HyReturnException]
      e.retvalue)))
(defn pprs [o]
  (try
    (do
      (when (isinstance o xml.etree.ElementTree.ElementTree)
        (setv root (o.getroot))
        (setv rough_string (ET.tostring root "utf-8"))
        (setv reparsed (minidom.parseString rough_string))
        (raise (Py2HyReturnException (reparsed.toprettyxml :indent "	"))))
      (when (isinstance o xml.etree.ElementTree.Element)
        (setv rough_string (ET.tostring o "utf-8"))
        (setv reparsed (minidom.parseString rough_string))
        (raise (Py2HyReturnException (reparsed.toprettyxml :indent "	")))))
    (except [e Py2HyReturnException]
      e.retvalue)))
(defn pretty_sql [sql]
  (sqlparse.format sql :reindent True :keyword_case "upper"))
(defn k [o]
  "
    Pickle an object then put it into a vim.

    :param o: The object to be opened in tmux
    "
  (try
    (do
      (when (is o None)
        (raise (Py2HyReturnException None)))
      (try
        (do
          (import [tempfile])
          (setv pickle_file (tempfile.NamedTemporaryFile "wb" :suffix ".pickle" :delete False))
          (pickle.dump o pickle_file)
          (raise (Py2HyReturnException (bash (+ (+ "tm -d nw \"vim \\\"" pickle_file.name) "\\\"\"")))))
        (except [e Py2HyReturnException]
          (raise e))
        (except []
          (do))))
    (except [e Py2HyReturnException]
      e.retvalue)))
(when (> sys.version_info (, 3 0))
  (try
    (import [past.builtins [execfile]])
    (except [e Py2HyReturnException]
      (raise e))
    (except []
      True)))
(defn source_file [fp]
  "Directly includes python source code."
  (setv sys.stdout (open os.devnull "w"))
  (execfile fp)
  (setv sys.stdout sys.__stdout__))
(defn read_csv_smart [&kwargs kwargs &rest args]
  (try
    (do
      (try
        (raise (Py2HyReturnException (pd.read_csv (unpack_iterable args) (unpack_mapping kwargs))))
        (except [e Py2HyReturnException]
          (raise e))
        (except []
          (do)))
      (try
        (raise (Py2HyReturnException (pd.read_csv (unpack_iterable args) (unpack_mapping kwargs))))
        (except [e Py2HyReturnException]
          (raise e))
        (except []
          (do))))
    (except [e Py2HyReturnException]
      e.retvalue)))
(defn my_to_csv [&kwargs kwargs &rest args]
  (pd.DataFrame.to_csv (unpack_iterable args) (unpack_mapping kwargs)))
(defn i []
  "Get stdin as a string"
  (import [sys])
  (sys.stdin.read))
(defn T [list_of_lists]
  "Transposes a list of lists"
  (import [six])
  (list (map list (six.moves.zip_longest (unpack_iterable list_of_lists) :fillvalue " "))))
(defn list_of_lists_to_text [arg]
  "Saves a list of lists to a text file"
  ((. "
" join) (list_comp ((. "" join) l) [l arg])))
(defn s [o fp]
  "Save object repr to path"
  (with [f (open fp "w")] (f.write o)))
(defn p [o]
  "Save object repr to stdout"
  (print o :end ""))
(defn xc [o]
  (b "xc -i" (p o)))
(defn ftf [s]
  "Creates a temporary file from a string and returns the path"
  (import [tempfile])
  (setv f (tempfile.NamedTemporaryFile "w" :suffix ".txt" :delete False))
  (f.seek 0)
  (f.write s)
  f.name)
(defn l [fp]
  (list_comp (line.rstrip "
") [line (open fp)]))
(defn o [fp]
  "
    Opens the file given by the path into a python object.

    :param str fp: The path of the file
    "
  (try
    (do
      (setv fp (xv fp))
      (import [re])
      (when (is_not (re.match ".*\\.npy" fp) None)
        (import [numpy :as np])
        (try
          (do
            (setv ret (np.load fp))
            (raise (Py2HyReturnException ret)))
          (except [e Py2HyReturnException]
            (raise e))
          (except []
            (do)))
        (try
          (do
            (setv ret (np.load fp :encoding "latin1"))
            (raise (Py2HyReturnException ret)))
          (except [e Py2HyReturnException]
            (raise e))
          (except []
            (do)))
        (try
          (do
            (setv ret (np.load fp :encoding "bytes"))
            (raise (Py2HyReturnException ret)))
          (except [e Py2HyReturnException]
            (raise e))
          (except []
            (do))))
      (when (is_not (re.match ".*\\.xml" fp) None)
        (import [pandas :as pd])
        (setv ret (read_csv_smart fp))
        (raise (Py2HyReturnException ret)))
      (when (is_not (re.match ".*\\.csv" fp) None)
        (import [pandas :as pd])
        (setv ret (read_csv_smart fp))
        (raise (Py2HyReturnException ret)))
      (when (is_not (re.match ".*\\.xls" fp) None)
        (import [pandas :as pd])
        (setv ret (pd.read_excel fp))
        (raise (Py2HyReturnException ret)))
      (when (or (is_not (re.match ".*\\.pickle$" fp) None) (is_not (re.match ".*\\.p$" fp) None))
        (import [pickle])
        (with [f (open fp "rUb")] (setv data (f.read)))
        (setv ret (pickle.loads data))
        (raise (Py2HyReturnException ret)))
      (with [f (open fp "rU")] (defn strip_list [l]
          (if (= (get l (- 1)) "
")
            (get l (slice None (- 1) None))
            l)) (raise (Py2HyReturnException (list_comp (strip_list (list line)) [line (f.readlines)])))))
    (except [e Py2HyReturnException]
      e.retvalue)))
(defn r [o]
  "
    Splits the terminal, and runs a program on the string representation of the object, depending on its type, such as visidata for a DataFrame.

    :param o: The object to be opened in tmux
    "
  (try
    (do
      (sys.stdout.write (str (type o)))
      (when (and (< sys.version_info (, 3 0)) (isinstance o unicode))
        (raise (Py2HyReturnException (bash "tnw 'fpvd'" (o.decode "utf-8")))))
      (cond
        [(isinstance o pd.core.frame.DataFrame)
         (raise (Py2HyReturnException (bash "tnw fpvd" (o.to_csv :index False))))]
        [(isinstance o set)
         (raise (Py2HyReturnException (bash "tnw fpvd" ((. (pd.DataFrame (dict o)) to_csv) :index False))))]
        [True
         (cond
           [(isinstance o tuple)
            (raise (Py2HyReturnException (bash "tnw fpvd" ((. (pd.DataFrame (list o)) to_csv) :index False))))]
           [(isinstance o numpy.ndarray)
            (raise (Py2HyReturnException (bash "tnw fpvd" ((. (pd.DataFrame o) to_csv) :index False))))]
           [True
            (cond
              [(isinstance o set)
               (raise (Py2HyReturnException (bash "tnw fpvd" ((. (pd.DataFrame o) to_csv) :index False))))]
              [(isinstance o list)
               (raise (Py2HyReturnException (bash "tnw fpvd" ((. (pd.DataFrame o) to_csv) :index False))))]
              [True
               (cond
                 [(and (hasattr (type o) "to_csv") (callable (getattr (type o) "to_csv")))
                  (raise (Py2HyReturnException (bash "tnw fpvd" (o.to_csv :index False))))]
                 [(isinstance o dict)
                  (do
                    (bash "dict")
                    (raise (Py2HyReturnException (bash "tnw v" (ppr o)))))]
                 [True
                  (cond
                    [(isinstance o str)
                     (raise (Py2HyReturnException (bash "tnw v" o)))]
                    [(or (isinstance o xml.etree.ElementTree.ElementTree) (isinstance o xml.etree.ElementTree.Element))
                     (raise (Py2HyReturnException (bash "tnw v" (pprs o))))]
                    [True
                     (if (is o None)
                       (do
                         (do))
                       (do
                         (try
                           (raise (Py2HyReturnException (bash "tm -d -tout nw 'v'" (pickle.dumps o))))
                           (except [e Py2HyReturnException]
                             (raise e))
                           (except []
                             (do)))))])])])])]))
    (except [e Py2HyReturnException]
      e.retvalue)))
(defn v [o]
  "
    Splits the terminal, and runs a program on the string representation of the object, depending on its type, such as visidata for a DataFrame.

    :param o: The object to be opened in tmux
    "
  (try
    (do
      (sys.stdout.write (str (type o)))
      (when (and (< sys.version_info (, 3 0)) (isinstance o unicode))
        (raise (Py2HyReturnException (bash "tm -d nw 'vlf'" (o.decode "utf-8")))))
      (cond
        [(isinstance o pd.core.frame.DataFrame)
         (raise (Py2HyReturnException (bash "tm -d nw 'vlf'" (o.to_csv :index False))))]
        [(isinstance o set)
         (raise (Py2HyReturnException (bash "tm -d nw 'vlf'" ((. (pd.DataFrame (dict o)) to_csv) :index False))))]
        [True
         (cond
           [(isinstance o tuple)
            (raise (Py2HyReturnException (bash "tm -d nw 'vlf'" ((. (pd.DataFrame (list o)) to_csv) :index False))))]
           [(isinstance o numpy.ndarray)
            (raise (Py2HyReturnException (bash "tm -d nw 'vlf'" ((. (pd.DataFrame o) to_csv) :index False))))]
           [True
            (cond
              [(isinstance o set)
               (raise (Py2HyReturnException (bash "tm -d nw 'vlf'" ((. (pd.DataFrame o) to_csv) :index False))))]
              [(isinstance o list)
               (raise (Py2HyReturnException (bash "tm -d nw 'vlf'" ((. (pd.DataFrame o) to_csv) :index False))))]
              [True
               (cond
                 [(and (hasattr (type o) "to_csv") (callable (getattr (type o) "to_csv")))
                  (raise (Py2HyReturnException (bash "tm -d nw 'vlf'" (o.to_csv :index False))))]
                 [(isinstance o dict)
                  (do
                    (bash "dict")
                    (raise (Py2HyReturnException (bash "tm -d nw 'vlf'" (ppr o)))))]
                 [True
                  (cond
                    [(isinstance o str)
                     (raise (Py2HyReturnException (bash "tm -d nw 'vlf'" o)))]
                    [(or (isinstance o xml.etree.ElementTree.ElementTree) (isinstance o xml.etree.ElementTree.Element))
                     (raise (Py2HyReturnException (bash "tm -d nw 'vlf'" (pprs o))))]
                    [True
                     (if (is o None)
                       (do
                         (do))
                       (do
                         (try
                           (raise (Py2HyReturnException (bash "tm -d nw 'vlf'" (pickle.dumps o))))
                           (except [e Py2HyReturnException]
                             (raise e))
                           (except []
                             (do)))))])])])])]))
    (except [e Py2HyReturnException]
      e.retvalue)))
(defn c [command]
  "Run a bash command and open the result in fpvd."
  (setv lf (pd.DataFrame ((. (get (bash command) 0) splitlines))))
  (r lf))
(defn bpy []
  "Splits the screen, and starts a bpython."
  ((. (get (bash "tm -d sph -c ~ -n bpython bpython &") 0) decode) "utf-8"))
(defn t [o]
  "Alias for type()"
  (type o))
(defn env [s]
  "Alias for os.environ()"
  (get os.environ s))
(defn full_type_name [o]
  (try
    (do
      (setv module o.__class__.__module__)
      (if (or (is module None) (= module str.__class__.__module__))
        (do
          (raise (Py2HyReturnException o.__class__.__name__)))
        (do
          (raise (Py2HyReturnException (+ (+ module ".") o.__class__.__name__))))))
    (except [e Py2HyReturnException]
      e.retvalue)))
(defn ts [o]
  "type() string name"
  (full_type_name o))
(defn pdl [s]
  "get type from string"
  (pydoc.locate s))
(defn pwd []
  "Just runs bash pwd"
  ((. (get (bash "pwd") 0) rstrip) "
"))
(defn ls []
  "Just runs bash ls"
  ((. ((. (get (bash "ls") 0) rstrip) "
") splitlines)))
(defn lsautofiles [&optional [dir None]]
  "Just runs bash lsautofiles"
  (try
    (if dir
      (do
        (raise (Py2HyReturnException ((. (get (bash (+ (+ "lsautofiles \"" dir) "\"")) 0) splitlines)))))
      (do
        (raise (Py2HyReturnException ((. (get (bash "lsautofiles") 0) splitlines))))))
    (except [e Py2HyReturnException]
      e.retvalue)))
(defn find []
  "Just runs bash find"
  ((. (get (bash "f find-all-no-git") 0) splitlines)))
(defn dirinfo []
  "Just runs bash dirinfo"
  (get (bash "u dirinfo") 0))
(defn tv [&optional [input ""]]
  "Puts the output into vim"
  (get (bash "tv" input) 0))
(defn tsp []
  "Just runs bash tmux split pane"
  (get (bash "tm -te -d sph") 0))
(defn cipe [&optional [cmd ""] [ins ""]]
  "vipe but any command"
  (when (not cmd)
    (setv cmd "vipe"))
  (get (bash (+ "tm spv " (q cmd)) ins) 0))
(defn spv [&optional [cmd ""] [ins ""] [has_output False]]
  "Just runs bash tmux split pane"
  (get (bash (+ "tm -S -tout spv " (q cmd)) ins) 0))
(defn sph [&optional [cmd ""] [ins ""] [has_output False]]
  "Just runs bash tmux split pane"
  (get (bash (+ "tm -S -tout sph " (q cmd)) ins) 0))
(defn spvi [&optional [cmd ""] [ins ""]]
  "spv. takes stdin. has tty out"
  (get (bash (+ "tm -S -tout spv " (q cmd)) ins) 0))
(defn sphi [&optional [cmd ""] [ins ""]]
  "sph. takes stdin. has tty out"
  (get (bash (+ "tm -S -tout sph " (q cmd)) ins) 0))
(defn spvio [&optional [cmd ""] [ins ""]]
  "spv. takes stdin. returns stdout"
  (get (bash (+ (+ "tm spv " (q cmd)) " | cat") ins) 0))
(defn sphio [&optional [cmd ""] [ins ""]]
  "sph. takes stdin. returns stdout"
  (get (bash (+ (+ "tm sph " (q cmd)) " | cat") ins) 0))
(defn vipe [&optional [ins ""]]
  "vipe"
  (spvio "vipe" ins))
(defn spvd [&optional [cmd ""]]
  "Just runs bash tmux split pane"
  (get (bash (+ "tm -te -d spv " (q cmd))) 0)
  None)
(defn sphd [&optional [cmd ""]]
  "Just runs bash tmux split pane"
  (get (bash (+ "tm -te -d sph " (q cmd))) 0)
  None)
(defn fish []
  "Just runs fish in a tmux split pane"
  (get (bash "tm -te -d sph fish") 0))
(defn zsh []
  "Just runs zsh in a tmux split pane"
  (get (bash "tm -te -d sph") 0))
(defn sh []
  "Just runs bash in a tmux split pane"
  (get (bash "tm -te -d sph bash") 0))
(defn tcp []
  "Just runs bash tmux capture"
  (get (bash "tm -te -d capture -clean -editor ec") 0))
(defn sl []
  "Opens locals() in pvd"
  (r (locals)))
(defn rl []
  "Opens locals() in pvd"
  (sl))
(defn map_funcs [obj func_list]
  (list_comp (func obj) [func func_list]))
(defn get_stats_dataframe_for_series [o]
  (setv df (pd.DataFrame o))
  (setv fl [pd.Series.std pd.Series.median pd.Series.mean pd.Series.max pd.Series.min])
  (for [c (df.columns.tolist)]
    (setv dti (list (zip (list (map (fn [f] f.__name__) fl)) (map_funcs c fl))))
    (setv dfi (pd.DataFrame dti)))
  dfi)
(defn show_stats [o]
  (try
    (do
      (try
        (setv df (pd.DataFrame o))
        (except [e Py2HyReturnException]
          (raise e))
        (except [ValueError]
          (print "Can't be converted to DataFrame")
          (raise (Py2HyReturnException None))))
      (setv fl [pd.Series.std pd.Series.median pd.Series.mean pd.Series.max pd.Series.min])
      (for [c (df.columns.tolist)]
        (try
          (do
            (setv dti (list (zip (list (map (fn [f] f.__name__) fl)) (map_funcs (get df c) fl))))
            (setv dfi (pd.DataFrame dti))
            (print c)
            (ppr dfi)
            (print "
"))
          (except [e Py2HyReturnException]
            (raise e))
          (except []
            (do)))))
    (except [e Py2HyReturnException]
      e.retvalue)))
(import [time])
(import [threading])
(import [functools [wraps]])
(defn rate_limited [max_per_second &optional [mode "wait"] [delay_first_call False]]
  "
    Decorator that make functions not be called faster than

    set mode to 'kill' to just ignore requests that are faster than the
    rate.

    set delay_first_call to True to delay the first call as well
    "
  (try
    (do
      (setv lock (threading.Lock))
      (setv min_interval (/ 1.0 (float max_per_second)))
      (defn decorate [func]
        (try
          (do
            (setv last_time_called [0.0])
            (with_decorator
              (wraps func)
              (defn rate_limited_function [&kwargs kwargs &rest args]
                (try
                  (do
                    (defn run_func []
                      (lock.release)
                      (setv ret (func (unpack_iterable args) (unpack_mapping kwargs)))
                      (assoc last_time_called 0 (time.perf_counter))
                      ret)
                    (lock.acquire)
                    (setv elapsed (- (time.perf_counter) (get last_time_called 0)))
                    (setv left_to_wait (- min_interval elapsed))
                    (cond
                      [delay_first_call
                       (if (> left_to_wait 0)
                         (do
                           (cond
                             [(= mode "wait")
                              (do
                                (time.sleep left_to_wait)
                                (raise (Py2HyReturnException (run_func))))]
                             [(= mode "kill")
                              (do
                                (lock.release)
                                (raise (Py2HyReturnException None)))]
                             [True
                              (do)]))
                         (do
                           (raise (Py2HyReturnException (run_func)))))]
                      [(or (not (get last_time_called 0)) (> elapsed min_interval))
                       (raise (Py2HyReturnException (run_func)))]
                      [True
                       (when (> left_to_wait 0)
                         (cond
                           [(= mode "wait")
                            (do
                              (time.sleep left_to_wait)
                              (raise (Py2HyReturnException (run_func))))]
                           [(= mode "kill")
                            (do
                              (lock.release)
                              (raise (Py2HyReturnException None)))]
                           [True
                            (do)]))]))
                  (except [e Py2HyReturnException]
                    e.retvalue))))
            (raise (Py2HyReturnException rate_limited_function)))
          (except [e Py2HyReturnException]
            e.retvalue)))
      (raise (Py2HyReturnException decorate)))
    (except [e Py2HyReturnException]
      e.retvalue)))
(defn make_unicode [input]
  (try
    (if (!= (type input) unicode)
      (do
        (setv input (input.decode "utf-8"))
        (raise (Py2HyReturnException input)))
      (do
        (raise (Py2HyReturnException input))))
    (except [e Py2HyReturnException]
      e.retvalue)))
(defn exhaust_properties [o]
  "enumerate properties (for finding methods)"
  (setv methods (list_comp method_name [method_name (dir o)] (and (callable (getattr o method_name)))))
  ((. "
" join) methods))
(defn list_methods [o]
  (exhaust_properties o))
(defn list_methods_and_classes [o]
  (exhaust_properties o))
(defn list_children [o]
  (exhaust_properties o))
(defn ep [o]
  (exhaust_properties o))
(import [inspect])
(import [inspect [ismodule]])
(defn em [o]
  "enumerate properties (for finding submodules)"
  (setv modules (list_comp module_name [module_name (dir o)] (and (ismodule (getattr o module_name)))))
  ((. "
" join) modules))
(import [os] [sys])
(setv GLOBAL_INDENT 0)
(defn wi [&rest args]
  " Function to print lines indented according to level "
  (when GLOBAL_INDENT
    (, (print (* " " GLOBAL_INDENT))))
  (for [arg args]
    (, (print arg)))
  (print))
(defn global_indent []
  " Increase indentation "
  (global GLOBAL_INDENT)
  (+= GLOBAL_INDENT 4))
(defn global_dedent []
  " Decrease indentation "
  (global GLOBAL_INDENT)
  (_= GLOBAL_INDENT 4))
(defn describe_builtin [obj]
  " Describe a builtin function "
  (wi (% "+Built-in Function: %s" obj.__name__))
  (setv docstr obj.__doc__)
  (setv args "")
  (when docstr
    (setv items (docstr.split "
"))
    (when items
      (setv func_descr (get items 0))
      (setv s (func_descr.replace obj.__name__ ""))
      (setv idx1 (s.find "("))
      (setv idx2 (s.find ")" idx1))
      (when (and (!= idx1 (- 1)) (!= idx2 (- 1)) (> idx2 (+ idx1 1)))
        (setv args (get s (slice (+ idx1 1) idx2 None)))
        (wi "	-Method Arguments:" args))))
  (when (= args "")
    (wi "	-Method Arguments: None"))
  print)
(defn describe_func [obj &optional [method False]]
  " Describe the function object passed as argument.
   If this is a method object, the second argument will
   be passed as True "
  (try
    (do
      (if method
        (do
          (wi (% "+Method: %s" obj.__name__)))
        (do
          (wi (% "+Function: %s" obj.__name__))))
      (try
        (setv arginfo (inspect.getargspec obj))
        (except [e Py2HyReturnException]
          (raise e))
        (except [TypeError]
          print
          (raise (Py2HyReturnException None))))
      (setv args (get arginfo 0))
      (setv argsvar (get arginfo 1))
      (when args
        (when (= (get args 0) "self")
          (wi (% "	%s is an instance method" obj.__name__))
          (args.pop 0))
        (wi "	-Method Arguments:" args)
        (when (get arginfo 3)
          (setv dl (len (get arginfo 3)))
          (setv al (len args))
          (setv defargs (get args (slice (- al dl) al None)))
          (wi "	--Default arguments:" (zip defargs (get arginfo 3)))))
      (when (get arginfo 1)
        (wi (% "	-Positional Args Param: %s" (get arginfo 1))))
      (when (get arginfo 2)
        (wi (% "	-Keyword Args Param: %s" (get arginfo 2))))
      print)
    (except [e Py2HyReturnException]
      e.retvalue)))
(defn describe_klass [obj]
  " Describe the class object passed as argument,
   including its methods "
  (wi (% "+Class: %s" obj.__name__))
  (global_indent)
  (setv count 0)
  (for [name obj.__dict__]
    (setv item (getattr obj name))
    (when (inspect.ismethod item)
      (+= count 1)
      (describe_func item True)))
  (when (= count 0)
    (wi "(No members)"))
  (global_dedent)
  print)
(defn describe_module [module]
  " Describe the module object passed as argument
   including its classes and functions "
  (wi (% "[Module: %s]
" module.__name__))
  (global_indent)
  (setv count 0)
  (for [name (dir module)]
    (setv obj (getattr module name))
    (cond
      [(inspect.isclass obj)
       (do
         (+= count 1)
         (describe_klass obj))]
      [(or (inspect.ismethod obj) (inspect.isfunction obj))
       (do
         (+= count 1)
         (describe_func obj))]
      [True
       (when (inspect.isbuiltin obj)
         (+= count 1)
         (describe_builtin obj))]))
  (when (= count 0)
    (wi "(No members)"))
  (global_dedent))
(import [scipy])
(defn describe_ndarray [a]
  (try
    (try
      (raise (Py2HyReturnException scipy.stats.describe))
      (except [e Py2HyReturnException]
        (raise e))
      (except []
        (do)
        (raise (Py2HyReturnException None))))
    (except [e Py2HyReturnException]
      e.retvalue)))
(defn d [obj]
  "
    Describe an object
    "
  (try
    (do
      (print (type obj))
      (setv switchDict {"module" describe_module "type" describe_klass "function" describe_func "builtin_function_or_method" describe_builtin "numpy.ndarray" describe_ndarray})
      (try
        ((get switchDict (. (type obj) __name__)) obj)
        (except [e Py2HyReturnException]
          (raise e))
        (except [KeyError]
          (ppr obj))))
    (except [e Py2HyReturnException]
      e.retvalue)))
(defn mygetsourcefile [thing]
  (try
    (do
      (setv path "")
      (try
        (setv path (inspect.getsourcefile thing))
        (except [e Py2HyReturnException]
          (raise e))
        (except []
          (do)))
      (if (not path)
        (do
          (print "source code not available")
          (raise (Py2HyReturnException None)))
        (do
          (raise (Py2HyReturnException path)))))
    (except [e Py2HyReturnException]
      e.retvalue)))
(defn pathof [thing]
  "
    Describe a thing
    "
  (try
    (do
      (ppr thing)
      (setv switchDict {"module" (fn [x] x.__file__) "type" (fn [x] (mygetsourcefile x)) "function" (fn [x] (mygetsourcefile x)) "method" (fn [x] (mygetsourcefile x)) "builtin_function_or_method" None})
      (try
        (raise (Py2HyReturnException ((get switchDict (. (type thing) __name__)) thing)))
        (except [e Py2HyReturnException]
          (raise e))
        (except []
          (raise (Py2HyReturnException (mygetsourcefile (type thing)))))))
    (except [e Py2HyReturnException]
      e.retvalue)))
(defn lt [th]
  " Get the path of the type of the thing"
  (pathof th))
(defn po [th]
  " Get the path of the type of the thing"
  (pathof th))
(defn pathoftypeof [obj]
  (pathof obj))
(defn version []
  (print sys.version_info)
  sys.version_info)
(import [shlex])
(defn py_q [s]
  (shlex.quote s))
(import [django])
(defn getenv [varname]
  "
    gets an environment variable
    "
  (os.environ.get varname))
(import [tabulate [tabulate]])