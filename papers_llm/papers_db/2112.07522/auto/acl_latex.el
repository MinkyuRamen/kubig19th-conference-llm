(TeX-add-style-hook
 "acl_latex"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "11pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("fontenc" "T1") ("inputenc" "utf8") ("subfig" "caption=false")))
   (TeX-run-style-hooks
    "latex2e"
    "fewshotlearnercomp"
    "appendix"
    "article"
    "art11"
    "acl"
    "times"
    "latexsym"
    "fontenc"
    "inputenc"
    "microtype"
    "soul"
    "xspace"
    "multirow"
    "amsmath"
    "amssymb"
    "bbding"
    "graphicx"
    "textcomp"
    "pifont"
    "subfig")
   (TeX-add-symbols
    '("MF" 1)
    '("mf" 1)
    "argmax"
    "argmin"
    "enotesoff"
    "enoteson"
    "tasksymbol"
    "md"
    "mds"
    "mdr"
    "mdrs"
    "figref"
    "figlabel"
    "chapref"
    "chaplabel"
    "Tabref"
    "tabref"
    "tabsref"
    "tabrefbare"
    "tablabel"
    "Secref"
    "secref"
    "seclabel"
    "qref"
    "eqrefn"
    "eqsref"
    "eqlabel"
    "subsp")
   (LaTeX-add-labels
    "fig:#1"
    "p:#1"
    "chap:#1"
    "tab:#1"
    "sec:#1"
    "eqn:#1")
   (LaTeX-add-environments
    '("loneinnerlist" LaTeX-env-args ["argument"] 0)
    '("innerlist" LaTeX-env-args ["argument"] 0)
    '("lonelist" LaTeX-env-args ["argument"] 0)
    '("outerlist" LaTeX-env-args ["argument"] 0))
   (LaTeX-add-bibliographies
    "custom")
   (LaTeX-add-counters
    "notecounter"))
 :latex)

