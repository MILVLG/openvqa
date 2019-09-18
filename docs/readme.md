# How to Edit OpenVQA Document

OpenVQA Document is built by [Sphix](https://www.sphinx-doc.org/en/master/) and hosted on [Read the Docs](https://readthedocs.org/).
You need know both [Markdown](https://markdown-zh.readthedocs.io/) and 
[reStructuredText](http://docutils.sourceforge.net/rst.html) plaintext markup syntax.
We use the `.md` and `.rst` suffixes to distinguish them. 
Usually OpenVQA source coders will participate in the maintenance of the document.
In most cases, programmers have learned markdown syntax. So the markdown syntax is used for simple content.
In order to use the [autodoc](https://www.sphinx-doc.org/ext/autodoc.html) feature in Sphix, 
you must be familiar with the documentation content mentioned above.


## Edit and Debug

Different developers have different document maintenance habits, 
it is recommended to maintain the document with a separate `docs: xxxx` branch 
instead of directly making Pull Requests to the master branch.

When debugging locally, we usually use two instructions:

```shell
.\make.bat clean              
.\make.bat html
```

Note: 

- Make sure the current path is under the `docs` folder and have installed all things in `requirements.txt`.
- `clean` operation must be performed before `build`, otherwise undetectable errors may occur.

## Push to GitHub

In order to simplify the code review process and reduce `.git` size, changes to the `_build` folder are usually not logged.
(Check the `.gitignore` file in the root path of the project and find `docs/_build/` line for Sphinx documentation)
Only the contents in the `_source` folder will be submitted to GitHub (unless `_template` or `_theme` is used).

## Build and Host on Readthedocs

Readthedocs detect changes to the source code of the document through webhooks, 
after the source code is updated, you need to check whether the document hosted in readthedocs is successfully built.