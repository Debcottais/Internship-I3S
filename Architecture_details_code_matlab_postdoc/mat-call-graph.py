# python mat-call-grapy code_folder > call-graph.dot
# dot -Tsvg call-graph.dot > call-graph.svg

import pathlib as pl_
import re as re_
import sys as sy_


assert sy_.argv.__len__() > 1

base_folder = pl_.Path(sy_.argv[1])
assert base_folder.is_dir()

mod_of_fun  = {}
dep_of_fun  = {}
funs_in_mod = {}

for elm in base_folder.rglob('*.m'):
    if not elm.is_file():
        continue

    module = elm.__str__()

    with open(elm) as elm_accessor:
        raw_code = elm_accessor.read()

    # --- Join lines continued with ellipses: raw_code -> code
    code = ''
    while True:
        newline_idx = raw_code.find('\n')
        if newline_idx < 0:
            code += raw_code
            break

        line = raw_code[:(newline_idx + 1)]
        raw_code = raw_code[(newline_idx + 1):]

        comment_match = re_.match(' *%', line)
        if comment_match is None:
            ellipsis_match = re_.search('\.\.\. *$', line)
            if ellipsis_match is None:
                code += line
            else:
                code += line[:ellipsis_match.start(0)]

    # --- Parse code to fill up mod_of_fun, dep_of_fun, and funs_in_mod
    function = None # Not in a function context yet
    for line in code.split('\n'):
        fun_match = re_.match('function +(\[?[^]]+\]? *= *)?(\w+)\(', line)
        if fun_match is None:
            if function is not None: # Within a function body => look for function calls
                dep_match = re_.search('(\w+)\(', line)
                while dep_match is not None:
                    dependency = dep_match.group(1)
                    if function in dep_of_fun:
                        if dependency not in dep_of_fun[function]:
                            dep_of_fun[function].append(dependency)
                    else:
                        dep_of_fun[function] = [dependency]
                    line = line[dep_match.end(1):]
                    dep_match = re_.search('(\w+)\(', line)
        else:
            function = fun_match.group(2)
            mod_of_fun[function] = module
            if module in funs_in_mod:
                funs_in_mod[module].append(function)
            else:
                funs_in_mod[module] = [function]

function_lst = list(mod_of_fun.keys())

# --- Remove non-function dependencies (e.g., array indexing)
for function in dep_of_fun.keys():
    non_sys_functions = []
    for idx, dependency in enumerate(dep_of_fun[function]):
        if dependency in function_lst:
            non_sys_functions.append(dependency)
    dep_of_fun[function] = non_sys_functions

#import pprint
#pprint.pprint(mod_of_fun)
#pprint.pprint(dep_of_fun)
#pprint.pprint(funs_in_mod)

print('digraph call_graph {\n'
      '    rankdir=LR;\n'
      '    node[shape=Mrecord, color=Blue, fontcolor=Blue];')

for idx, function in enumerate(function_lst):
    is_a_dependency = any(function in dependencies for dependencies in dep_of_fun.values())
    if (dep_of_fun[function].__len__() > 0) or is_a_dependency:
        print(f'    {idx}[label={function+"___"+pl_.Path(mod_of_fun[function]).stem}]')

for function, dependencies in dep_of_fun.items():
    function_node = function_lst.index(function)

    for dependency in dependencies:
        dep_node = function_lst.index(dependency)
        print(f'    {function_node} -> {dep_node}')

print('}')
