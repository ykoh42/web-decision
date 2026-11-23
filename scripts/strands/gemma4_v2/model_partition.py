"""Partition model Relax operations into small, independently shaped functions."""

import ast

import tvm
import tvm_ffi
from tvm import relax
from tvm.relax.expr_functor import PyExprMutator


@relax.expr_functor.mutator
class ReplaceGlobals(PyExprMutator):
    def __init__(self, module, mapping):
        super().__init__(module)
        self.mapping = mapping

    def visit_global_var_(self, value):
        return self.mapping.get(value, value)


def model_class(tree):
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if len(classes) != 1 or classes[0].name != "Module":
        raise ValueError("Unexpected model TVMScript module structure")
    return classes[0]


def call_binding(node, annotations):
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        annotations[node.target.id] = ast.unparse(node.annotation)
        if isinstance(node.value, ast.Call) and ast.unparse(node.value.func).startswith(
            "R."
        ):
            return node, node.target.id, node.value
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return None
    if not isinstance(node.targets[0], ast.Name) or not isinstance(
        node.value, ast.Call
    ):
        return None
    if ast.unparse(node.value.func) != "R.call_tir":
        return None
    output = next((kw.value for kw in node.value.keywords if kw.arg == "out_ty"), None)
    if output is None:
        raise ValueError("Model call_tir is missing its output type")
    annotations[node.targets[0].id] = ast.unparse(output)
    return node, node.targets[0].id, node.value


def model_bindings(function):
    annotations = {arg.arg: ast.unparse(arg.annotation) for arg in function.args.args}
    bindings = []
    for node in ast.walk(function):
        binding = call_binding(node, annotations)
        if binding is not None:
            bindings.append(binding)
    return annotations, sorted(bindings, key=lambda binding: binding[0].lineno)


def partition_binding(lines, node, target, value, annotations):
    variables = []
    for ref in ast.walk(value):
        if (
            isinstance(ref, ast.Name)
            and ref.id in annotations
            and ref.id not in variables
        ):
            variables.append(ref.id)
    helper = "part_" + target
    arguments = ", ".join(f"{v}: {annotations[v]}" for v in variables)
    cls_binding = (
        "        cls = Module\n"
        if any(isinstance(ref, ast.Name) and ref.id == "cls" for ref in ast.walk(value))
        else ""
    )
    definition = (
        f"    @R.function\n    def {helper}({arguments}) -> {annotations[target]}:\n"
        f"{cls_binding}        return {ast.unparse(value)}\n"
    )
    original = lines[node.lineno - 1]
    lhs = original[: original.index(" = ")]
    lines[node.lineno - 1] = lhs + " = cls." + helper + "(" + ", ".join(variables) + ")"
    return definition


def partition_script(module, name):
    source = module.script()
    lines = source.splitlines()
    cls = model_class(ast.parse(source))
    function = next(
        (
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ),
        None,
    )
    if function is None:
        raise ValueError(f"Missing model function {name}")
    annotations, bindings = model_bindings(function)
    if not bindings:
        raise ValueError(f"No partitionable model operations in {name}")
    helpers = [
        partition_binding(lines, node, target, value, annotations)
        for node, target, value in bindings
    ]
    source = "\n".join(lines) + "\n"
    source = source.replace(
        "class Module:\n", "class Module:\n" + "\n".join(helpers) + "\n", 1
    )
    if not any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "cls" for t in node.targets)
        for node in function.body
    ):
        source = source.replace(
            "        with R.dataflow():\n",
            "        cls = Module\n        with R.dataflow():\n",
            1,
        )
    return source


def partial_relax(source, name):
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    cls = model_class(tree)
    selected = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and (node.name == name or node.name.startswith("part_"))
    ]
    globals_ = [
        lines[node.lineno - 1]
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "TypeVar"
    ]
    refs = {
        node.attr
        for function in selected
        for node in ast.walk(function)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "cls"
    }
    stubs = sorted(refs - {function.name for function in selected})
    snippets = []
    for function in selected:
        first = (
            function.decorator_list[0].lineno
            if function.decorator_list
            else function.lineno
        )
        snippets.append("".join(lines[first - 1 : function.end_lineno]))
    for stub in stubs:
        snippets.append(f"    @T.prim_func\n    def {stub}():\n        T.evaluate(0)\n")
    return (
        "".join(globals_)
        + "\n@I.ir_module(check_well_formed=False)\nclass Module:\n"
        + "\n".join(snippets)
    )


def partition_model(module, name, *, preserve_tir=False):
    source = partition_script(module, name)
    if name == "main" and not preserve_tir:
        result = tvm.script.from_source(source)
    else:
        result = attach_relax(
            module,
            tvm.script.from_source(
                partial_relax(source, name), check_well_formed=False
            ),
        )
    relax.analysis.well_formed(result)
    if not tvm_ffi.structural_equal(module[name].ty, result[name].ty):
        raise ValueError(f"Partition changed the {name} signature")
    return result


def attach_relax(original, part):
    builder = relax.BlockBuilder(original)
    mapping = {}
    originals = {gv.name_hint: gv for gv in original.get_global_vars()}
    for gv, func in part.functions_items():
        mapping[gv] = (
            originals[gv.name_hint]
            if gv.name_hint in originals
            else builder.add_func(func, gv.name_hint)
        )
    mutator = ReplaceGlobals(builder.get(), mapping)
    for gv, func in part.functions_items():
        if isinstance(func, relax.Function):
            builder.update_func(mapping[gv], mutator.visit_expr(func))
    return builder.get()
