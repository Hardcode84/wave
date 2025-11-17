# 🔥 Wave Repository Roast 🔥

*"A comprehensive, brutally honest critique of the Wave codebase"*

## Executive Summary

This codebase is like a beautiful mansion built on a foundation of TODOs, with 834 TODO/FIXME comments scattered throughout. It's impressive, functional, but also a monument to "we'll fix it later" culture.

---

## 🎯 The Big Issues

### 1. The TODO Apocalypse (834 TODOs)

You have **834 TODO/FIXME comments**. That's not a codebase, that's a wishlist with some code attached. Highlights include:

- `TODO: investigate why bytecode deserialization is not working` - Classic "it's broken but we'll ship it anyway"
- `TODO: this logic looks suspicious` - When even the author doesn't trust their own code
- `TODO: unhardcode` - The eternal promise
- `TODO: factor this out` - Code duplication? What code duplication?
- `TODO: Only ELEMS_PER_THREAD == 1` - Three times in the same test file. Copy-paste much?

### 2. The God Class: `WaveCompileOptions`

**116 fields** in a single dataclass. This isn't configuration, it's a configuration *universe*. You've got:
- General options
- Symbol mappings  
- Scheduling options
- Runtime options
- Backend options
- Benchmark options
- Cache options
- Debug options
- Performance options
- Compiler options
- Print options
- ASM backend options

At what point do you admit you need a builder pattern or at least split this into logical groups? This is the configuration equivalent of a Swiss Army knife that's also a car, boat, and airplane.

### 3. The Import Wildcard Massacre

**108 instances** of `from ... import *`. You know what this means? You have no idea what's actually imported in any given file. It's like opening a box of chocolates, except every chocolate is a namespace collision waiting to happen.

```python
from ...lang.global_symbols import *  # What could go wrong?
```

### 4. The "Not Implemented" Graveyard

**73 `NotImplementedError`** raises across 26 files. That's not a feature, that's a roadmap disguised as code:

```python
def aot_execute(self, args, kwargs):
    raise NotImplementedError("AOT execution for wave not implemented yet.")

def eager_execute(self, args, kwargs):
    raise NotImplementedError("Eager execution for wave not implemented yet.")
```

The core execution methods? Not implemented. But hey, at least you're honest about it!

### 5. The Circular Import Workaround Circus

```python
def _get_location_capture_config():
    """Wrapper to avoid circular import with debugging module."""
    from ...support.debugging import get_location_capture_config
    return get_location_capture_config()
```

When your architecture is so tangled that you need wrapper functions just to avoid circular imports, maybe it's time to reconsider your module structure. This is the Python equivalent of "I'll just walk around the building instead of going through the door."

### 6. The 1041-Line Monster

`wave.py` is **1041+ lines**. That's not a file, that's a novel. Single Responsibility Principle? Never heard of it. This file does:
- Tracing
- Compilation
- Optimization
- Scheduling
- Code generation
- Debug handling
- And probably makes coffee

### 7. The Dual Runtime System

You have **TWO** kernel classes (`WaveKernel` and `WaveKernel2`). One uses IREE, one uses a custom execution engine. Which one should users use? Who knows! The documentation doesn't say, and the code comments are equally unhelpful. It's like having two front doors to your house - confusing and unnecessary.

### 8. The "Stub Implementation" Special

```python
# TODO: Add more tests once we have more than a stub implementation.
@pytest.mark.skip(reason="getitem: Currently only stub implementation")
```

Tests that test stubs. Tests that are skipped because they test stubs. The recursion is beautiful.

### 9. The Hardcoded Architecture

`target: str = "gfx942"` - Hardcoded everywhere. Want to support other GPUs? Good luck finding all the places where `gfx942` is baked in. It's like writing "Made in 2024" directly into your code.

### 10. The Power-of-2 Tyranny

Your ASM backend **only supports power-of-2 modulo operations**. Want to do `x % 3`? Nope. `x % 5`? Nope. `x % 7`? Absolutely not. It's like a calculator that only works with even numbers.

From the docs: *"Power-of-2 Constraints: Non-power-of-2 modulo and division operations are not supported"*

This is documented as a "limitation" but really it's just "we didn't implement the general case."

---

## 🐛 Code Quality Issues

### The Suspicious Logic

```python
# TODO: this logic looks suspicious. Specifically, there's no check that
```

When you write a TODO saying your own logic is suspicious, maybe just... fix it? Or at least add the check?

### The Debug Print Cleanup

```python
# remove debug print
```

Commit messages like this suggest debug prints are just... everywhere. And someone had to manually remove them. Multiple times. Maybe add a logging framework?

### The "Bug:" Comments

```python
), "Bug: Read Write for a same shared space has different access pattern."
), "Bug: Consumer node and producer node should never be None."
), "Bug: signal the same barId twice before any waits."
```

Assertions with "Bug:" in the message. If you know it's a bug, why is it still there? These are like warning labels on a product: "Warning: This code has bugs."

### The Unhardcoded Hardcode

```python
unsigned optimizeSize = 0; // TODO: unhardcode
```

A hardcoded zero with a TODO to unhardcode it. The irony is delicious.

### The Broken Bytecode

```python
# TODO: investigate why bytecode deserialization is not working
# Serialize MLIR module to text if needed
```

You know bytecode deserialization is broken, so you just... serialize to text instead. That's not fixing the problem, that's working around it forever.

---

## 🏗️ Architecture Concerns

### The Pass Pipeline Spaghetti

Your compilation pipeline has passes that depend on other passes that depend on other passes. It's like a Jenga tower - remove one piece and everything falls apart. And good luck understanding the order without reading the entire `wave.py` file.

### The Constraint System

Constraints are everywhere, but they're also kind of... optional? Some code paths use them, some don't. Some validate them, some just assume they're correct. It's like having traffic lights that sometimes work.

### The Two Backends

You have two backends (`llvm` and `asm`), but the code paths are so different that they might as well be separate projects. Code duplication? Check. Inconsistent behavior? Check. Maintenance nightmare? Double check.

### The Scheduling System

You have manual scheduling, automatic scheduling, constraint-based scheduling, and probably scheduling for your scheduling. It's scheduling all the way down.

---

## 🧪 Testing "Excellence"

### Skipped Tests

Multiple tests marked as `@pytest.mark.skip` or `@pytest.mark.xfail`. Tests that are expected to fail. Tests that are skipped because they test incomplete features. It's like having a restaurant where half the menu says "not available."

### The CI Failure Workaround

```python
# TODO: Investigate why specific CI machine fail for below case.
```

When your tests fail on CI but you don't know why, so you just... skip them? That's not debugging, that's giving up.

### The Comparison TODO

```python
# TODO: switch to comparison against generated iree_ref
```

Tests that don't actually test what they should. They're placeholders for real tests.

---

## 📚 Documentation Issues

### The Unstable API

```python
"""
The API and semantics of this operation are not yet stable, but since it is 
just a debugging tool, you want to take any debug logging out of your kernel 
before shipping it anyway.
"""
```

An API that's explicitly unstable, but it's fine because "it's just debugging." That's like saying "this bridge is unstable, but it's fine because it's just for testing."

### The Missing Documentation

Your `WaveKernel2` class? Barely documented. The difference between `WaveKernel` and `WaveKernel2`? Not explained. Which runtime to use when? Good luck figuring that out.

---

## 🎨 Design Choices

### The Global Symbols Import

Every file imports `from ...lang.global_symbols import *`. This means every file has access to every global symbol. It's like having a global variable namespace that everyone can read and write to. What could possibly go wrong?

### The Optional Everything

Half your function signatures have `Optional[...]` types. It's like every function is saying "maybe I'll work, maybe I won't, who knows?"

### The Type Annotations

Some files have great type hints. Others have... `Any` everywhere. It's like having a codebase where some parts are strongly typed and others are "trust me, it works."

---

## 🚀 Performance "Features"

### The Debug-Only Code Paths

```python
if NDEBUG or isinstance(node, fx.Node):
```

Debug-only assertions scattered throughout. In release builds, these checks disappear. So if something breaks in production, good luck debugging it.

### The Profiling Overhead

You have profiling built into the core execution path. Every kernel invocation can be profiled. That's great for debugging, but what about the overhead? Is it always on? Can it be disabled? The code doesn't make it clear.

---

## 💡 The Silver Linings

Despite all this, the codebase *works*. It compiles kernels, it runs them, and it produces results. The architecture, while complex, is functional. The optimization passes are sophisticated. The MLIR integration is solid.

But it's like a race car that's held together with duct tape - it goes fast, but you're not sure if it'll survive the next turn.

---

## 🎯 Recommendations

1. **Reduce the TODO count by 50%** - Either implement or remove. No more "we'll do it later."
2. **Split `WaveCompileOptions`** - Use composition, not a 116-field monolith.
3. **Eliminate wildcard imports** - Explicit is better than implicit.
4. **Document the dual runtime system** - Users need to know which to use.
5. **Fix or remove `NotImplementedError`** - Either implement the feature or remove the code.
6. **Refactor `wave.py`** - 1041 lines is too much for one file.
7. **Standardize error handling** - Some places use exceptions, some use assertions, some just print.
8. **Add integration tests** - Test the whole pipeline, not just individual pieces.
9. **Document the pass pipeline** - The order matters, but it's not obvious why.
10. **Consolidate the backends** - Or at least document when to use which.

---

## 🏁 Final Thoughts

This codebase is the result of rapid development, feature additions, and "we'll refactor later" promises. It's functional, it's impressive, but it's also a maintenance nightmare waiting to happen.

The good news? Most of these issues are fixable. The bad news? Fixing them will take time, and there are 834 TODOs suggesting you don't have time.

But hey, at least you're honest about the problems. That's more than can be said for most codebases.

**Rating: 7/10** - Works great, but you'll need a map to navigate it.

---

*"Code is like humor. When you have to explain it, it's bad."* - This codebase needs a lot of explaining.
