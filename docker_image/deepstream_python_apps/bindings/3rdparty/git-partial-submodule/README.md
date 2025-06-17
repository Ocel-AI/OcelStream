# Partial Submodules for Git

**git-partial-submodule** is a command-line script for setting up and working with submodules while
enabling them to use git's [partial clone](https://github.blog/2020-12-21-get-up-to-speed-with-partial-clone-and-shallow-clone/)
and [sparse checkout](https://github.blog/2020-01-17-bring-your-monorepo-down-to-size-with-sparse-checkout/)
features.

In top-level repositories, git provides various partial-clone options such as "blobless" clone, which
reduce the burden of downloading large repositories. For submodules, only "shallow" clones are
supported by git, but shallow clones have usability issues that make the resulting repo difficult to
work with. **git-partial-submodule** clones your submodules as blobless repos, leaving them fully
functional while saving bandwidth and disk space compared to a full clone.

Similarly, top-level repositories support sparse checkout, which lets you cut down the worktree to
just the files you're interested in. This is particularly relevant for submodules, as their
repositories often contain extra contents such as tests, examples, ancillary tools, and suchlike
that we don't need if we just want to use the submodule as a library in our project.
**git-partial-submodule** stores sparse checkout patterns in `.gitmodules`, so they can be managed
under version control and automatically applied when the submodules are cloned.

## Prerequisites

* git 2.27.0 or later
* Python 3.8 or later

## Installation

**git-partial-submodule** is a single-file Python script. [Download the script](https://github.com/Reedbeta/git-partial-submodule/blob/main/git-partial-submodule.py)
and put it somewhere in your `PATH`, or add it to your repository. Or add this repository as a
submodule.

## Usage

```
git-partial-submodule.py add [-b BRANCH] [--name NAME] [--sparse] <repository> <path>
git-partial-submodule.py clone [<path>...]
git-partial-submodule.py save-sparse [<path>...]
git-partial-submodule.py restore-sparse [<path>...]
```

### Add

Creates and clones a new submodule, much like `git submodule add`, but performs a blobless clone.
If `--sparse` is specified, also enables sparse checkout on the new submodule, with the default
pattern set of `/* !/*/`.

### Clone

Use this to initialize submodules after a fresh clone of the superproject. Performs blobless clones
of any submodules that are not already cloned. Also applies any sparse checkout patterns specified
in `.gitmodules`.

### Save-Sparse

After making changes to the sparse patterns in a submodule, use this to save them to `.gitmodules`.
Patterns are stored space-delimited in the `sparse-checkout` property.

### Restore-Sparse

Reapplies the sparse patterns saved in `.gitmodules` to the actual submodules. Use this after
pulling or switching branches, etc.

## Limitations and Cautions

Partial clone and sparse checkout are both still experimental git features that may have sharp
edges.

This tool works by fiddling with the internals of your repository in not-officially-supported ways,
so it might fail or do the wrong thing in some edge cases I haven't considered (and might leave your
repo in a bad state afterward).

Not all of the various command-line options to the underlying `git clone`, `git submodule add`, etc.
are supported. In particular, recursive clone is not currently supported.

"Cone" mode for sparse checkout is not currently supported.

Spaces in sparse checkout patterns are not currently handled correctly.
