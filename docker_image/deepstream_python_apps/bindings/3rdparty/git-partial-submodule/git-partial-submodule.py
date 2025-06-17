#!/usr/bin/python3
#coding: utf-8

# Copyright 2021 Nathan Reed
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# For more information, see https://github.com/Reedbeta/git-partial-submodule

# Check Python version
import sys
if sys.hexversion < 0x03080000:
	sys.exit('You need at least Python 3.8. (You have %s.)' % sys.version.split()[0])

import argparse, codecs, configparser, os, re, subprocess

# Parse arguments

parser = argparse.ArgumentParser(description="Add or clone partial git submodules; save and restore sparse-checkout patterns.")
parser.add_argument('-n', '--dry-run', dest='dryRun', default=False, action='store_true', help='Dry run (display git commands without executing them)')
parser.add_argument('-v', '--verbose', dest='verbose', default=False, action='store_true', help='Verbose (display git commands being run, and other info)')
subparsers = parser.add_subparsers(dest='command', metavar='command')

addCmdParser = subparsers.add_parser(
    'add',
    help = "Add a new partial submodule.")
addCmdParser.add_argument('-b', '--branch', dest='branch', help='Branch in the submodule repository to check out')
addCmdParser.add_argument('--name', dest='name', help='Logical name for the submodule')
addCmdParser.add_argument('--sparse', dest='sparse', default=False, action='store_true', help='Enable sparse checkout in the submodule')
addCmdParser.add_argument('repository', help='URL to the git repository to be added as a submodule')
addCmdParser.add_argument('path', help='Directory where the submodule will be checked out')

cloneCmdParser = subparsers.add_parser(
    'clone',
    help = "Clone partial submodules from .gitmodules.")
cloneCmdParser.add_argument('paths', nargs='*', default=[], help='Submodule path(s) to clone (if unspecified, all submodules)')
# TODO: recursive clone option

saveSparseCmdParser = subparsers.add_parser(
    'save-sparse',
    help = "Save sparse-checkout patterns to .gitmodules.")
saveSparseCmdParser.add_argument('paths', nargs='*', default=[], help='Submodule path(s) to save (if unspecified, all submodules)')

restoreSparseCmdParser = subparsers.add_parser(
    'restore-sparse',
    help = "Restore sparse-checkout patterns from .gitmodules.")
restoreSparseCmdParser.add_argument('paths', nargs='*', default=[], help='Submodule path(s) to restore (if unspecified, all submodules)')

args = parser.parse_args()

if args.command is None:
    parser.print_help()
    sys.exit()

# Helper functions

def Git(*gitArgs, okReturnCodes = [0]):
    if args.verbose or args.dryRun:
        print('git ' + ' '.join(gitArgs))
    if args.dryRun:
        return
    cp = subprocess.run(('git',) + gitArgs)
    if cp.returncode not in okReturnCodes:
        sys.exit("Git command failed: git %s" % ' '.join(gitArgs))
    return cp

def ReadGitOutput(*gitArgs, okReturnCodes = [0]):
    # Not respecting verbose or dryRun here since this is just used for querying things, not modifying anything
    cp = subprocess.run(('git',) + gitArgs, stdout=subprocess.PIPE)
    if cp.returncode not in okReturnCodes:
        sys.exit("Git command failed: git %s" % ' '.join(gitArgs))
    return codecs.decode(cp.stdout, sys.stdout.encoding)

def CheckGitVersion(minVersion):
    if m := re.match(r'git version (\d+)\.(\d+)\.(\d+)', ReadGitOutput('--version')):
        gitVersion = tuple(int(m.group(i)) for i in range(1, 4))
    if gitVersion < minVersion:
        sys.exit("Git version is too old. You need at least %d.%d.%d, and you have %d.%d.%d." % (minVersion + gitVersion))

def ReadGitmodules(worktreeRoot):
    # Read the .gitmodules file
    gitmodulesConfig = configparser.ConfigParser(
        allow_no_value = True,
        default_section = None,
        inline_comment_prefixes = ('#', ';'),
        interpolation = None)
    if not gitmodulesConfig.read(os.path.join(worktreeRoot, '.gitmodules')):
        sys.exit("Couldn't parse .gitmodules!")
    if args.verbose:
        print("parsed %d submodules from .gitmodules" % len(gitmodulesConfig.sections()))

    # Build mapping tables
    gitmodules = argparse.Namespace(byName={}, byPath={})
    for section in gitmodulesConfig.sections():
        if m := re.match(r'submodule "(.*)"', section):
            name = m.group(1)
            contents = dict(gitmodulesConfig[section])
            contents['name'] = name
            gitmodules.byName[name] = contents
            if 'path' in contents:
                gitmodules.byPath[contents['path']] = contents
    return gitmodules

# Version 2.27.0 is needed for --filter and --sparse options on git clone
CheckGitVersion((2, 27, 0))

# Locate directories
worktreeRoot = os.path.abspath(ReadGitOutput('rev-parse', '--show-toplevel').strip())
repoRoot = os.path.abspath(ReadGitOutput('rev-parse', '--git-dir').strip())
if args.verbose:
    print("worktree root: %s\nrepo root: %s" % (worktreeRoot, repoRoot))

# Process commands

if args.dryRun:
    print("DRY RUN:")

if args.command == 'add':
    # Make passed-in submodule path relative to the worktree root
    submoduleRelPath = (os.path.relpath(os.path.abspath(args.path), worktreeRoot)
                            .replace('\\', '/'))    # Git always uses forward slashes

    # Determine submodule name from the relative path if name is not provided
    submoduleName = args.name or submoduleRelPath

    # Find the submodule's repo directory under the superproject's .git/modules/
    submoduleRepoRoot = os.path.join(repoRoot, 'modules', os.path.normpath(submoduleName))
    if os.path.isdir(submoduleRepoRoot):
        sys.exit("submodule %s repo already exists!" % submoduleName)

    # Find the submodule's worktree directory
    submoduleWorktreeRoot = os.path.join(worktreeRoot, os.path.normpath(submoduleRelPath))
    if os.path.isdir(submoduleWorktreeRoot) and any(os.scandir(submoduleWorktreeRoot)):
        sys.exit("%s submodule worktree is nonempty!" % args.path)

    # Make sure the submodule worktree-to-be is empty in the index as well
    # (otherwise the final git submodule add will fail)
    if ReadGitOutput('-C', worktreeRoot, 'ls-files', '--cached', submoduleRelPath).strip():
        sys.exit("%s submodule worktree is nonempty in the index!\nYou might need to `git rm` that directory first." % args.path)

    # Create directories if necessary
    if not args.dryRun:
        os.makedirs(os.path.dirname(submoduleRepoRoot), exist_ok=True)
        os.makedirs(submoduleWorktreeRoot, exist_ok=True)   # Should have been created by 'git submodule init', but just make sure

    # Perform the partial clone!!!
    Git('clone',
        '--filter=blob:none',
        '--no-checkout',
        '--separate-git-dir', submoduleRepoRoot,
        *(['--branch', args.branch] if args.branch else []),
        *(['--sparse'] if args.sparse else []),
        args.repository,
        submoduleWorktreeRoot)

    # Checkout the submodule
    Git('-C', submoduleWorktreeRoot, 'checkout', *([args.branch] if args.branch else []))

    # Set core.worktree config on the submodule, as for some reason neither the clone nor the checkout does so
    # TODO: normal submodule checkouts in the primary worktree set this to a relative path,
    # but we're always setting an absolute path. Does it matter?
    Git('-C', submoduleWorktreeRoot, 'config', 'core.worktree',
        submoduleWorktreeRoot.replace('\\', '/'))       # Git always uses forward slashes

    # Do the submodule add, which will pick up the now existing repository
    Git('-C', worktreeRoot,
        'submodule', 'add',
        *(['-b', args.branch] if args.branch else []),
        *(['--name', args.name] if args.name else []),
        args.repository,
        submoduleRelPath)

    # TODO: if sparse, save the initial set of sparse patterns to .gitmodules

elif args.command == 'clone':
    # Load .gitmodules information
    gitmodules = ReadGitmodules(worktreeRoot)

    # Init the submodules - this ensures git config "submodule.foo" options are set up
    Git('submodule', 'init', *args.paths)

    if args.paths:
        # Make passed-in submodule paths relative to the worktree root
        submoduleRelPathsToProcess = [os.path.relpath(os.path.abspath(path), worktreeRoot)
                                        .replace('\\', '/')     # Git always uses forward slashes
                                        for path in args.paths]
    else:
        submoduleRelPathsToProcess = gitmodules.byPath.keys()

    # TODO: filter by active submodules

    submodulesSkipped = 0
    for submoduleRelPath in submoduleRelPathsToProcess:
        if submoduleRelPath not in gitmodules.byPath:
            sys.stderr.write("Couldn't find %s in .gitmodules! Skipping.\n" % submoduleRelPath)
            submodulesSkipped += 1
            continue
        submodule = gitmodules.byPath[submoduleRelPath]

        # Find the submodule's repo directory under the superproject's .git/modules/
        submoduleRepoRoot = os.path.join(repoRoot, 'modules', os.path.normpath(submodule['name']))
        if os.path.isdir(submoduleRepoRoot) and any(os.scandir(submoduleRepoRoot)):
            if args.verbose:
                print("submodule %s repo already exists; skipping" % submodule['name'])
            submodulesSkipped += 1
            continue

        # Find the submodule's worktree directory
        submoduleWorktreeRoot = os.path.join(worktreeRoot, os.path.normpath(submoduleRelPath))
        if os.path.isdir(submoduleWorktreeRoot) and any(os.scandir(submoduleWorktreeRoot)):
            sys.stderr.write("%s submodule worktree is nonempty! Skipping.\n" % submoduleRelPath)
            submodulesSkipped += 1
            continue

        # Create directories if necessary
        if not args.dryRun:
            os.makedirs(os.path.dirname(submoduleRepoRoot), exist_ok=True)
            os.makedirs(submoduleWorktreeRoot, exist_ok=True)   # Should have been created by 'git submodule init', but just make sure

        # Perform the partial clone!!!
        Git('clone',
            '--filter=blob:none',
            '--no-checkout',
            '--separate-git-dir', submoduleRepoRoot,
            *(['--branch', branchName] if (branchName := submodule.get('branch')) else []),
            submodule['url'],
            submoduleWorktreeRoot)

        # Apply sparse-checkout patterns in the submodule worktree
        # TODO: support "cone" mode?
        if sparseCheckoutPatterns := submodule.get('sparse-checkout'):
            Git('-C', submoduleWorktreeRoot, 'sparse-checkout', 'init')
            # Split patterns by whitespace - TODO: support quoted paths with embedded spaces etc?
            Git('-C', submoduleWorktreeRoot, 'sparse-checkout', 'set', *sparseCheckoutPatterns.split())
            print("Applied sparse-checkout patterns: %s" % sparseCheckoutPatterns)

        # Retrieve the commit sha1 that the submodule is supposed to be at
        treeInfo = ReadGitOutput('-C', worktreeRoot, 'ls-tree', 'HEAD', submoduleRelPath).split()
        if len(treeInfo) != 4:
            sys.exit("git ls-tree produced unexpected output:\n%s" % ' '.join(treeInfo))
        submoduleCommit = treeInfo[2]
        if args.verbose:
            print("%s submodule sha1 is %s" % (submodule['name'], submoduleCommit))

        # Checkout the submodule
        checkoutArgs = ['--detach', submoduleCommit]
        if (branchName := submodule.get('branch')) and not args.dryRun:
            # Retrieve the commit sha1 of the current branch head
            branchHeadCommit = ReadGitOutput('-C', submoduleWorktreeRoot, 'rev-parse', branchName).strip()
            if args.verbose:
                print("%s branch %s is at sha1 %s" % (submoduleRelPath, branchName, branchHeadCommit))
            # Checkout the branch head directly if it matches the submodule pointer,
            # rather than putting the submodule in a detached head state.
            if branchHeadCommit == submoduleCommit:
                checkoutArgs = [branchName]
        Git('-C', submoduleWorktreeRoot, 'checkout', *checkoutArgs)

        # Set core.worktree config on the submodule, as for some reason neither the clone nor the checkout does so
        # TODO: normal submodule checkouts in the primary worktree set this to a relative path,
        # but we're always setting an absolute path. Does it matter?
        Git('-C', submoduleWorktreeRoot, 'config', 'core.worktree',
            submoduleWorktreeRoot.replace('\\', '/'))       # Git always uses forward slashes

    print("Cloned %d submodules and skipped %d." %
            (len(submoduleRelPathsToProcess) - submodulesSkipped,
             submodulesSkipped))

elif args.command == 'save-sparse':
    # Load .gitmodules information
    gitmodules = ReadGitmodules(worktreeRoot)

    if args.paths:
        # Make passed-in submodule paths relative to the worktree root
        submoduleRelPathsToProcess = [os.path.relpath(os.path.abspath(path), worktreeRoot)
                                        .replace('\\', '/')     # Git always uses forward slashes
                                        for path in args.paths]
    else:
        submoduleRelPathsToProcess = gitmodules.byPath.keys()

    for submoduleRelPath in submoduleRelPathsToProcess:
        if submoduleRelPath not in gitmodules.byPath:
            sys.stderr.write("Couldn't find %s in .gitmodules! Skipping.\n" % submoduleRelPath)
            continue
        submodule = gitmodules.byPath[submoduleRelPath]

        # Find the submodule's worktree directory
        submoduleWorktreeRoot = os.path.join(worktreeRoot, os.path.normpath(submoduleRelPath))
        if not os.path.isdir(submoduleWorktreeRoot) or not any(os.scandir(submoduleWorktreeRoot)):
            sys.stderr.write("%s submodule worktree is empty! Skipping.\n" % submoduleRelPath)
            continue

        # Determine if the submodule has sparse-checkout enabled
        if ReadGitOutput('-C', submoduleWorktreeRoot, 'config', 'core.sparseCheckout',
                            okReturnCodes=[0, 1]).strip() == 'true':    # code 1 = missing key, which means false here
            # Retrieve the sparse-checkout patterns
            sparsePatterns = ReadGitOutput('-C', submoduleWorktreeRoot, 'sparse-checkout', 'list').strip()
            # Save to the .gitmodules file
            Git('-C', worktreeRoot, 'config', '-f', '.gitmodules',
                'submodule.%s.sparse-checkout' % submodule['name'],
                sparsePatterns.replace('\n', ' '))
            print("Saved sparse-checkout patterns for %s." % submodule['name'])

        else:   # sparse-checkout not enabled
            # Unset it in the .gitmodules file
            Git('-C', worktreeRoot, 'config', '-f', '.gitmodules',
                '--unset', 'submodule.%s.sparse-checkout' % submodule['name'],
                okReturnCodes=[0, 5])   # code 5 = "you try to unset an option which does not exist"
            print("Sparse checkout not enabled for %s." % submodule['name'])

elif args.command == 'restore-sparse':
    # Load .gitmodules information
    gitmodules = ReadGitmodules(worktreeRoot)

    if args.paths:
        # Make passed-in submodule paths relative to the worktree root
        submoduleRelPathsToProcess = [os.path.relpath(os.path.abspath(path), worktreeRoot)
                                        .replace('\\', '/')     # Git always uses forward slashes
                                        for path in args.paths]
    else:
        submoduleRelPathsToProcess = gitmodules.byPath.keys()

    for submoduleRelPath in submoduleRelPathsToProcess:
        if submoduleRelPath not in gitmodules.byPath:
            sys.stderr.write("Couldn't find %s in .gitmodules! Skipping.\n" % submoduleRelPath)
            continue
        submodule = gitmodules.byPath[submoduleRelPath]

        # Find the submodule's worktree directory
        submoduleWorktreeRoot = os.path.join(worktreeRoot, os.path.normpath(submoduleRelPath))
        if not os.path.isdir(submoduleWorktreeRoot) or not any(os.scandir(submoduleWorktreeRoot)):
            sys.stderr.write("%s submodule worktree is empty! Skipping.\n" % submoduleRelPath)
            continue

        # Determine if the submodule should have sparse-checkout enabled
        # TODO: support "cone" mode?
        if sparseCheckoutPatterns := submodule.get('sparse-checkout'):
            Git('-C', submoduleWorktreeRoot, 'sparse-checkout', 'init')
            # Split patterns by whitespace - TODO: support quoted paths with embedded spaces etc?
            Git('-C', submoduleWorktreeRoot, 'sparse-checkout', 'set', *sparseCheckoutPatterns.split())
            print("Applied sparse-checkout patterns for %s." % submodule['name'])

        else:   # sparse-checkout not enabled
            Git('-C', submoduleWorktreeRoot, 'sparse-checkout', 'disable')
            print("Sparse checkout disabled for %s." % submodule['name'])
