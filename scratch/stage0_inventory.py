import os, subprocess, hashlib, json, sys

def list_all_files(root):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            files.append(os.path.relpath(os.path.join(dirpath, f), root))
    return files

def write_list(files, out_path):
    with open(out_path, 'w') as f:
        for p in files:
            f.write(p + '\n')

def git_index(out_path):
    # capture git ls-files with mode and hash
    result = subprocess.run(['git','ls-files','--stage'], cwd=root, capture_output=True, text=True)
    with open(out_path, 'w') as f:
        f.write(result.stdout)

def compute_hashes(root, out_path):
    with open(out_path, 'w') as f:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                fp = os.path.join(dirpath, fn)
                # compute sha256
                h = hashlib.sha256()
                with open(fp, 'rb') as fh:
                    while True:
                        chunk = fh.read(8192)
                        if not chunk:
                            break
                        h.update(chunk)
                rel = os.path.relpath(fp, root)
                f.write(f"{h.hexdigest()}  {rel}\n")

if __name__ == '__main__':
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # 0.1 all files
    all_files = list_all_files(root)
    write_list(all_files, os.path.join(root, 'audit', 'all_files.txt'))
    # 0.2 git index
    git_index(os.path.join(root, 'audit', 'git_index.txt'))
    # 0.3 hashes
    compute_hashes(root, os.path.join(root, 'audit', 'file_hashes.sha256'))
    # 0.4 env
    env_out = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True)
    with open(os.path.join(root, 'audit', 'ENVIRONMENT.md'), 'w') as f:
        f.write(env_out.stdout)
    # 0.5 commit id
    commit = subprocess.run(['git','rev-parse','HEAD'], cwd=root, capture_output=True, text=True).stdout.strip()
    with open(os.path.join(root, 'audit','COMMIT_ID.txt'),'w') as f:
        f.write(commit + '\n')

